import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset.CramedDataset import CramedDataset
from dataset.KSDataset import KSDataset_Noise
from models.basic_model import AVClassifier_AUXI_UDML

def parse_noise_type(value):
    normalized = value.strip().lower()
    mapping = {"gaussian": "Gaussian", "salt": "Salt", "none": "None"}
    if normalized not in mapping:
        raise argparse.ArgumentTypeError("noise type must be Gaussian, Salt, or None")
    return mapping[normalized]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an AVClassifier_AUXI_UDML checkpoint under controlled noise."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["CREMAD", "KineticSound"],
        help="Test dataset.",
    )
    parser.add_argument(
        "--pretrained_model",
        "--checkpoint",
        dest="pretrained_model",
        required=True,
        help="Checkpoint produced by main_auxi_weight_udml.py.",
    )
    parser.add_argument(
        "--noise_type",
        type=parse_noise_type,
        default="Gaussian",
        choices=["Gaussian", "Salt", "None"],
        help=(
            "Noise applied to both modalities. Default: Gaussian, matching the "
            "first noisy setting reported by UDML. Use None for clean evaluation."
        ),
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=5.0,
        help=(
            "Shared noise level for both modalities (default: 5, the first noisy "
            "strength reported by UDML). Gaussian uses level*10 as pixel-domain "
            "std; Salt uses level/100 as execution probability."
        ),
    )
    parser.add_argument(
        "--visual_variance",
        type=float,
        default=None,
        help=(
            "Optional visual-specific QMF noise level overriding --noise_level. "
            "This fixed value is used whenever the visual probability gate succeeds."
        ),
    )
    parser.add_argument(
        "--audio_variance",
        type=float,
        default=None,
        help=(
            "Optional audio-specific QMF noise level overriding --noise_level. "
            "This fixed value is used whenever the audio probability gate succeeds."
        ),
    )
    parser.add_argument(
        "--visual_noise_prob",
        type=float,
        default=0.5,
        help="Independent probability of applying visual noise to each test sample.",
    )
    parser.add_argument(
        "--audio_noise_prob",
        type=float,
        default=0.5,
        help="Independent probability of applying audio noise to each test sample.",
    )

    parser.add_argument("--fusion_method", default="concat", choices=["sum", "concat", "gated", "film"])
    parser.add_argument("--pe", type=int, default=1, choices=[0, 1])
    parser.add_argument("--modality", default="full", choices=["full"])
    parser.add_argument("--num_frame", type=int, default=None)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Evaluate only the first N samples. Intended for smoke tests; default uses the full test split.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON result path.")

    parser.add_argument("--audio_path", default="./train_test_data/CREMA-D/AudioWAV")
    parser.add_argument("--visual_path", default="./train_test_data/CREMA-D")
    parser.add_argument("--ks_data_path", default="./train_test_data/kinect_sound")

    # The model reads these fields through its shared args object. Values saved
    # in the checkpoint take precedence after loading.
    parser.add_argument("--audio_depend", type=float, default=32.0)
    parser.add_argument("--visual_depend", type=float, default=10.0)
    parser.add_argument("--drop", type=int, default=0)
    parser.add_argument("--cylcle_epoch", type=int, default=50)
    args = parser.parse_args()

    if args.num_frame is None:
        args.num_frame = 1 if args.dataset == "CREMAD" else 3
    if args.num_frame <= 0:
        parser.error("--num_frame must be positive")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.num_workers < 0:
        parser.error("--num_workers must be non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max_samples must be positive")
    shared_noise_level = 0.0 if args.noise_level is None else args.noise_level
    if args.visual_variance is None:
        args.visual_variance = shared_noise_level
    if args.audio_variance is None:
        args.audio_variance = shared_noise_level
    if args.noise_type == "None":
        args.noise_level = 0.0
        args.visual_variance = 0.0
        args.audio_variance = 0.0

    if shared_noise_level < 0 or args.visual_variance < 0 or args.audio_variance < 0:
        parser.error("noise strengths must be non-negative")
    if not 0.0 <= args.visual_noise_prob <= 1.0:
        parser.error("--visual_noise_prob must be in [0, 1]")
    if not 0.0 <= args.audio_noise_prob <= 1.0:
        parser.error("--audio_noise_prob must be in [0, 1]")
    if args.noise_type == "Salt":
        if not 0.0 <= args.visual_variance <= 100.0:
            parser.error("Salt --visual_variance must be in [0, 100]")
        if not 0.0 <= args.audio_variance <= 100.0:
            parser.error("Salt --audio_variance must be in [0, 100]")

    args.p = [0, 0]
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataset(args):
    if args.dataset == "CREMAD":
        dataset = CramedDataset(
            args,
            mode="test",
            add_noise=args.noise_type != "None",
        )
    else:
        # KSDataset_Noise still uses the removed NumPy alias np.float elsewhere.
        if not hasattr(np, "float"):
            np.float = float
        dataset = KSDataset_Noise(
            args,
            mode="test",
            add_noise=args.noise_type != "None",
            data_path=args.ks_data_path,
        )

    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    return dataset


def select_device(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    if args.device in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_model(args, device):
    checkpoint_path = Path(args.pretrained_model)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        args.audio_depend = float(checkpoint.get("audio_depend", args.audio_depend))
        args.visual_depend = float(checkpoint.get("visual_depend", args.visual_depend))
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a model state_dict")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}

    model = AVClassifier_AUXI_UDML(args)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "Checkpoint architecture does not match the requested --dataset, "
            "--fusion_method, --pe, or --modality settings."
        ) from error

    model.to(device)
    model.eval()
    return model


def evaluate(args, model, dataloader, device):
    correct_fused = 0
    correct_audio = 0
    correct_visual = 0
    total = 0

    with torch.inference_mode():
        for spectrogram, images, label, _, _ in tqdm(dataloader, desc="test"):
            spectrogram = spectrogram.unsqueeze(1).float().to(device, non_blocking=True)
            images = images.float().to(device, non_blocking=True)
            label = label.long().to(device, non_blocking=True)

            outputs = model(spectrogram, images)
            if len(outputs) < 11:
                raise RuntimeError("Model output does not contain fused/audio/visual logits")
            fused_logits = outputs[2]
            audio_logits = outputs[9]
            visual_logits = outputs[10]

            correct_fused += (fused_logits.argmax(dim=1) == label).sum().item()
            correct_audio += (audio_logits.argmax(dim=1) == label).sum().item()
            correct_visual += (visual_logits.argmax(dim=1) == label).sum().item()
            total += label.numel()

    if total == 0:
        raise RuntimeError("The selected test dataset contains no samples")
    return {
        "dataset": args.dataset,
        "split": "test",
        "samples": total,
        "checkpoint": str(Path(args.pretrained_model).resolve()),
        "noise_type": args.noise_type,
        "noise_level": args.noise_level,
        "visual_noise_level": args.visual_variance,
        "audio_noise_level": args.audio_variance,
        "visual_variance": args.visual_variance,
        "audio_variance": args.audio_variance,
        "visual_noise_probability": args.visual_noise_prob,
        "audio_noise_probability": args.audio_noise_prob,
        "fused_accuracy": correct_fused / total,
        "audio_accuracy": correct_audio / total,
        "visual_accuracy": correct_visual / total,
        "seed": args.seed,
    }


def main():
    args = parse_args()
    device = select_device(args)
    set_seed(args.seed)
    dataset = build_dataset(args)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    model = load_model(args, device)
    result = evaluate(args, model, dataloader, device)
    result["device"] = str(device)
    result["num_frame"] = args.num_frame
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
