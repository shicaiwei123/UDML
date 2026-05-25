#!/usr/bin/env python3
"""
eval_udml_noise.py - evaluate train_udml_noise_base.py checkpoints under noise sweeps.

This script reuses src.data.helpers.get_udml_test(...) so evaluation stays aligned
with the UDML training / helper pipeline.
"""

import argparse
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from src.data.helpers import get_labels_and_frequencies, get_udml_test, get_vocab
from src.utils.utils import set_seed
from train_udml_noise_base import MultimodalLateFusionUDML, get_args as train_get_args


def parse_strengths(raw):
    strengths = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            strengths.append(float(part))
    if not strengths:
        raise ValueError("--strengths cannot be empty")
    return strengths


def resolve_checkpoint_path(raw_path):
    if os.path.isdir(raw_path):
        return os.path.join(raw_path, "model_best.pt")
    return raw_path


def resolve_depend_path(checkpoint_path, depend_path):
    if depend_path:
        return depend_path
    return os.path.join(os.path.dirname(checkpoint_path), "model_best_depend.pt")


def prepare_args(args):
    args.labels, args.label_freqs = get_labels_and_frequencies(
        os.path.join(args.data_path, args.task, "train.jsonl")
    )
    args.vocab = get_vocab(args)
    args.vocab_sz = args.vocab.vocab_sz
    args.n_classes = len(args.labels)
    return args


def load_model(checkpoint_path, depend_path, args, device):
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "txt.mu.0.weight" in state_dict:
        args.fusion_dim = int(state_dict["txt.mu.0.weight"].shape[0])
    elif "img.mu.0.weight" in state_dict:
        args.fusion_dim = int(state_dict["img.mu.0.weight"].shape[1])

    model = MultimodalLateFusionUDML(args)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    args.text_depend = 1.0
    if os.path.exists(depend_path):
        dep = torch.load(depend_path, map_location="cpu", weights_only=False)
        if isinstance(dep, dict):
            args.text_depend = float(dep.get("text_depend", 1.0))
            args.visual_depend = float(dep.get("visual_depend", args.visual_depend))

    return model, args


def resolve_helper_levels(strength):
    strength = float(strength)
    if strength <= 0:
        return False, None, None

    txt_noise_level = strength + 1.0
    img_noise_level = strength
    if 0 < img_noise_level <= 1.0:
        img_noise_level = 1.0 + img_noise_level * 1e-4
    return True, txt_noise_level, img_noise_level


def maybe_subset_loader(loader, max_samples):
    if max_samples is None:
        return loader
    subset = Subset(loader.dataset, range(min(max_samples, len(loader.dataset))))
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
    )


def build_loader(args, strength):
    add_noise, txt_noise_level, img_noise_level = resolve_helper_levels(strength)
    loader = get_udml_test(
        args,
        add_noise=add_noise,
        txt_noise_level=txt_noise_level,
        img_noise_level=img_noise_level,
    )["test"]
    return maybe_subset_loader(loader, args.max_samples)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for batch in tqdm(loader, desc="Eval", leave=False):
        text, seg, mask, img, target, _ = [x.to(device) for x in batch[:6]]
        logits = model(text, mask, seg, img)[0]
        preds.append(torch.softmax(logits, 1).argmax(1).cpu().numpy())
        targets.append(target.cpu().numpy())
    return accuracy_score(np.concatenate(targets), np.concatenate(preds))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate UDML checkpoints under different noise strengths."
    )
    train_get_args(parser)
    parser.add_argument("--checkpoint", required=True, help="Path to model_best.pt or its directory.")
    parser.add_argument("--depend", default=None, help="Optional path to model_best_depend.pt.")
    parser.add_argument(
        "--strengths",
        default="0,1,2,3,4,5",
        help="Comma-separated public noise strengths. Text maps to helper tv=strength+1.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional small-sample sanity check size.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    depend_path = resolve_depend_path(checkpoint_path, args.depend)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    args = prepare_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_args = load_model(checkpoint_path, depend_path, args, device)
    strengths = parse_strengths(args.strengths)

    print(f"Loading model from {checkpoint_path} ...")
    print(
        f"  fusion_dim={model_args.fusion_dim}, "
        f"text_depend={model_args.text_depend:.2f}, "
        f"visual_depend={model_args.visual_depend:.2f}"
    )
    print("\nstrength\taccuracy\ttext_depend\tvisual_depend")

    for strength in strengths:
        loader = build_loader(model_args, strength)
        acc = evaluate(model, loader, device)
        shown = int(strength) if float(strength).is_integer() else strength
        print(
            f"{shown}	{acc:.4f}	"
            f"{model_args.text_depend:.2f}	{model_args.visual_depend:.2f}"
        )


if __name__ == "__main__":
    main()
