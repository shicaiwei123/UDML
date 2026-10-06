#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


def checkpoint_accuracy(path):
    match = re.search(r"_acc_([0-9.]+)\.pth$", path.name)
    return float(match.group(1)) if match else float("-inf")


def checkpoint_epoch(path):
    match = re.search(r"_epoch_(\d+)_acc_", path.name)
    return int(match.group(1)) if match else -1


def find_best_checkpoints(results_root):
    selected = []
    for directory in sorted(path for path in results_root.iterdir() if path.is_dir()):
        checkpoints = list(directory.glob("*.pth"))
        if not checkpoints:
            continue
        best = max(checkpoints, key=checkpoint_accuracy)
        selected.append((directory.name, best))
    return selected


def parse_condition(value):
    noise_type, level = value.split(":", 1)
    if noise_type not in {"Gaussian", "Salt"}:
        raise argparse.ArgumentTypeError("noise type must be Gaussian or Salt")
    return noise_type, float(level)


def write_summary(rows, path):
    fields = [
        "model_version", "checkpoint", "checkpoint_epoch", "checkpoint_named_accuracy",
        "noise_type", "noise_level", "seed", "samples",
        "fused_accuracy", "audio_accuracy", "visual_accuracy", "result_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run the requested CREMAD noise matrix.")
    parser.add_argument("--results-root", type=Path, default=Path("results/cramed"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", type=parse_condition,
                        default=[("Gaussian", 5.0), ("Gaussian", 10.0), ("Salt", 5.0), ("Salt", 10.0)])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--audio-noise-prob", type=float, default=0.5)
    parser.add_argument("--visual-noise-prob", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    if not 0.0 <= args.audio_noise_prob <= 1.0:
        parser.error("--audio-noise-prob must be in [0, 1]")
    if not 0.0 <= args.visual_noise_prob <= 1.0:
        parser.error("--visual-noise-prob must be in [0, 1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = find_best_checkpoints(args.results_root)
    if not selected:
        raise RuntimeError(f"No checkpoints found below {args.results_root}")

    manifest = {
        "dataset": "CREMAD",
        "selection_rule": "highest accuracy encoded in checkpoint filename within each model-version directory",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "audio_noise_probability": args.audio_noise_prob,
        "visual_noise_probability": args.visual_noise_prob,
        "max_samples": args.max_samples,
        "conditions": [{"noise_type": n, "noise_level": l} for n, l in args.conditions],
        "models": [{"model_version": name, "checkpoint": str(path),
                    "checkpoint_epoch": checkpoint_epoch(path),
                    "checkpoint_named_accuracy": checkpoint_accuracy(path)}
                   for name, path in selected],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    rows = []
    total = len(selected) * len(args.conditions)
    index = 0
    for model_version, checkpoint in selected:
        for noise_type, noise_level in args.conditions:
            index += 1
            level_label = f"{noise_level:g}".replace(".", "p")
            result_path = args.output_dir / f"{model_version}__{noise_type.lower()}_{level_label}.json"
            print(f"\n[{index}/{total}] {model_version}: {noise_type} {noise_level:g}", flush=True)
            if not result_path.exists():
                command = [
                    sys.executable, "test.py",
                    "--dataset", "CREMAD",
                    "--pretrained_model", str(checkpoint),
                    "--noise_type", noise_type,
                    "--noise_level", f"{noise_level:g}",
                    "--batch_size", str(args.batch_size),
                    "--num_workers", str(args.num_workers),
                    "--audio_noise_prob", str(args.audio_noise_prob),
                    "--visual_noise_prob", str(args.visual_noise_prob),
                    "--seed", str(args.seed),
                    "--output", str(result_path),
                ]
                if args.max_samples is not None:
                    command.extend(["--max_samples", str(args.max_samples)])
                subprocess.run(command, check=True)
            else:
                print(f"Reusing {result_path}", flush=True)

            result = json.loads(result_path.read_text(encoding="utf-8"))
            rows.append({
                "model_version": model_version,
                "checkpoint": str(checkpoint),
                "checkpoint_epoch": checkpoint_epoch(checkpoint),
                "checkpoint_named_accuracy": checkpoint_accuracy(checkpoint),
                "noise_type": noise_type,
                "noise_level": noise_level,
                "seed": args.seed,
                "samples": result["samples"],
                "fused_accuracy": result["fused_accuracy"],
                "audio_accuracy": result["audio_accuracy"],
                "visual_accuracy": result["visual_accuracy"],
                "result_json": str(result_path),
            })
            write_summary(rows, args.output_dir / "noise_matrix.csv")

    print(f"\nCompleted {len(rows)} evaluations. Summary: {args.output_dir / 'noise_matrix.csv'}", flush=True)


if __name__ == "__main__":
    main()
