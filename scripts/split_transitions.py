#!/usr/bin/env python
"""Split transition JSONL by dialogue_id to avoid multi-turn leakage."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("data/xguard-train/transitions/xguard_transitions_rewards.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/xguard-train/splits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split transition JSONL by dialogue_id.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Reward-labeled transition JSONL.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for split JSONL files.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--prefix", default="", help="Optional filename prefix, for example xguard_.")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc


def validate_ratios(args: argparse.Namespace) -> None:
    ratios = [args.train_ratio, args.val_ratio, args.test_ratio]
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")


def collect_dialogue_ids(path: Path) -> list[str]:
    seen: set[str] = set()
    dialogue_ids: list[str] = []
    for item in iter_jsonl(path):
        dialogue_id = str(item.get("dialogue_id") or "")
        if not dialogue_id:
            raise ValueError("Encountered transition without dialogue_id")
        if dialogue_id not in seen:
            seen.add(dialogue_id)
            dialogue_ids.append(dialogue_id)
    return dialogue_ids


def assign_splits(dialogue_ids: list[str], args: argparse.Namespace) -> dict[str, str]:
    shuffled = list(dialogue_ids)
    random.Random(args.seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * args.train_ratio)
    val_count = int(total * args.val_ratio)

    train_ids = set(shuffled[:train_count])
    val_ids = set(shuffled[train_count : train_count + val_count])
    test_ids = set(shuffled[train_count + val_count :])

    mapping: dict[str, str] = {}
    for dialogue_id in train_ids:
        mapping[dialogue_id] = "train"
    for dialogue_id in val_ids:
        mapping[dialogue_id] = "val"
    for dialogue_id in test_ids:
        mapping[dialogue_id] = "test"
    return mapping


def split_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "train": output_dir / f"{prefix}train.jsonl",
        "val": output_dir / f"{prefix}val.jsonl",
        "test": output_dir / f"{prefix}test.jsonl",
    }


def main() -> None:
    args = parse_args()
    validate_ratios(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dialogue_ids = collect_dialogue_ids(args.input)
    split_by_dialogue = assign_splits(dialogue_ids, args)
    paths = split_paths(args.output_dir, args.prefix)
    stats: Counter[str] = Counter()

    handles = {
        split: path.open("w", encoding="utf-8", newline="\n")
        for split, path in paths.items()
    }
    try:
        for item in iter_jsonl(args.input):
            dialogue_id = str(item.get("dialogue_id") or "")
            split = split_by_dialogue[dialogue_id]
            handles[split].write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            handles[split].write("\n")
            stats[f"{split}_transitions"] += 1
    finally:
        for handle in handles.values():
            handle.close()

    for split in split_by_dialogue.values():
        stats[f"{split}_dialogues"] += 1

    id_payload = {
        "seed": args.seed,
        "input": str(args.input),
        "split_by_dialogue": split_by_dialogue,
    }
    (args.output_dir / f"{args.prefix}dialogue_splits.json").write_text(
        json.dumps(id_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats_payload = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "paths": {split: str(path) for split, path in paths.items()},
        "counts": dict(stats),
    }
    stats_path = args.output_dir / f"{args.prefix}split_stats.json"
    stats_path.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote train: {paths['train']}")
    print(f"wrote val: {paths['val']}")
    print(f"wrote test: {paths['test']}")
    print(f"wrote stats: {stats_path}")
    print(f"dialogues: {len(dialogue_ids)}")
    print(f"transitions: {sum(value for key, value in stats.items() if key.endswith('_transitions'))}")


if __name__ == "__main__":
    main()
