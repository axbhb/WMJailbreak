#!/usr/bin/env python
"""Download evaluator model weights into a project-controlled HF cache.

By default this stores the Hugging Face cache under F:/WMjailbreak/hf_cache so
large evaluator weights do not land in the user profile cache on C:.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


EVALUATOR_REPOS = {
    "wildguard": "allenai/wildguard",
    "harmbench": "cais/HarmBench-Llama-2-13b-cls",
}

ALLOW_PATTERNS = [
    "*.json",
    "*.model",
    "*.txt",
    "*.md",
    "*.safetensors",
]

IGNORE_PATTERNS = [
    "*.bin",
    "*.h5",
    "*.msgpack",
    "*.onnx",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download WildGuard/HarmBench evaluator models.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("F:/WMjailbreak/hf_cache"),
        help="Hugging Face cache directory for model weights.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(EVALUATOR_REPOS),
        default=["wildguard", "harmbench"],
        help="Evaluator models to download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = args.cache_dir.parent / "evaluator_model_paths.json"
    if manifest.exists():
        results: dict[str, str] = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        results = {}
    for name in args.models:
        repo_id = EVALUATOR_REPOS[name]
        print(f"downloading {name}: {repo_id}", flush=True)
        path = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(args.cache_dir),
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
            resume_download=True,
        )
        results[name] = path
        print(f"downloaded {name} -> {path}", flush=True)

    manifest.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
