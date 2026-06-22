#!/usr/bin/env python
"""Repair JSONL files that contain concatenated JSON objects on one line.

The annotator writes one JSON object per line, but interrupted Windows runs can
occasionally concatenate two objects on one physical line. This utility parses
all JSON objects, emits canonical JSONL, and optionally deduplicates by
``(dialogue_id, turn)``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair and deduplicate JSONL objects.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-dedupe", action="store_true")
    return parser.parse_args()


def iter_objects(path: Path):
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            pos = 0
            while pos < len(text):
                while pos < len(text) and text[pos].isspace():
                    pos += 1
                if pos >= len(text):
                    break
                try:
                    obj, end = decoder.raw_decode(text, pos)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at line {line_no}, char {pos}: {exc}") from exc
                yield obj
                pos = end


def key_for(obj: dict[str, Any]) -> tuple[Any, Any]:
    return obj.get("dialogue_id"), obj.get("turn")


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple[Any, Any]] = set()
    written = 0
    skipped = 0

    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        for obj in iter_objects(args.input):
            if not args.no_dedupe:
                key = key_for(obj)
                if key in seen:
                    skipped += 1
                    continue
                seen.add(key)
            out.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
            out.write("\n")
            written += 1

    print(f"written={written} skipped_duplicates={skipped} output={args.output}")


if __name__ == "__main__":
    main()
