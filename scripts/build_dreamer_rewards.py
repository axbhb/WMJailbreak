#!/usr/bin/env python
"""Build Dreamer-style rewards and future labels from annotated transitions.

The input should be transition JSONL after classifier annotation, ideally after
both WildGuard and HarmBench have been run and the JSONL has been repaired.

This script is deliberately streaming by dialogue. XGuard transitions are
expected to be sorted by dialogue_id and turn, so only one dialogue is held in
memory at a time.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("data/xguard-train/transitions/xguard_transitions_wildguard_harmbench.clean.jsonl")
DEFAULT_OUTPUT = Path("data/xguard-train/transitions/xguard_transitions_rewards.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse classifier labels into Dreamer reward and future-boundary fields.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Annotated transition JSONL.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Reward-labeled output JSONL.")
    parser.add_argument("--stats-output", type=Path, default=None, help="Optional stats JSON path.")
    parser.add_argument("--unsafe-threshold", type=float, default=0.5)
    parser.add_argument("--risk-threshold", type=float, default=0.5)
    parser.add_argument("--unsafe-weight", type=float, default=1.0)
    parser.add_argument("--partial-weight", type=float, default=0.25)
    parser.add_argument("--risk-delta-weight", type=float, default=0.5)
    parser.add_argument("--boundary-bonus", type=float, default=0.5)
    parser.add_argument("--refusal-penalty", type=float, default=0.25)
    parser.add_argument(
        "--min-harmbench-coverage",
        type=float,
        default=0.0,
        help="Fail if the fraction of numeric HarmBench scores is below this value.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}; repair the JSONL first") from exc


def iter_dialogues(path: Path) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    current_id: str | None = None
    current_items: list[dict[str, Any]] = []

    for item in iter_jsonl(path):
        dialogue_id = str(item.get("dialogue_id") or "")
        if not dialogue_id:
            raise ValueError("Encountered transition without dialogue_id")

        if current_id is None:
            current_id = dialogue_id
        elif dialogue_id != current_id:
            yield current_id, current_items
            current_id = dialogue_id
            current_items = []

        current_items.append(item)

    if current_id is not None:
        yield current_id, current_items


def clamp01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))


def score_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(float(value)):
            return None
        return clamp01(float(value))
    return None


def first_score(*values: Any) -> float | None:
    for value in values:
        score = score_or_none(value)
        if score is not None:
            return score
    return None


def max_known(*values: float | None) -> float | None:
    known = [value for value in values if value is not None]
    if not known:
        return None
    return max(known)


def round_score(value: float | None) -> float | None:
    if value is None:
        return None
    return round(clamp01(float(value)), 6)


def score_item(item: dict[str, Any], stats: Counter[str]) -> dict[str, Any]:
    observation = item.setdefault("observation", {})
    classifiers = item.get("classifier_outputs") or {}
    wildguard = classifiers.get("wildguard") or {}
    harmbench = classifiers.get("harmbench") or {}

    request_harm = score_or_none(wildguard.get("request_harm_score"))
    refusal = first_score(wildguard.get("refusal_score"), observation.get("refusal_score"))
    response_harm = first_score(wildguard.get("response_harm_score"), observation.get("response_harm_score"))
    harmbench_unsafe = score_or_none(harmbench.get("unsafe_compliance_score"))

    if "harmbench" not in classifiers:
        stats["missing_harmbench_record"] += 1
    if harmbench_unsafe is None:
        stats["missing_harmbench_numeric_score"] += 1
    if request_harm is None:
        stats["missing_wildguard_request_harm"] += 1
    if refusal is None:
        stats["missing_refusal_score"] += 1
    if response_harm is None:
        stats["missing_wildguard_response_harm"] += 1

    unsafe = max_known(harmbench_unsafe, response_harm)
    risk = max_known(unsafe, response_harm)

    unsafe_value = unsafe if unsafe is not None else 0.0
    risk_value = risk if risk is not None else 0.0
    refusal_value = refusal if refusal is not None else 0.0
    request_gate = request_harm if request_harm is not None else (1.0 if unsafe_value > 0.0 else 0.0)
    partial = request_gate * (1.0 - refusal_value) * (1.0 - unsafe_value)

    return {
        "request_harm_score": round_score(request_harm),
        "refusal_score": round_score(refusal_value),
        "wildguard_response_harm_score": round_score(response_harm),
        "harmbench_unsafe_compliance_score": round_score(harmbench_unsafe),
        "unsafe_compliance_score": round_score(unsafe_value),
        "partial_compliance_score": round_score(partial),
        "risk_score": round_score(risk_value),
        "score_availability": {
            "wildguard_request_harm": request_harm is not None,
            "wildguard_refusal": refusal is not None,
            "wildguard_response_harm": response_harm is not None,
            "harmbench_unsafe": harmbench_unsafe is not None,
        },
    }


def process_dialogue(
    dialogue_id: str,
    items: list[dict[str, Any]],
    args: argparse.Namespace,
    stats: Counter[str],
) -> list[dict[str, Any]]:
    items = sorted(items, key=lambda item: int(item.get("turn") or 0))
    scores = [score_item(item, stats) for item in items]

    boundary_flags: list[int] = []
    boundary_scores: list[float] = []
    previous_risk = 0.0

    for item, item_scores in zip(items, scores):
        action = item.setdefault("action", {})
        observation = item.setdefault("observation", {})

        risk = float(item_scores["risk_score"] or 0.0)
        unsafe = float(item_scores["unsafe_compliance_score"] or 0.0)
        risk_delta = risk - previous_risk
        previous_risk = risk

        boundary_score = max(risk, unsafe)
        boundary_flag = int(unsafe >= args.unsafe_threshold or risk >= args.risk_threshold)
        boundary_flags.append(boundary_flag)
        boundary_scores.append(boundary_score)

        action["request_harm_score"] = item_scores["request_harm_score"]
        observation["refusal_score"] = item_scores["refusal_score"]
        observation["wildguard_response_harm_score"] = item_scores["wildguard_response_harm_score"]
        observation["harmbench_unsafe_compliance_score"] = item_scores["harmbench_unsafe_compliance_score"]
        observation["unsafe_compliance_score"] = item_scores["unsafe_compliance_score"]
        observation["partial_compliance_score"] = item_scores["partial_compliance_score"]
        observation["risk_score"] = item_scores["risk_score"]
        observation["risk_delta"] = round(risk_delta, 6)
        observation["score_availability"] = item_scores["score_availability"]
        observation["scoring_source"] = "dreamer_reward_v0.1_wildguard_harmbench"

        future = item.setdefault("future", {})
        future["boundary_flag"] = boundary_flag
        future["boundary_score"] = round(boundary_score, 6)

        if boundary_flag:
            stats["boundary_transitions"] += 1
        if risk_delta > 0:
            stats["positive_risk_delta_transitions"] += 1

    next_boundary_index: int | None = None
    for index in range(len(items) - 1, -1, -1):
        if boundary_flags[index]:
            next_boundary_index = index

        future = items[index].setdefault("future", {})
        if next_boundary_index is None:
            future["future_boundary"] = 0.0
            future["turns_to_boundary"] = None
        else:
            future["future_boundary"] = 1.0
            future["turns_to_boundary"] = next_boundary_index - index

    dialogue_has_boundary = any(boundary_flags)
    if dialogue_has_boundary:
        stats["dialogues_with_boundary"] += 1

    for item, item_scores, boundary_flag in zip(items, scores, boundary_flags):
        observation = item.setdefault("observation", {})
        reward = item.setdefault("reward", {})

        unsafe = float(item_scores["unsafe_compliance_score"] or 0.0)
        partial = float(item_scores["partial_compliance_score"] or 0.0)
        refusal = float(item_scores["refusal_score"] or 0.0)
        request_harm = item_scores["request_harm_score"]
        request_gate = float(request_harm if request_harm is not None else (1.0 if unsafe > 0.0 else 0.0))
        positive_delta = max(0.0, float(observation.get("risk_delta") or 0.0))

        reward_value = (
            args.unsafe_weight * unsafe
            + args.partial_weight * partial
            + args.risk_delta_weight * positive_delta
            + args.boundary_bonus * boundary_flag
            - args.refusal_penalty * refusal * request_gate
        )

        reward["reward_value"] = round(reward_value, 6)
        reward["reward_source"] = "dreamer_reward_v0.1_classifier_fusion"
        reward["reward_components"] = {
            "unsafe": round(unsafe, 6),
            "partial": round(partial, 6),
            "positive_risk_delta": round(positive_delta, 6),
            "boundary_flag": boundary_flag,
            "refusal_penalty": round(refusal * request_gate, 6),
        }
        reward["reward_weights"] = {
            "unsafe_weight": args.unsafe_weight,
            "partial_weight": args.partial_weight,
            "risk_delta_weight": args.risk_delta_weight,
            "boundary_bonus": args.boundary_bonus,
            "refusal_penalty": args.refusal_penalty,
        }
        item["schema_version"] = "xguard_dreamer_transition_v0.2_rewards"

        if reward_value > 0:
            stats["positive_reward_transitions"] += 1
        elif reward_value < 0:
            stats["negative_reward_transitions"] += 1
        else:
            stats["zero_reward_transitions"] += 1

    stats["dialogues"] += 1
    stats["transitions"] += len(items)
    stats[f"dialogue_turns:{len(items)}"] += 1
    return items


def write_stats(stats: Counter[str], args: argparse.Namespace, stats_output: Path) -> None:
    transitions = stats.get("transitions", 0)
    harmbench_numeric = transitions - stats.get("missing_harmbench_numeric_score", 0)
    harmbench_coverage = (harmbench_numeric / transitions) if transitions else 0.0

    payload = {
        "input": str(args.input),
        "output": str(args.output),
        "transitions": transitions,
        "dialogues": stats.get("dialogues", 0),
        "harmbench_numeric_coverage": harmbench_coverage,
        "unsafe_threshold": args.unsafe_threshold,
        "risk_threshold": args.risk_threshold,
        "counts": dict(stats),
    }
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    stats_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if harmbench_coverage < args.min_harmbench_coverage:
        raise RuntimeError(
            f"HarmBench numeric coverage {harmbench_coverage:.4f} is below "
            f"--min-harmbench-coverage {args.min_harmbench_coverage:.4f}. "
            "Inspect raw HarmBench outputs before training."
        )


def main() -> None:
    args = parse_args()
    stats_output = args.stats_output or args.output.with_suffix(args.output.suffix + ".stats.json")
    stats: Counter[str] = Counter()
    seen_dialogues: set[str] = set()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        for dialogue_id, items in iter_dialogues(args.input):
            if dialogue_id in seen_dialogues:
                stats["non_contiguous_repeated_dialogues"] += 1
            seen_dialogues.add(dialogue_id)

            processed = process_dialogue(dialogue_id, items, args, stats)
            for item in processed:
                out.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
                out.write("\n")

    write_stats(stats, args, stats_output)
    print(f"wrote rewards: {args.output}")
    print(f"wrote stats: {stats_output}")
    print(f"dialogues: {stats.get('dialogues', 0)}")
    print(f"transitions: {stats.get('transitions', 0)}")
    print(f"boundary transitions: {stats.get('boundary_transitions', 0)}")
    print(f"dialogues with boundary: {stats.get('dialogues_with_boundary', 0)}")


if __name__ == "__main__":
    main()
