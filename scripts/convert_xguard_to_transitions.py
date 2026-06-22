#!/usr/bin/env python
"""Convert XGuard-Train conversations into Dreamer-style transition JSONL.

This script is intentionally offline and deterministic. It does not call any
external judge model. The scores it produces are lightweight placeholders that
keep the schema stable until a proper judge ensemble is added.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("data/xguard-train/xguard-train.json")
DEFAULT_OUTPUT = Path("data/xguard-train/transitions/xguard_transitions.jsonl")


REFUSAL_PATTERNS = [
    r"\bi\s+(?:am\s+)?(?:sorry|unable|can't|cannot|won't)\b",
    r"\bi\s+can't\s+(?:help|assist|provide|comply)\b",
    r"\bi\s+cannot\s+(?:help|assist|provide|comply)\b",
    r"\bnot\s+(?:able|appropriate|safe)\s+to\b",
    r"\bagainst\s+(?:policy|guidelines)\b",
    r"\bcan't\s+assist\s+with\b",
    r"\b我(?:不能|无法|不会|不可以)\b",
    r"\b不能(?:帮助|提供|协助)\b",
    r"\b无法(?:帮助|提供|协助)\b",
]

SAFE_ALT_PATTERNS = [
    r"\binstead\b",
    r"\bi\s+can\s+(?:help|provide|offer)\b",
    r"\bsafer\s+alternative\b",
    r"\bhigh-level\b",
    r"\b一般性\b",
    r"\b安全(?:的)?替代\b",
    r"\b可以(?:提供|帮助).*安全\b",
]

CLARIFICATION_PATTERNS = [
    r"\bcan\s+you\s+(?:clarify|provide|explain)\b",
    r"\bcould\s+you\s+(?:clarify|provide|explain)\b",
    r"\bwhat\s+do\s+you\s+mean\b",
    r"\?",
    r"\b你(?:能|可以).*澄清\b",
    r"\b什么意思\b",
    r"？",
]

ACTION_RULES: list[tuple[str, list[str]]] = [
    ("role_context_shift", [r"\bpretend\b", r"\brole[- ]?play\b", r"\bact\s+as\b", r"\bsuppose\b", r"\b扮演\b", r"\b假设\b"]),
    ("decomposition", [r"\bstep[- ]by[- ]step\b", r"\bbreak\s+(?:it|this)\s+down\b", r"\bfirst\b.*\bthen\b", r"\b步骤\b", r"\b分解\b"]),
    ("specificity_increase", [r"\bspecific\b", r"\bexact\b", r"\bdetailed\b", r"\bmore\s+detail\b", r"\b具体\b", r"\b详细\b", r"\b精确\b"]),
    ("procedure_request", [r"\bprocedure\b", r"\bworkflow\b", r"\bhow\s+to\b", r"\binstructions?\b", r"\b流程\b", r"\b教程\b", r"\b怎么\b"]),
    ("tool_or_code_request", [r"\bcode\b", r"\bscript\b", r"\btool\b", r"\bcommand\b", r"\b代码\b", r"\b脚本\b", r"\b工具\b", r"\b命令\b"]),
    ("boundary_probe", [r"\ballowed\b", r"\bpolicy\b", r"\bsafe\b", r"\b合法\b", r"\b允许\b", r"\b政策\b", r"\b安全吗\b"]),
    ("clarification", [r"\bwhat\b", r"\bwhy\b", r"\bwhich\b", r"\bcan\s+you\s+explain\b", r"\b什么\b", r"\b为什么\b", r"\b解释\b"]),
    ("continuation", [r"\bcontinue\b", r"\bnext\b", r"\bmore\b", r"\b继续\b", r"\b然后\b", r"\b下一步\b"]),
]


def compile_patterns(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


REFUSAL_RE = compile_patterns(REFUSAL_PATTERNS)
SAFE_ALT_RE = compile_patterns(SAFE_ALT_PATTERNS)
CLARIFICATION_RE = compile_patterns(CLARIFICATION_PATTERNS)
ACTION_RE = [(label, compile_patterns(patterns)) for label, patterns in ACTION_RULES]


def any_match(regexes: list[re.Pattern[str]], text: str) -> bool:
    return any(regex.search(text) for regex in regexes)


def truncate_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or max_chars < 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


def detect_response_type(response: str) -> tuple[str, float]:
    """Return a heuristic response type and refusal score.

    These values are scaffolding only. Replace them with judge outputs before
    training the final Dreamer world model.
    """
    refusal = any_match(REFUSAL_RE, response)
    safe_alt = any_match(SAFE_ALT_RE, response)
    clarification = any_match(CLARIFICATION_RE, response)

    if refusal and safe_alt:
        return "refusal_with_safe_alternative", 0.9
    if refusal:
        return "refusal", 0.85
    if clarification:
        return "clarification", 0.2
    if safe_alt:
        return "safe_general_answer", 0.25
    return "direct_or_general_answer", 0.05


def detect_abstract_action(user_text: str, turn_index: int) -> str:
    for label, regexes in ACTION_RE:
        if any_match(regexes, user_text):
            return label
    if turn_index == 1:
        return "initial_request"
    return "general_followup"


def response_features(response: str) -> dict[str, Any]:
    response_type, refusal_score = detect_response_type(response)
    return {
        "response_type": response_type,
        "refusal_score": refusal_score,
        "risk_score": None,
        "partial_compliance_score": None,
        "unsafe_compliance_score": None,
        "benign_helpfulness_score": None,
        "risk_delta": None,
        "scoring_source": "heuristic_v0_no_safety_judge",
    }


def normalize_role(role: str) -> str:
    role = (role or "").strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant"}:
        return "assistant"
    return role or "unknown"


def conversation_to_turn_pairs(conversation: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Convert alternating user/assistant messages into turn pairs."""
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(conversation) - 1:
        first = conversation[i]
        second = conversation[i + 1]
        first_role = normalize_role(str(first.get("from", "")))
        second_role = normalize_role(str(second.get("from", "")))
        if first_role == "user" and second_role == "assistant":
            pairs.append((str(first.get("value", "")), str(second.get("value", ""))))
            i += 2
        else:
            i += 1
    return pairs


def build_transition(
    *,
    dialogue_id: str,
    turn_index: int,
    total_turns: int,
    history_before: list[dict[str, str]],
    user_text: str,
    assistant_text: str,
    include_text: bool,
    max_text_chars: int | None,
) -> dict[str, Any]:
    action = detect_abstract_action(user_text, turn_index)
    features = response_features(assistant_text)
    done = turn_index == total_turns

    transition: dict[str, Any] = {
        "dialogue_id": dialogue_id,
        "turn": turn_index,
        "total_turns": total_turns,
        "state_prev": {
            "history_turns": len(history_before) // 2,
            "history_messages": len(history_before),
        },
        "action": {
            "abstract_action": action,
            "action_source": "heuristic_v0",
        },
        "observation": features,
        "future": {
            "future_boundary": None,
            "turns_to_boundary": None,
        },
        "reward": {
            "reward_value": None,
            "reward_source": "pending_judge_scores",
        },
        "done": done,
        "schema_version": "xguard_dreamer_transition_v0.1",
    }

    if include_text:
        transition["state_prev"]["history"] = [
            {"role": item["role"], "content": truncate_text(item["content"], max_text_chars)}
            for item in history_before
        ]
        transition["action"]["user_text"] = truncate_text(user_text, max_text_chars)
        transition["observation"]["assistant_response"] = truncate_text(assistant_text, max_text_chars)

    return transition


def convert_records(
    records: list[dict[str, Any]],
    *,
    include_text: bool,
    max_text_chars: int | None,
) -> tuple[Iterable[dict[str, Any]], Counter[str]]:
    stats: Counter[str] = Counter()

    def generator() -> Iterable[dict[str, Any]]:
        for idx, record in enumerate(records):
            conversation = record.get("conversations", [])
            if not isinstance(conversation, list):
                stats["invalid_conversation"] += 1
                continue

            pairs = conversation_to_turn_pairs(conversation)
            if not pairs:
                stats["empty_pairs"] += 1
                continue

            dialogue_id = f"xguard_{idx:06d}"
            history: list[dict[str, str]] = []
            stats["dialogues"] += 1
            stats[f"turn_count_{len(pairs)}"] += 1

            for turn_index, (user_text, assistant_text) in enumerate(pairs, start=1):
                transition = build_transition(
                    dialogue_id=dialogue_id,
                    turn_index=turn_index,
                    total_turns=len(pairs),
                    history_before=history,
                    user_text=user_text,
                    assistant_text=assistant_text,
                    include_text=include_text,
                    max_text_chars=max_text_chars,
                )
                stats["transitions"] += 1
                stats[f"action:{transition['action']['abstract_action']}"] += 1
                stats[f"response:{transition['observation']['response_type']}"] += 1
                yield transition

                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": assistant_text})

    return generator(), stats


def write_jsonl(items: Iterable[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert XGuard-Train conversations to Dreamer transition JSONL.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to xguard-train.json.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument("--stats-output", type=Path, default=None, help="Optional stats JSON path.")
    parser.add_argument("--max-records", type=int, default=None, help="Only convert the first N dialogues.")
    parser.add_argument("--no-text", action="store_true", help="Omit raw text fields from output.")
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=-1,
        help="Truncate raw text fields to N chars; negative keeps full text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output
    stats_output = args.stats_output or output_path.with_suffix(output_path.suffix + ".stats.json")
    max_text_chars = None if args.max_text_chars is None or args.max_text_chars < 0 else args.max_text_chars

    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise TypeError(f"Expected top-level list in {input_path}, got {type(records).__name__}")

    if args.max_records is not None:
        records = records[: args.max_records]

    items, stats = convert_records(
        records,
        include_text=not args.no_text,
        max_text_chars=max_text_chars,
    )
    write_jsonl(items, output_path)

    stats_payload = {
        "input": str(input_path),
        "output": str(output_path),
        "records_loaded": len(records),
        "include_text": not args.no_text,
        "max_text_chars": max_text_chars,
        "stats": dict(stats),
        "notes": [
            "risk_score, partial_compliance_score, unsafe_compliance_score, future_boundary, turns_to_boundary, and reward_value are placeholders until judge annotation is added.",
            "refusal_score, response_type, and abstract_action are heuristic_v0 labels for bootstrapping only.",
        ],
    }
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    stats_output.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote transitions: {output_path}")
    print(f"wrote stats: {stats_output}")
    print(f"dialogues: {stats.get('dialogues', 0)}")
    print(f"transitions: {stats.get('transitions', 0)}")


if __name__ == "__main__":
    main()
