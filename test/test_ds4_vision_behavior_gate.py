#!/usr/bin/env python3
"""GPU-free checks for the digest-bound DS4 vision behavioural runner."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ds4_vision_behavior_gate", ROOT / "scripts" / "ds4_vision_behavior_gate.py")
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def policy() -> dict:
    return {
        "synthetic": {
            "classes": ["count", "colour", "spatial", "ocr"],
            "arm_a_min_accuracy_per_class": 0.90,
            "arm_a_significance": {"alpha": 0.01},
            "arm_b_requirement": {"alpha": 0.05},
        },
        "natural": {"min_retained_after_cuts": 80,
                    "arm_a_min_accuracy": 0.70},
    }


def synthetic_rows() -> list[dict]:
    chance = {"count": 0.125, "colour": 1.0 / 6.0,
              "spatial": 0.5, "ocr": 1.0 / (36.0**4)}
    rows = []
    for name in policy()["synthetic"]["classes"]:
        for index in range(25):
            rows.append({"id": f"{name}_{index:03d}", "class": name,
                         "chance": chance[name], "arm_b_correct": False,
                         "arm_a_correct": index < 23})
    return rows


def main() -> int:
    assert gate.normalize("  FOUR... blue-green! ") == "four bluegreen"
    assert gate.answer_matches("The answer is four.", "four")
    assert gate.answer_matches("Sign says BUD LIGHT outside", "bud light")
    assert not gate.answer_matches("fourteen", "four")
    assert not gate.answer_matches("before after", "for")

    assert gate.binomial_upper_tail(25, 25, 0.5) < 0.01
    assert math.isclose(gate.binomial_upper_tail(0, 25, 0.125), 1.0)

    rows = synthetic_rows()
    verdict = gate.score_synthetic(rows, policy())
    assert verdict["passed"] is True
    assert verdict["first_red_class"] is None
    assert all(row["passed"] for row in verdict["classes"])

    # First-red is per policy class, never hidden by the aggregate.
    changed = [dict(row) for row in rows]
    for row in changed:
        if row["class"] == "colour" and row["id"].endswith(("022", "023", "024")):
            row["arm_a_correct"] = False
    verdict = gate.score_synthetic(changed, policy())
    assert verdict["passed"] is False
    assert verdict["first_red_class"] == "colour"
    assert next(row for row in verdict["classes"] if row["class"] == "colour")[
        "checks"]["arm_a_accuracy"] is False

    # Withheld answers are listed and removed from the Arm-A denominator.
    changed = [dict(row) for row in rows]
    count_rows = [row for row in changed if row["class"] == "count"]
    for row in count_rows[:12]:
        row["arm_b_correct"] = True
    verdict = gate.score_synthetic(changed, policy())
    count = verdict["classes"][0]
    assert count["retained"] == 13
    assert count["cut_ids"] == [row["id"] for row in count_rows[:12]]
    assert count["checks"]["arm_b_at_chance"] is False
    assert verdict["passed"] is False

    natural = [{"id": f"tvqa_{index}", "arm_b_correct": index < 10,
                "arm_a_correct": 10 <= index < 73} for index in range(100)]
    verdict = gate.score_natural(natural, policy())
    assert verdict["passed"] is True
    assert verdict["retained"] == 90
    assert verdict["arm_a_correct"] == 63
    assert len(verdict["cut_ids"]) == 10

    compromised = [dict(row) for row in natural]
    for row in compromised[:21]:
        row["arm_b_correct"] = True
    verdict = gate.score_natural(compromised, policy())
    assert verdict["passed"] is False
    assert verdict["first_red"] == "retained_count"

    payload = gate.build_payload(
        {"question": "How many?", "image": b"\x89PNG\r\n\x1a\n"}, "model", "A")
    assert payload["temperature"] == 0.0 and payload["max_tokens"] == 32
    assert payload["messages"][0]["content"][0]["image_url"]["url"].startswith(
        "data:image/png;base64,")
    withheld = gate.build_payload(
        {"question": "How many?", "image": b"ignored"}, "model", "B")
    assert withheld["messages"][0]["content"] == [
        {"type": "text", "text": "How many?"}]

    print("ds4 vision behavior gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
