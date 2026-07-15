"""Dry-run tests for src/judges/tiered.py — no torch, no API key, no network.

Uses a mode-aware fake backend (returns a different verdict depending on which mode
TieredJudge last set) to verify escalation actually routes through to the slow pass,
not just that the length-ratio heuristic fires.
Run: python tests/test_tiered.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.judges.base import DEFAULT_PROMPT_NAME, LLMJudge, Verdict
from src.judges.tiered import DEFAULT_LENGTH_RATIO_THRESHOLD, TieredJudge


class _ModeAwareFake(LLMJudge):
    """Stands in for LocalHFJudge: exposes enable_thinking/max_new_tokens as plain mutable
    attributes (what TieredJudge toggles) and answers differently per mode, so a test can
    tell which pass actually produced the final verdict.
    """

    def __init__(self) -> None:
        super().__init__(prompt_name=DEFAULT_PROMPT_NAME)
        self.name = "llm-fake-v1"
        self.enable_thinking = False
        self.max_new_tokens = 64
        self.calls: list[bool] = []

    def _generate(self, prompt: str) -> str:
        self.calls.append(self.enable_thinking)
        return "INCORRECT" if self.enable_thinking else "CORRECT"


def test_short_response_not_escalated() -> int:
    backend = _ModeAwareFake()
    tiered = TieredJudge(backend)
    result = tiered.judge("transcript", "question", "a short gold answer", "a short response")
    checks = {
        "no escalation for a response about the same length as gold": tiered.escalated_count == 0,
        "one fast-mode call only": backend.calls == [False],
        "verdict is the fast pass's": result.verdict == Verdict.CORRECT,
        "judge_name is the tiered name, not the backend's": result.judge_name == tiered.name,
    }
    failures = 0
    for description, ok in checks.items():
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] {description}")
    return failures


def test_long_response_escalates() -> int:
    backend = _ModeAwareFake()
    tiered = TieredJudge(backend)
    gold = "short gold"
    response = " ".join(["word"] * (int(DEFAULT_LENGTH_RATIO_THRESHOLD) * len(gold.split()) + 5))
    result = tiered.judge("transcript", "question", gold, response)
    checks = {
        "escalates when response >> gold in length": tiered.escalated_count == 1,
        "fast pass ran first, then slow": backend.calls == [False, True],
        "final verdict is the slow pass's, not the fast pass's": result.verdict == Verdict.INCORRECT,
    }
    failures = 0
    for description, ok in checks.items():
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] {description}")
    return failures


def test_counters_accumulate_across_calls() -> int:
    backend = _ModeAwareFake()
    tiered = TieredJudge(backend)
    tiered.judge("t", "q", "short gold", "short response")  # not escalated
    tiered.judge("t", "q", "short gold", " ".join(["word"] * 50))  # escalated
    checks = {
        "total_count counts every call": tiered.total_count == 2,
        "escalated_count counts only escalated calls": tiered.escalated_count == 1,
    }
    failures = 0
    for description, ok in checks.items():
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] {description}")
    return failures


def main() -> None:
    failures = 0
    print("=== TieredJudge: short response stays on the fast pass ===")
    failures += test_short_response_not_escalated()
    print("\n=== TieredJudge: long response escalates to the slow pass ===")
    failures += test_long_response_escalates()
    print("\n=== TieredJudge: escalation counters ===")
    failures += test_counters_accumulate_across_calls()

    print("\n================ РЕЗУЛЬТАТ ================")
    if failures:
        print(f"[-] {failures} проверок не прошло.")
        sys.exit(1)
    print("[+] Все проверки прошли.")


if __name__ == "__main__":
    main()
