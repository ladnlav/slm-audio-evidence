"""Dry-run tests for src/judges/ — no torch, no API key, no network.

Covers the part every backend shares and that a refactor is most likely to
silently break: prompt rendering and verdict parsing (src/judges/base.py),
plus run_eval.py's wiring (caching, provenance tagging, category routing)
exercised end-to-end against tests/fixtures/smoke_synthetic/ via the 'fake'
backend (src/judges/fake.py) — no GPU, no API key needed.
Run: python tests/test_judges.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.judges import PROMPT_VERSION, build_judge
from src.judges.base import LLMJudge, Verdict, parse_verdict, render_judge_prompt
from src.judges.fake import FakeJudge as SharedFakeJudge
from src.run_eval import run_evaluation

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "smoke_synthetic"

PARSE_CASES = [
    ("CORRECT", Verdict.CORRECT, "exact one-word reply"),
    ("Incorrect.", Verdict.INCORRECT, "punctuation after the word"),
    ("The answer is ABSTAINED because it declines.", Verdict.ABSTAINED, "verdict embedded in a sentence"),
    ("correct", Verdict.CORRECT, "lowercase"),
    ("I think this is CORRECT, definitely correct.", Verdict.CORRECT, "same verdict repeated (still one distinct word)"),
    ("Maybe CORRECT or maybe INCORRECT, hard to say.", Verdict.UNPARSEABLE, "two distinct verdicts -> ambiguous"),
    ("The system answer is great.", Verdict.UNPARSEABLE, "no verdict word at all"),
    ("", Verdict.UNPARSEABLE, "empty reply"),
]


def test_parse_verdict() -> int:
    failures = 0
    for raw, expected, description in PARSE_CASES:
        got = parse_verdict(raw)
        ok = got == expected
        failures += not ok
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] parse_verdict({raw!r}) = {got} (expected {expected}) — {description}")
    return failures


def test_verdict_to_correctness() -> int:
    expected = {
        Verdict.CORRECT: True,
        Verdict.INCORRECT: False,
        Verdict.ABSTAINED: False,
        Verdict.UNPARSEABLE: None,
    }
    failures = 0
    for verdict, want in expected.items():
        got = verdict.to_correctness()
        ok = got is want
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] {verdict}.to_correctness() = {got} (expected {want})")
    return failures


def test_render_judge_prompt() -> int:
    prompt = render_judge_prompt(
        transcript="The bridge was built in 1912.",
        question="When was the bridge built?",
        gold="1912",
        response="It was built in 1912.",
    )
    checks = {
        "transcript substituted": "The bridge was built in 1912." in prompt,
        "question substituted": "When was the bridge built?" in prompt,
        "gold substituted": "GOLD ANSWER: 1912" in prompt,
        "response substituted": "It was built in 1912." in prompt,
        "no placeholders left": "{" not in prompt,
    }
    failures = 0
    for description, ok in checks.items():
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] {description}")
    return failures


class FakeJudge(LLMJudge):
    """Deterministic stand-in backend — proves LLMJudge.judge() wiring without any model."""

    name = "llm-fake-v1"

    def __init__(self, canned_reply: str) -> None:
        self.canned_reply = canned_reply
        self.last_prompt: str | None = None

    def _generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.canned_reply


def test_fake_judge_end_to_end() -> int:
    judge = FakeJudge(canned_reply="CORRECT")
    result = judge.judge(transcript="t", question="q", gold="g", response="r")
    checks = {
        "verdict parsed": result.verdict == Verdict.CORRECT,
        "raw_output kept": result.raw_output == "CORRECT",
        "judge_name tagged": result.judge_name == "llm-fake-v1",
        "prompt was rendered (not raw template)": judge.last_prompt is not None and "{transcript}" not in judge.last_prompt,
    }
    failures = 0
    for description, ok in checks.items():
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] {description}")
    return failures


def test_build_judge_fake_backend() -> int:
    judge = build_judge("fake", canned="INCORRECT")
    result = judge.judge(transcript="t", question="q", gold="g", response="r")
    checks = {
        "build_judge('fake') returns a SharedFakeJudge": isinstance(judge, SharedFakeJudge),
        "canned kwarg respected": result.verdict == Verdict.INCORRECT,
        "call counter increments": judge.calls == 1,
    }
    failures = 0
    for description, ok in checks.items():
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] {description}")
    return failures


def test_run_eval_end_to_end() -> int:
    """Exercises run_eval.py's actual orchestration (not just the LLMJudge contract):
    category routing (only B+answer hits the judge), provenance tagging, and the
    on-disk cache that keeps a rerun from re-paying for expensive LLM calls.
    """
    out_dir = Path(tempfile.mkdtemp(prefix="judge_smoke_"))
    failures = 0
    try:
        run_evaluation(
            manifest_path=str(FIXTURES / "manifest.jsonl"),
            responses_path=str(FIXTURES / "responses.jsonl"),
            out_dir=str(out_dir),
            judge_backend="fake",
            judge_model=None,
        )
        judged_path = out_dir / "responses_judged.jsonl"
        by_id = {
            row["id"]: row
            for row in (json.loads(l) for l in judged_path.read_text(encoding="utf-8").splitlines())
        }

        # Per fixtures/README.md: syn-b1 is the only category-B item labeled 'answer' -> routed
        # to the LLM judge. syn-b2 is abstain and syn-b3 is hedge -> both stay on the rule path.
        # syn-a1/syn-c3 are category A/C -> never routed regardless of backend.
        checks = {
            "syn-b1 tagged with judge name": by_id["syn-b1"]["judge"] == "llm-fake-v1",
            "syn-b1 correct=True (canned CORRECT)": by_id["syn-b1"]["correct"] is True,
            "syn-b2 (abstain) stays on rules path": by_id["syn-b2"]["judge"] == "rules",
            "syn-a1 (category A) unaffected by judge backend": by_id["syn-a1"]["judge"] == "rules",
            "syn-c3 (category C) unaffected by judge backend": by_id["syn-c3"]["judge"] == "rules",
        }
        for description, ok in checks.items():
            failures += not ok
            print(f"  [{'OK' if ok else 'FAIL'}] {description}")

        cache_path = out_dir / "judge_cache.jsonl"
        cache_rows = [json.loads(l) for l in cache_path.read_text(encoding="utf-8").splitlines()] if cache_path.exists() else []
        cache_checks = {
            "judge_cache.jsonl created": cache_path.exists(),
            "one cache row for the one LLM-judged item (syn-b1)": len(cache_rows) == 1,
            "cache row carries prompt_version": bool(cache_rows) and cache_rows[0]["prompt_version"] == PROMPT_VERSION,
        }
        for description, ok in cache_checks.items():
            failures += not ok
            print(f"  [{'OK' if ok else 'FAIL'}] {description}")

        # Rerun: cached verdicts must be reused, not recomputed — LLM calls are the expensive part.
        run_evaluation(
            manifest_path=str(FIXTURES / "manifest.jsonl"),
            responses_path=str(FIXTURES / "responses.jsonl"),
            out_dir=str(out_dir),
            judge_backend="fake",
            judge_model=None,
        )
        cache_rows_after = [json.loads(l) for l in cache_path.read_text(encoding="utf-8").splitlines()]
        ok = len(cache_rows_after) == 1
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] cache not duplicated on rerun (still 1 row, not 2)")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    return failures


def main() -> None:
    suites = [
        ("parse_verdict", test_parse_verdict),
        ("Verdict.to_correctness", test_verdict_to_correctness),
        ("render_judge_prompt", test_render_judge_prompt),
        ("FakeJudge end-to-end", test_fake_judge_end_to_end),
        ("build_judge('fake') factory", test_build_judge_fake_backend),
        ("run_eval.py end-to-end (fake backend)", test_run_eval_end_to_end),
    ]
    total_failures = 0
    for name, suite in suites:
        print(f"\n=== {name} ===")
        total_failures += suite()

    print("\n================ РЕЗУЛЬТАТ ================")
    if total_failures == 0:
        print("[+] Все проверки прошли.")
    else:
        print(f"[-] {total_failures} проверок провалено.")
        sys.exit(1)


if __name__ == "__main__":
    main()
