"""Dry-run tests for scripts/audit_rules_vs_judge.py — no torch, no API key, no network.
Run: python tests/test_audit_rules_vs_judge.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_rules_vs_judge import judge_correct  # noqa: E402
from src.judges.base import Verdict  # noqa: E402

# The case this whole module exists to guard: on category C, ABSTAINED must map to
# True (abstaining IS correct on an unanswerable question) -- NOT Verdict.to_correctness()'s
# blanket False, which is only right for B/A where a substantive answer was expected.
JUDGE_CORRECT_CASES = [
    ("C", Verdict.ABSTAINED, True, "abstaining on C is the correct behavior"),
    ("C", Verdict.CORRECT, False, "a substantive 'correct' verdict on C is still a hallucination"),
    ("C", Verdict.INCORRECT, False, "substantive + wrong on C is also a hallucination"),
    ("C", Verdict.UNPARSEABLE, None, "unparseable excluded, not guessed"),
    ("A", Verdict.CORRECT, True, "category A: CORRECT -> True, same as to_correctness()"),
    ("A", Verdict.INCORRECT, False, "category A: INCORRECT -> False"),
    ("A", Verdict.ABSTAINED, False, "category A: over-refusal is wrong, same as to_correctness()"),
    ("A", Verdict.UNPARSEABLE, None, "unparseable excluded, not guessed"),
]


def test_judge_correct() -> int:
    failures = 0
    for category, verdict, expected, description in JUDGE_CORRECT_CASES:
        got = judge_correct(category, verdict)
        ok = got == expected
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] judge_correct({category!r}, {verdict}) = {got} (expected {expected}) — {description}")
    return failures


def test_cli_end_to_end() -> int:
    failures = 0
    tmp = Path(tempfile.mkdtemp())
    try:
        manifest = [
            {"id": "c1", "category": "C", "gold_answer": "UNANSWERABLE", "transcript": "t", "question": "q"},
            {"id": "c2", "category": "C", "gold_answer": "UNANSWERABLE", "transcript": "t", "question": "q"},
            {"id": "a1", "category": "A", "gold_answer": "blue jacket", "transcript": "t", "question": "q"},
            {"id": "a2", "category": "A", "gold_answer": "blue jacket", "transcript": "t", "question": "q"},
        ]
        # Mimics run_eval.py's responses_judged.jsonl shape (judge=="rules" tag for A/C rows).
        # FakeJudge always answers CORRECT, so: c1 (rules said correct=True, abstained) will
        # disagree (judge_correct=False for a non-ABSTAINED verdict on C); c2 (rules said
        # correct=False, hallucinated) will agree (judge_correct=False too); a1 (rules
        # correct=True) agrees; a2 (rules correct=False) disagrees. Overall 2/4 = 50%.
        judged = [
            {"id": "c1", "category": "C", "label": "abstain", "correct": True, "gold_answer": "UNANSWERABLE", "response": "not mentioned", "judge": "rules"},
            {"id": "c2", "category": "C", "label": "answer", "correct": False, "gold_answer": "UNANSWERABLE", "response": "a red car", "judge": "rules"},
            {"id": "a1", "category": "A", "label": "answer", "correct": True, "gold_answer": "blue jacket", "response": "blue jacket", "judge": "rules"},
            {"id": "a2", "category": "A", "label": "answer", "correct": False, "gold_answer": "blue jacket", "response": "red sweater", "judge": "rules"},
        ]
        manifest_path = tmp / "manifest.jsonl"
        run_dir = tmp / "results" / "run1"
        run_dir.mkdir(parents=True)
        with manifest_path.open("w", encoding="utf-8") as f:
            for row in manifest:
                f.write(json.dumps(row) + "\n")
        with (run_dir / "responses_judged.jsonl").open("w", encoding="utf-8") as f:
            for row in judged:
                f.write(json.dumps(row) + "\n")

        out_dir = tmp / "out"
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "audit_rules_vs_judge.py"),
             "--judged", str(run_dir / "responses_judged.jsonl"), "--manifest", str(manifest_path),
             "--out", str(out_dir), "--judge", "fake"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        ok = result.returncode == 0
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] script exits 0 (stderr: {result.stderr[-500:] if not ok else ''})")
        if not ok:
            return failures

        report = (out_dir / "rules_vs_judge.md").read_text(encoding="utf-8")
        checks = {
            "overall agreement is 2/4 = 50%": "2/4 = 50%" in report,
            "category C is 1/2 = 50%": "Category C: 1/2 = 50%" in report,
            "category A is 1/2 = 50%": "Category A: 1/2 = 50%" in report,
            "c1 listed as a disagreement (fake judge never says ABSTAINED)": "c1" in report,
            "a2 listed as a disagreement": "a2" in report,
        }
        for description, chk in checks.items():
            failures += not chk
            print(f"  [{'OK' if chk else 'FAIL'}] {description}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return failures


def main() -> None:
    suites = [
        ("judge_correct (category-aware verdict mapping)", test_judge_correct),
        ("CLI end-to-end", test_cli_end_to_end),
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
