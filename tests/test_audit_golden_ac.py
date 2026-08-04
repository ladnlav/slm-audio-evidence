"""Dry-run tests for scripts/audit_golden_ac.py — no torch, no API key, no network.
Run: python tests/test_audit_golden_ac.py
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_golden_ac import human_to_verdict  # noqa: E402
from src.judges.base import Verdict  # noqa: E402

MAPPING_CASES = [
    ("answer", "yes", Verdict.CORRECT.value, "answer+correct -> CORRECT"),
    ("answer", "no", Verdict.INCORRECT.value, "answer+incorrect -> INCORRECT"),
    ("answer", "", None, "answer with no correctness call is malformed, not a guess"),
    ("abstain", "", Verdict.ABSTAINED.value, "abstain -> ABSTAINED"),
    ("hedge", "", Verdict.ABSTAINED.value, "hedge collapses to ABSTAINED (judge_v1.txt has no HEDGE option)"),
    ("", "", None, "empty label is malformed"),
]


def test_human_to_verdict() -> int:
    failures = 0
    for label, correct, expected, description in MAPPING_CASES:
        got = human_to_verdict(label, correct)
        ok = got == expected
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] human_to_verdict({label!r}, {correct!r}) = {got} (expected {expected}) — {description}")
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
        responses = [
            {"id": "c1", "response": "not mentioned"},
            {"id": "c2", "response": "the speaker drives a red car"},
            {"id": "a1", "response": "blue jacket"},
            {"id": "a2", "response": "a red sweater"},
        ]
        run_dir = tmp / "results" / "run1"
        run_dir.mkdir(parents=True)
        manifest_path = tmp / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f:
            for row in manifest:
                f.write(json.dumps(row) + "\n")
        with (run_dir / "responses.jsonl").open("w", encoding="utf-8") as f:
            for row in responses:
                f.write(json.dumps(row) + "\n")

        out_dir = tmp / "golden_ac"
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "audit_golden_ac.py"), "sample",
             "--responses", str(run_dir / "responses.jsonl"), "--manifest", str(manifest_path),
             "--out", str(out_dir), "--n-c", "2", "--n-a", "2", "--judge", "fake"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        ok = result.returncode == 0
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] sample subcommand exits 0 (stderr: {result.stderr[-500:] if not ok else ''})")
        if not ok:
            return failures

        # FakeJudge always answers CORRECT (canned default) -- fill human labels to get a
        # known 50%/50%/50% split: c1 disagrees (human says abstain, judge said CORRECT),
        # c2 agrees (both effectively "answer, correct"), a1 agrees, a2 disagrees.
        sheet_path = out_dir / "sheet.csv"
        with sheet_path.open("r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        human_fill = {
            "c1": ("abstain", ""), "c2": ("answer", "yes"),
            "a1": ("answer", "yes"), "a2": ("answer", "no"),
        }
        for row in rows:
            item_id = row["sheet_id"].split("::")[-1]
            row["human_label"], row["human_correct"] = human_fill[item_id]
        with sheet_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "audit_golden_ac.py"), "score", "--dir", str(out_dir)],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        ok = result.returncode == 0
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] score subcommand exits 0 (stderr: {result.stderr[-500:] if not ok else ''})")
        if not ok:
            return failures

        report = (out_dir / "agreement.md").read_text(encoding="utf-8")
        checks = {
            "overall agreement is 2/4 = 50%": "2/4 = 50%" in report,
            "category C agreement is 1/2 = 50%": "1/2 = 50%" in report,
            "category A agreement is 1/2 = 50%": report.count("1/2 = 50%") == 2,  # once for C, once for A
            "below-gate warning present (50% < 80%)": "BELOW 80% GATE" in report,
            "c1 disagreement listed": "c1" in report and "Disagreements" in report,
        }
        for description, chk in checks.items():
            failures += not chk
            print(f"  [{'OK' if chk else 'FAIL'}] {description}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return failures


def main() -> None:
    suites = [
        ("human_to_verdict mapping", test_human_to_verdict),
        ("CLI end-to-end (sample + score)", test_cli_end_to_end),
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
