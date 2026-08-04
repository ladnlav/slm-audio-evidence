"""Dry-run tests for scripts/soft_labels.py — no torch, no API key, no network.

Covers the incorrect() formula (PLAN.md section 3 step 1) in isolation, then the
whole pipeline end-to-end via the 'fake' judge backend, checking specifically that
a category-B sample never contributes to incorrect() even when the judge says
CORRECT/INCORRECT for it, and that per-sample judge cache keys don't collide across
different samples of the same item id.
Run: python tests/test_soft_labels.py
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

from soft_labels import incorrect  # noqa: E402

INCORRECT_CASES = [
    ("C", "answer", None, 1, "hallucinated on an unanswerable question"),
    ("C", "answer", False, 1, "category branch fires regardless of correct's value"),
    ("C", "abstain", None, 0, "correct refusal on C"),
    ("C", "hedge", None, 0, "hedge on C is not counted as a hallucination here (own metric elsewhere)"),
    ("A", "answer", False, 1, "hallucinated a checkable fact"),
    ("A", "answer", True, 0, "correct A answer"),
    ("A", "abstain", None, 0, "over-refusal on A -- not this formula's target"),
    ("A", "hedge", None, 0, "hedge on A"),
    ("B", "answer", True, 0, "B is always 0 by design, even when correct"),
    ("B", "answer", False, 0, "B is always 0 by design, even when wrong -- see module docstring"),
    ("B", "abstain", None, 0, "B abstain still 0"),
    ("B", "hedge", None, 0, "B hedge still 0"),
]


def test_incorrect_formula() -> int:
    failures = 0
    for category, label, correct, expected, description in INCORRECT_CASES:
        got = incorrect(category, label, correct)
        ok = got == expected
        failures += not ok
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] incorrect({category!r}, {label!r}, {correct!r}) = {got} (expected {expected}) — {description}")
    return failures


def test_end_to_end() -> int:
    failures = 0
    tmp = Path(tempfile.mkdtemp())
    try:
        manifest = [
            {"id": "c1", "category": "C", "gold_answer": "UNANSWERABLE", "transcript": "t", "question": "q"},
            {"id": "a1", "category": "A", "gold_answer": "blue jacket", "transcript": "t", "question": "q"},
            {"id": "b1", "category": "B", "gold_answer": "three times", "transcript": "t", "question": "q"},
        ]
        responses = [
            {"id": "c1", "samples": ["not mentioned in the audio", "the speaker drives a red sedan", "there is no information"]},
            {"id": "a1", "samples": ["blue jacket", "he was wearing a red sweater"]},
            {"id": "b1", "samples": ["three times", "I don't know how many times"]},
        ]
        manifest_path = tmp / "manifest.jsonl"
        responses_path = tmp / "responses.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f:
            for row in manifest:
                f.write(json.dumps(row) + "\n")
        with responses_path.open("w", encoding="utf-8") as f:
            for row in responses:
                f.write(json.dumps(row) + "\n")

        out_dir = tmp / "out"
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "soft_labels.py"),
             "--manifest", str(manifest_path), "--responses", str(responses_path),
             "--out", str(out_dir), "--judge", "fake"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        ok = result.returncode == 0
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] soft_labels.py exits 0 (stderr: {result.stderr[-500:] if not ok else ''})")
        if not ok:
            return failures

        soft_labels = {json.loads(l)["id"]: json.loads(l) for l in (out_dir / "soft_labels.jsonl").read_text(encoding="utf-8").splitlines()}

        checks = {
            "c1 soft_label = 1/3 (one hallucinated sample of three)": abs(soft_labels["c1"]["soft_label"] - 1 / 3) < 1e-9,
            "a1 soft_label = 0.5 (one wrong fact of two)": abs(soft_labels["a1"]["soft_label"] - 0.5) < 1e-9,
            "b1 soft_label = 0.0 (B never contributes, even with an answer+judge call)": soft_labels["b1"]["soft_label"] == 0.0,
            "each item reports n_samples matching its own sample count": all(
                soft_labels[i]["n_samples"] == len(r["samples"]) for i, r in zip(["c1", "a1", "b1"], responses)
            ),
        }
        for description, chk in checks.items():
            failures += not chk
            print(f"  [{'OK' if chk else 'FAIL'}] {description}")

        cache = [json.loads(l) for l in (out_dir / "judge_cache.jsonl").read_text(encoding="utf-8").splitlines()]
        cache_checks = {
            "exactly 1 judge call made (only b1's sample_idx=0 is category B + label=answer)": len(cache) == 1,
            "cached row carries sample_idx so it doesn't collide with b1's other sample": cache and cache[0]["sample_idx"] == 0,
        }
        for description, chk in cache_checks.items():
            failures += not chk
            print(f"  [{'OK' if chk else 'FAIL'}] {description}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return failures


def main() -> None:
    suites = [
        ("incorrect() formula", test_incorrect_formula),
        ("soft_labels.py end-to-end (fake backend)", test_end_to_end),
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
