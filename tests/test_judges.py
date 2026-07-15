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
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent

from src.judges import DEFAULT_PROMPT_NAME, build_judge
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
    ("<think>Is this CORRECT or INCORRECT? Let me check... it matches the gold answer.</think>\nCORRECT",
     Verdict.CORRECT, "thinking block weighs both words as hypotheses, only text after </think> counts"),
    ("<think>hmm, could be CORRECT, or maybe INCORRECT, still deciding</think>",
     Verdict.UNPARSEABLE, "</think> closes but nothing follows it -- genuinely no verdict given"),
    ("<think>still reasoning about whether this is CORRECT or INCORRECT and never finishes",
     Verdict.UNPARSEABLE, "</think> never closes (ran out of tokens) -- correctly still ambiguous"),
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
        super().__init__()
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


def test_prompt_versioning() -> int:
    """prompt_version must be filename+content-hash, not just the filename, so an
    in-place rubric edit can never silently keep matching a stale judge_cache.jsonl
    entry -- and swapping judge_v1 <-> judge_v2 must actually render different text.
    """
    v1 = SharedFakeJudge(prompt_name="judge_v1.txt")
    v2 = SharedFakeJudge(prompt_name="judge_v2.txt")
    checks = {
        "v1 prompt_version starts with the filename": v1.prompt_version.startswith("judge_v1.txt@"),
        "v2 prompt_version starts with the filename": v2.prompt_version.startswith("judge_v2.txt@"),
        "v1 and v2 hash differently (different rubric text)": v1.prompt_version != v2.prompt_version,
        "default prompt_name is judge_v1.txt": DEFAULT_PROMPT_NAME == "judge_v1.txt",
    }
    rendered_v1 = render_judge_prompt("t", "q", "g", "r", prompt_name="judge_v1.txt")
    rendered_v2 = render_judge_prompt("t", "q", "g", "r", prompt_name="judge_v2.txt")
    checks["v1 and v2 render different prompt text"] = rendered_v1 != rendered_v2
    checks["v2 adds the verbosity-bias fix as a single trailing sentence, rest byte-identical to v1"] = (
        "confident-sounding answer that is missing the gold answer's specific facts is still INCORRECT" in rendered_v2
        and rendered_v2.startswith(rendered_v1)
    )

    v3 = SharedFakeJudge(prompt_name="judge_v3.txt")
    rendered_v3 = render_judge_prompt("t", "q", "g", "r", prompt_name="judge_v3.txt")
    checks["v3 prompt_version starts with the filename"] = v3.prompt_version.startswith("judge_v3.txt@")
    checks["v3 hashes differently from v1 and v2"] = len({v1.prompt_version, v2.prompt_version, v3.prompt_version}) == 3
    checks["v3 adds the transcript-grounding fix as a single trailing sentence, rest byte-identical to v1"] = (
        "states a specific fact that is absent from the transcript" in rendered_v3
        and rendered_v3.startswith(rendered_v1)
    )

    v4 = SharedFakeJudge(prompt_name="judge_v4.txt")
    v5 = SharedFakeJudge(prompt_name="judge_v5.txt")
    rendered_v4 = render_judge_prompt("t", "q", "g", "r", prompt_name="judge_v4.txt")
    rendered_v5 = render_judge_prompt("t", "q", "g", "r", prompt_name="judge_v5.txt")
    checks["v4 hashes differently from v1/v2/v3"] = len(
        {v1.prompt_version, v2.prompt_version, v3.prompt_version, v4.prompt_version}
    ) == 4
    checks["v4 = v1 with exactly the TRANSCRIPT line removed, nothing else"] = (
        rendered_v4 == rendered_v1.replace("TRANSCRIPT: t \n", "", 1)
    )
    checks["v5 hashes differently from v1/v2/v3/v4"] = len(
        {v1.prompt_version, v2.prompt_version, v3.prompt_version, v4.prompt_version, v5.prompt_version}
    ) == 5
    checks["v5 = v3 with the TRANSCRIPT line dropped (still mentions 'transcript' in the instruction text)"] = (
        "TRANSCRIPT:" not in rendered_v5
        and "states a specific fact that is absent from the transcript" in rendered_v5
    )
    failures = 0
    for description, ok in checks.items():
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] {description}")
    return failures


def test_set_prompt_reuses_instance() -> int:
    """set_prompt() must update prompt_name/prompt_version on the SAME instance -- for a local
    model this is what lets a notebook cell A/B a rubric without loading a second copy of the
    weights (two int8 8B copies don't fit on a 16 GB GPU; OOM'd for real, docs/decisions.md
    2026-07-16). No new object, no reload -- just swap the field.
    """
    judge = SharedFakeJudge(canned="CORRECT", prompt_name="judge_v1.txt")
    v1_version = judge.prompt_version
    before_id = id(judge)

    judge.set_prompt("judge_v3.txt")

    checks = {
        "same object, not a new instance": id(judge) == before_id,
        "prompt_name updated": judge.prompt_name == "judge_v3.txt",
        "prompt_version updated to match the new file": judge.prompt_version != v1_version
        and judge.prompt_version.startswith("judge_v3.txt@"),
    }
    result = judge.judge(transcript="t", question="q", gold="g", response="r")
    checks[".judge() still works after the swap (renders judge_v3.txt now, not judge_v1.txt)"] = (
        result.verdict == Verdict.CORRECT
    )
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
            "cache row carries prompt_version (name@hash, so an in-place rubric edit can't stay silently cached)":
                bool(cache_rows) and cache_rows[0]["prompt_version"].startswith(DEFAULT_PROMPT_NAME + "@"),
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


def test_run_eval_subset_ids() -> int:
    """subset_ids restricts evaluation to a fixed handful of items (src/judges/dev_subset.py)
    so a prompt/model change can be checked in seconds instead of paying for a full run.
    """
    out_dir = Path(tempfile.mkdtemp(prefix="judge_subset_"))
    failures = 0
    try:
        run_evaluation(
            manifest_path=str(FIXTURES / "manifest.jsonl"),
            responses_path=str(FIXTURES / "responses.jsonl"),
            out_dir=str(out_dir),
            judge_backend="fake",
            subset_ids={"syn-a1", "syn-b1"},
        )
        judged_path = out_dir / "responses_judged.jsonl"
        ids = {json.loads(l)["id"] for l in judged_path.read_text(encoding="utf-8").splitlines()}
        checks = {
            "only the requested subset gets evaluated": ids == {"syn-a1", "syn-b1"},
            "items outside the subset are skipped entirely, not just unjudged": "syn-c3" not in ids,
        }
        for description, ok in checks.items():
            failures += not ok
            print(f"  [{'OK' if ok else 'FAIL'}] {description}")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    return failures


def test_audit_compare_disagreements() -> int:
    """scripts/audit_judge.py compare: agreement count AND the disagreements log
    (both labels side by side) — the part that only had manual, not automated,
    coverage until a real Kaggle run surfaced actual judge failure modes.
    """
    out_dir = Path(tempfile.mkdtemp(prefix="audit_compare_"))
    failures = 0
    try:
        run_dir = out_dir / "results" / "fake_run"
        (run_dir / "llm_audit").mkdir(parents=True)

        # 4 manual-M1 ground-truth rows: 2 the judge will agree with, 2 it will get wrong.
        judged_rows = [
            {"id": "x1", "category": "B", "label": "answer", "correct": True,
             "gold_answer": "g1", "response": "r1", "judge": "manual-M1"},
            {"id": "x2", "category": "B", "label": "answer", "correct": False,
             "gold_answer": "g2", "response": "r2", "judge": "manual-M1"},
            {"id": "x3", "category": "B", "label": "answer", "correct": True,
             "gold_answer": "g3", "response": "r3", "judge": "manual-M1"},  # judge will say INCORRECT -> disagree
            {"id": "x4", "category": "B", "label": "answer", "correct": False,
             "gold_answer": "g4", "response": "r4", "judge": "manual-M1"},  # judge will say CORRECT -> disagree
        ]
        with (run_dir / "responses_judged.jsonl").open("w", encoding="utf-8") as f:
            for row in judged_rows:
                f.write(json.dumps(row) + "\n")

        cache_rows = [
            {"id": "x1", "judge_name": "llm-fake-v1", "prompt_version": "judge_v1.txt", "verdict": "CORRECT", "raw_output": "CORRECT"},
            {"id": "x2", "judge_name": "llm-fake-v1", "prompt_version": "judge_v1.txt", "verdict": "INCORRECT", "raw_output": "INCORRECT"},
            {"id": "x3", "judge_name": "llm-fake-v1", "prompt_version": "judge_v1.txt", "verdict": "INCORRECT", "raw_output": "well, INCORRECT I think"},
            {"id": "x4", "judge_name": "llm-fake-v1", "prompt_version": "judge_v1.txt", "verdict": "CORRECT", "raw_output": "CORRECT"},
        ]
        with (run_dir / "llm_audit" / "judge_cache.jsonl").open("w", encoding="utf-8") as f:
            for row in cache_rows:
                f.write(json.dumps(row) + "\n")

        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "audit_judge.py"), "compare",
             "--judged", str(run_dir / "responses_judged.jsonl"),
             "--manifest", "",  # no manifest -- question text is optional enrichment
             "--out", str(out_dir / "judge_audit")],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        ok = result.returncode == 0
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] compare exits 0 (stderr: {result.stderr[-300:] if not ok else ''})")

        report = (out_dir / "judge_audit" / "compare_manual_m1.md").read_text(encoding="utf-8")
        checks = {
            "agreement is 2/4 = 50%": "2/4 = 50%" in report,
            "flags below the 80% gate": "BELOW 80% GATE" in report,
            "disagreement section header": "## Disagreements (2 of 4)" in report,
            "x3 disagreement logged (human correct, llm incorrect)": "### x3" in report and "Human (manual-M1): correct" in report,
            "x4 disagreement logged (human incorrect, llm correct)": "### x4" in report,
            "x1/x2 (agreements) NOT in disagreements": "### x1" not in report and "### x2" not in report,
        }
        for description, ok in checks.items():
            failures += not ok
            print(f"  [{'OK' if ok else 'FAIL'}] {description}")

        disagreements = [json.loads(l) for l in (out_dir / "judge_audit" / "disagreements.jsonl").read_text(encoding="utf-8").splitlines()]
        dis_checks = {
            "disagreements.jsonl has exactly 2 rows": len(disagreements) == 2,
            "each row carries both labels": all("human_correct" in r and "llm_verdict" in r for r in disagreements),
        }
        for description, ok in dis_checks.items():
            failures += not ok
            print(f"  [{'OK' if ok else 'FAIL'}] {description}")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    return failures


def main() -> None:
    suites = [
        ("parse_verdict", test_parse_verdict),
        ("Verdict.to_correctness", test_verdict_to_correctness),
        ("render_judge_prompt", test_render_judge_prompt),
        ("FakeJudge end-to-end", test_fake_judge_end_to_end),
        ("prompt versioning (judge_v1 vs judge_v2)", test_prompt_versioning),
        ("set_prompt() reuses the same instance", test_set_prompt_reuses_instance),
        ("build_judge('fake') factory", test_build_judge_fake_backend),
        ("run_eval.py end-to-end (fake backend)", test_run_eval_end_to_end),
        ("run_eval.py subset_ids filtering", test_run_eval_subset_ids),
        ("audit_judge.py compare + disagreements log", test_audit_compare_disagreements),
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
