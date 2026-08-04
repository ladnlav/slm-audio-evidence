"""Which categories reach the LLM judge in run_eval.py — no torch, no API key, no network.

Regression guard for the 2026-07-31 fix: the judge used to be wired to category B only,
so on the scale set (A + C, no B by construction) it never fired and every category-A
answer was graded by fuzz.partial_ratio alone -- which false-negatives paraphrases
("oxygen therapy" vs gold "oxygen supplementation"), spelled-out numbers and reworded
dates. That silently depressed accuracy_A and, worse, would have poisoned the A2 soft
labels, where `incorrect = A + answer + wrong` turns every such false negative into a
fake "hallucination" in the probe's training signal.

Run: python tests/test_run_eval_judge_scope.py   (or: pytest tests/test_run_eval_judge_scope.py)
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.judges.fake import FakeJudge  # noqa: E402
from src.run_eval import run_evaluation  # noqa: E402


# One item per category. The A item is the exact shape the bug hid: a correct paraphrase
# that fuzz.partial_ratio scores below its 85 threshold.
MANIFEST = [
    {"id": "it-A", "category": "A", "gold_answer": "oxygen supplementation",
     "transcript": "Oxygen therapy is used to increase oxygen levels.", "question": "What treatment raises blood oxygen?"},
    {"id": "it-B", "category": "B", "gold_answer": "about 46 years",
     "transcript": "Built in 1910, demolished in 1956.", "question": "How long did the building stand?"},
    {"id": "it-C", "category": "C", "gold_answer": "UNANSWERABLE",
     "transcript": "The lecture covers photosynthesis.", "question": "What year was the lecturer born?"},
]

RESPONSES = [
    {"id": "it-A", "model": "m", "strategy": "plain", "response": "Oxygen therapy is used to increase oxygen levels."},
    {"id": "it-B", "model": "m", "strategy": "plain", "response": "Just under half a century."},
    {"id": "it-C", "model": "m", "strategy": "plain", "response": "The lecturer was born in 1948."},
]


def _write(tmp: Path, rows: list[dict], name: str) -> str:
    path = tmp / name
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return str(path)


def _run(tmp: Path, judge) -> dict[str, dict]:
    """Evaluate the 3-item fixture with `judge`; return judged rows keyed by id."""
    out_dir = tmp / "out"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    run_evaluation(
        manifest_path=_write(tmp, MANIFEST, "manifest.jsonl"),
        responses_path=_write(tmp, RESPONSES, "responses.jsonl"),
        out_dir=str(out_dir),
        judge=judge,
    )
    rows = [json.loads(l) for l in (out_dir / "responses_judged.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["id"]: r for r in rows}


def _check(cases: list[tuple[bool, str]]) -> int:
    failures = 0
    for ok, description in cases:
        print(f"  [{'ok' if ok else 'FAIL'}] {description}")
        failures += 0 if ok else 1
    return failures


def test_judge_scope_covers_a_and_b_but_not_c() -> int:
    """The core regression: A and B reach the judge, C never does."""
    print("test_judge_scope_covers_a_and_b_but_not_c")
    with tempfile.TemporaryDirectory() as tmp_name:
        judge = FakeJudge(canned="CORRECT")
        rows = _run(Path(tmp_name), judge)
        return _check([
            (rows["it-A"]["judge"] == judge.name, f"A judged by the LLM (got judge={rows['it-A']['judge']!r})"),
            (rows["it-B"]["judge"] == judge.name, f"B judged by the LLM (got judge={rows['it-B']['judge']!r})"),
            (rows["it-C"]["judge"] == "rules", f"C stays on the rules classifier (got {rows['it-C']['judge']!r})"),
            (judge.calls == 2, f"exactly 2 judge calls, one per answerable item (got {judge.calls})"),
        ])


def test_judge_overrides_fuzzy_false_negative_on_a() -> int:
    """A correct paraphrase that fuzzy rejects must come out correct once the judge sees it."""
    print("test_judge_overrides_fuzzy_false_negative_on_a")
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        without = _run(tmp, None)                       # rules + fuzzy only, the old behaviour
        with_judge = _run(tmp, FakeJudge(canned="CORRECT"))
        return _check([
            (without["it-A"]["correct"] is False, "fuzzy alone marks the paraphrase incorrect (the bug's effect)"),
            (with_judge["it-A"]["correct"] is True, "the judge's CORRECT verdict overrides it"),
            (without["it-C"]["correct"] is False and with_judge["it-C"]["correct"] is False,
             "C is unaffected by the judge either way -- a substantive answer stays a hallucination"),
        ])


def test_unparseable_falls_back_per_category() -> int:
    """An unparseable reply must not drop the item: A falls back to fuzzy, B stays pending-manual."""
    print("test_unparseable_falls_back_per_category")
    with tempfile.TemporaryDirectory() as tmp_name:
        # A reply with no verdict word at all -> parse_verdict returns UNPARSEABLE.
        rows = _run(Path(tmp_name), FakeJudge(canned="hmm, hard to say"))
        return _check([
            (rows["it-A"]["judge"] == "rules", f"A falls back to the rules tag (got {rows['it-A']['judge']!r})"),
            (rows["it-A"]["correct"] is False, "A keeps the fuzzy verdict instead of becoming None"),
            (rows["it-B"]["judge"] == "pending-manual", f"B stays pending-manual (got {rows['it-B']['judge']!r})"),
            (rows["it-B"]["correct"] is None, "B correctness stays None, so accuracy_B excludes it"),
        ])


def test_abstain_on_a_never_reaches_the_judge() -> int:
    """A refusal on A is over-refusal by definition -- no semantic judging needed."""
    print("test_abstain_on_a_never_reaches_the_judge")
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        judge = FakeJudge(canned="CORRECT")
        refusal = [dict(RESPONSES[0], response="I cannot answer, the audio does not say.")]
        out_dir = tmp / "out"
        run_evaluation(
            manifest_path=_write(tmp, [MANIFEST[0]], "m.jsonl"),
            responses_path=_write(tmp, refusal, "r.jsonl"),
            out_dir=str(out_dir),
            judge=judge,
        )
        row = json.loads((out_dir / "responses_judged.jsonl").read_text(encoding="utf-8").strip())
        return _check([
            (row["label"] == "abstain", f"the refusal is labelled abstain (got {row['label']!r})"),
            (judge.calls == 0, f"the judge is not called on a refusal (got {judge.calls} calls)"),
        ])


def main() -> int:
    failures = (
        test_judge_scope_covers_a_and_b_but_not_c()
        + test_judge_overrides_fuzzy_false_negative_on_a()
        + test_unparseable_falls_back_per_category()
        + test_abstain_on_a_never_reaches_the_judge()
    )
    print("\nOK" if failures == 0 else f"\n{failures} FAILURE(S)")
    return failures


# pytest entry points: assert instead of returning a count
def test_scope(): assert test_judge_scope_covers_a_and_b_but_not_c() == 0
def test_override(): assert test_judge_overrides_fuzzy_false_negative_on_a() == 0
def test_unparseable(): assert test_unparseable_falls_back_per_category() == 0
def test_abstain(): assert test_abstain_on_a_never_reaches_the_judge() == 0


if __name__ == "__main__":
    sys.exit(main())
