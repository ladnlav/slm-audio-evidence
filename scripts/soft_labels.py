"""Soft labels for A2 probing (M4 Task 3, PLAN.md section 3 step 2).

Input: a responses.jsonl produced with --strategy s4_consistency (src/inference.py),
where each row has one `id` and a `samples` list of k>=1 sampled responses instead of
a single `response`. For each sample, grades it through the same rules -> fuzzy ->
judge pipeline as src/run_eval.py (classify_response / check_correctness / LLMJudge
for category-B answers), then applies the incorrect() formula from PLAN.md/decisions.md
to get a 0/1 per sample, and averages over the k samples for one soft label per item.

incorrect() is deliberately NOT "was this response wrong" in general -- it targets one
specific failure mode (a substantive answer that should have been a refusal):
  category C + label answer            -> 1  (hallucinated on an unanswerable question)
  category A + label answer + wrong    -> 1  (hallucinated a fact that was checkable)
  everything else (abstain, hedge, any B, correct A) -> 0
B is always 0 by this formula, even for a wrong B answer -- see docs/decisions.md /
the M1 confirmation this needs before soft labels ship: this project's soft label is a
hallucination-on-refusal-worthy-question signal, not a general correctness signal, and
mixing in "hard inference B failures" would train the probe on the wrong target.

Usage:
  py -3 scripts/soft_labels.py \
      --manifest data/manifests/pilot.jsonl \
      --responses results/<run_id>/responses.jsonl \
      --out results/<run_id>/soft_labels \
      --judge local --judge-model <path or Hub id> --judge-display-name "Qwen/Qwen3-8B"

  # smoke test without a GPU:
  py -3 scripts/soft_labels.py --manifest data/manifests/pilot.jsonl \
      --responses results/<run_id>/responses.jsonl --out /tmp/soft --judge fake
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `from src...`

from src.judge import classify_response, check_correctness  # noqa: E402
from src.judges import DEFAULT_PROMPT_NAME, Verdict, build_judge  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def incorrect(category: str, label: str, correct: bool | None) -> int:
    """PLAN.md section 3 step 1 formula -- see module docstring for why B is always 0."""
    if category == "C" and label == "answer":
        return 1
    if category == "A" and label == "answer" and correct is False:
        return 1
    return 0


def grade_sample(
    item_id: str, sample_idx: int, category: str, gold_answer: str, transcript: str,
    question: str, response: str, judge, cache: dict, cache_file,
) -> dict:
    """Grades one sample the same way src/run_eval.py grades one response, plus incorrect().

    Cache key includes sample_idx (not just item id, as run_eval.py's judge_cache.jsonl
    does) -- one item id now has k *different* response texts, so an id-only key would
    silently collide k-1 samples onto whichever one was cached first.
    """
    label = classify_response(response)
    correct = check_correctness(category, label, response, gold_answer)
    judge_verdict_raw = None

    # Both answerable categories go to the judge, not just B -- the same scope defect
    # that src/run_eval.py carried until 2026-07-31 (commit ea40188). It matters more
    # here: incorrect() turns a wrongly-graded A answer into a "hallucination" in the
    # probe's training signal, so a paraphrase the fuzzy matcher misses would teach the
    # probe that a correct answer was a fabrication. C is excluded by design, since there
    # only the label counts and any substantive answer is a hallucination regardless.
    if judge is not None and category in ("A", "B") and label == "answer":
        cache_key = (item_id, sample_idx, judge.name, judge.prompt_version)
        cached = cache.get(cache_key)
        if cached is not None:
            verdict_value, raw_output = cached["verdict"], cached["raw_output"]
        else:
            try:
                result = judge.judge(transcript, question, gold_answer, response)
                verdict_value, raw_output = result.verdict.value, result.raw_output
            except Exception as exc:  # keep the batch alive; sample stays UNPARSEABLE
                verdict_value, raw_output = Verdict.UNPARSEABLE.value, f"[ERROR] {exc}"
            cache_file.write(json.dumps({
                "id": item_id, "sample_idx": sample_idx, "judge_name": judge.name,
                "prompt_version": judge.prompt_version, "verdict": verdict_value, "raw_output": raw_output,
            }, ensure_ascii=False) + "\n")
            cache_file.flush()
        correct = Verdict(verdict_value).to_correctness()
        judge_verdict_raw = verdict_value

    return {
        "sample_idx": sample_idx, "response": response, "label": label, "correct": correct,
        "judge_verdict": judge_verdict_raw, "incorrect": incorrect(category, label, correct),
    }


def load_cache(cache_path: Path) -> dict:
    cache = {}
    for row in load_jsonl(cache_path):
        key = (row["id"], row["sample_idx"], row["judge_name"], row["prompt_version"])
        cache[key] = row
    return cache


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="data/manifests/pilot.jsonl")
    p.add_argument("--responses", required=True, help="s4_consistency responses.jsonl (has a 'samples' list per id).")
    p.add_argument("--out", required=True, help="Output directory for soft_labels.jsonl + judge_cache.jsonl.")
    p.add_argument("--judge", choices=["none", "local", "gemini", "fake"], default="none",
                    help="Same as run_eval.py --judge. 'none' leaves B samples UNPARSEABLE/incorrect=0 "
                         "(only meaningful for a manifest with no B, e.g. the scale set).")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--judge-display-name", default=None)
    p.add_argument("--judge-prompt", default=DEFAULT_PROMPT_NAME)
    p.add_argument("--judge-thinking", action="store_true", help="See run_eval.py --judge-thinking.")
    args = p.parse_args()

    manifest = {row["id"]: row for row in load_jsonl(Path(args.manifest))}
    responses = load_jsonl(Path(args.responses))
    if not responses:
        raise SystemExit(f"No rows in {args.responses}")

    judge = None
    if args.judge != "none":
        judge_kwargs: dict = {"prompt_name": args.judge_prompt}
        if args.judge_model:
            judge_kwargs["model_id"] = args.judge_model
        if args.judge == "local":
            if args.judge_display_name:
                judge_kwargs["display_name"] = args.judge_display_name
            judge_kwargs["enable_thinking"] = args.judge_thinking
        judge = build_judge(args.judge, **judge_kwargs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "judge_cache.jsonl"
    cache = load_cache(cache_path) if judge is not None else {}

    soft_labels = []
    skipped_no_samples = 0
    with cache_path.open("a", encoding="utf-8") as cache_file:
        for row in responses:
            item_id = row["id"]
            m = manifest.get(item_id)
            if m is None:
                print(f"[!] id {item_id} not in manifest -- skipping")
                continue
            samples = row.get("samples")
            if not samples:
                skipped_no_samples += 1
                continue

            graded = [
                grade_sample(
                    item_id, i, m["category"], m["gold_answer"], m.get("transcript", ""),
                    m.get("question", ""), sample, judge, cache, cache_file,
                )
                for i, sample in enumerate(samples)
            ]
            soft_label = sum(g["incorrect"] for g in graded) / len(graded)
            soft_labels.append({
                "id": item_id, "category": m["category"], "n_samples": len(graded),
                "soft_label": soft_label, "per_sample": graded,
            })
            print(f"[{len(soft_labels)}/{len(responses)}] {item_id} soft_label={soft_label:.2f} (k={len(graded)})")

    if skipped_no_samples:
        print(f"[!] {skipped_no_samples} row(s) had no 'samples' field -- not from --strategy s4_consistency, skipped.")

    out_path = out_dir / "soft_labels.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in soft_labels:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n[+] {len(soft_labels)} soft labels -> {out_path}")


if __name__ == "__main__":
    main()
