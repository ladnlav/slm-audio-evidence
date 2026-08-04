"""Rules vs judge agreement on the scale-set (M4 Task 2 step 4 -- appendix bonus).

Answers the reviewer question "why do you need an LLM judge if you have rules" with a
number: takes the scale-set's already rule-graded A/C responses (responses_judged.jsonl,
judge=="rules" -- see src/run_eval.py) and additionally asks the LLM judge to grade the
same items cold, then reports how often the two agree.

COST WARNING: this calls the judge on every A/C item passed in (up to ~2000 per PLAN.md's
scale-set count) -- roughly the same GPU time as grading category B once (see the
DataSphere migration plan). PLAN.md/ROLE_M4.md mark this an appendix bonus, not a core
deliverable -- run it after soft labels and the real scale-set B grading are done, not
before, and consider --limit for a cheap first pass.

Verdict-to-correctness here is category-aware, NOT src/judges/base.py's
Verdict.to_correctness() directly: that mapping treats ABSTAINED as always incorrect,
which is right for B (a substantive answer was expected) but backwards for C (abstaining
IS the correct answer -- src/judge.py's check_correctness, category=="C" branch: `return
label == "abstain"`). Blindly reusing to_correctness() would score every judge-abstained
C item as "judge says wrong" even when it agrees with what rules already called correct.

Usage:
  py -3 scripts/audit_rules_vs_judge.py --judged "results/*/responses_judged.jsonl" \
      --manifest data/manifests/scale_nmsqa.jsonl --out results/rules_vs_judge \
      --judge local --judge-model <path> --judge-display-name "Qwen/Qwen3-8B"
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `from src...`

from src.judges import DEFAULT_PROMPT_NAME, Verdict, build_judge  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def judge_correct(category: str, verdict: Verdict) -> bool | None:
    """See module docstring: category-aware, deliberately not Verdict.to_correctness()."""
    if verdict == Verdict.UNPARSEABLE:
        return None
    if category == "C":
        return verdict == Verdict.ABSTAINED
    return verdict.to_correctness()  # category A: CORRECT->True, INCORRECT/ABSTAINED->False


def load_cache(cache_path: Path) -> dict:
    cache = {}
    for row in load_jsonl(cache_path):
        key = (row["id"], row["judge_name"], row["prompt_version"])
        cache[key] = row
    return cache


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judged", required=True, help="Glob for responses_judged.jsonl, e.g. 'results/*/responses_judged.jsonl'.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default="results/rules_vs_judge")
    p.add_argument("--limit", type=int, default=None, help="Only the first N eligible rows -- cheap dry run before the full set.")
    p.add_argument("--judge", choices=["local", "gemini", "fake"], default="local")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--judge-display-name", default=None)
    p.add_argument("--judge-prompt", default=DEFAULT_PROMPT_NAME)
    p.add_argument("--judge-thinking", action="store_true")
    args = p.parse_args()

    manifest = {row["id"]: row for row in load_jsonl(Path(args.manifest))}

    rows = []
    for judged_path in sorted(glob.glob(args.judged, recursive=True)):
        for row in load_jsonl(Path(judged_path)):
            if row.get("judge") != "rules":
                continue  # only items the rule classifier itself called a verdict on (A/C)
            m = manifest.get(row["id"])
            if m is None or m["category"] not in ("A", "C"):
                continue
            rows.append({
                "id": row["id"], "category": m["category"], "transcript": m.get("transcript", ""),
                "question": m.get("question", ""), "gold_answer": row["gold_answer"],
                "response": row["response"], "rules_correct": row["correct"],
            })
    if not rows:
        raise SystemExit(f"No rule-graded A/C rows found matching {args.judged} against {args.manifest}")
    if args.limit:
        rows = rows[: args.limit]

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
    cache = load_cache(cache_path)

    agree_by_cat: Counter = Counter()
    total_by_cat: Counter = Counter()
    unparseable = 0
    disagreements = []
    with cache_path.open("a", encoding="utf-8") as cache_file:
        for i, row in enumerate(rows, 1):
            cache_key = (row["id"], judge.name, judge.prompt_version)
            cached = cache.get(cache_key)
            if cached is not None:
                verdict_value, raw_output = cached["verdict"], cached["raw_output"]
            else:
                try:
                    result = judge.judge(row["transcript"], row["question"], row["gold_answer"], row["response"])
                    verdict_value, raw_output = result.verdict.value, result.raw_output
                except Exception as exc:
                    verdict_value, raw_output = Verdict.UNPARSEABLE.value, f"[ERROR] {exc}"
                cache_file.write(json.dumps({
                    "id": row["id"], "judge_name": judge.name, "prompt_version": judge.prompt_version,
                    "verdict": verdict_value, "raw_output": raw_output,
                }, ensure_ascii=False) + "\n")
                cache_file.flush()
            print(f"[{i}/{len(rows)}] {row['id']} ({row['category']}) {verdict_value}")

            jc = judge_correct(row["category"], Verdict(verdict_value))
            if jc is None:
                unparseable += 1
                continue
            total_by_cat[row["category"]] += 1
            if jc == row["rules_correct"]:
                agree_by_cat[row["category"]] += 1
            else:
                disagreements.append({**row, "judge_verdict": verdict_value, "judge_correct": jc})

    total = sum(total_by_cat.values())
    agree = sum(agree_by_cat.values())
    pct = 100 * agree / total if total else 0.0

    lines = [
        f"# Rule classifier vs LLM judge, scale-set A/C ({total} items scored, {unparseable} UNPARSEABLE excluded)",
        "",
        f"Overall agreement: **{agree}/{total} = {pct:.0f}%**",
    ]
    for category in sorted(total_by_cat):
        c_total, c_agree = total_by_cat[category], agree_by_cat[category]
        c_pct = 100 * c_agree / c_total if c_total else 0.0
        lines.append(f"- Category {category}: {c_agree}/{c_total} = {c_pct:.0f}%")
    lines.append(f"\n## Disagreements ({len(disagreements)} of {total})")
    for d in disagreements[:50]:  # cap the printed sample; full data is in judge_cache.jsonl
        lines.append(f"- {d['id']} ({d['category']}): rules_correct={d['rules_correct']}, "
                      f"judge_verdict={d['judge_verdict']} (judge_correct={d['judge_correct']})")

    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "rules_vs_judge.md").write_text(report + "\n", encoding="utf-8")
    print(f"\nWritten -> {out_dir / 'rules_vs_judge.md'}")


if __name__ == "__main__":
    main()
