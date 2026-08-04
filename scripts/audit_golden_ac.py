"""Golden A/C set: judge agreement outside category B (M4 Task 2 step 3, PLAN.md).

The 94% agreement number so far (docs/decisions.md) is measured ONLY on category B
(93 manual-M1 items) -- the categories the paper's headline metrics actually run on
(hallucination_rate / correct_refusal_rate on C, accuracy_a on A) have never been
checked against the judge at all. This script samples a fresh ~80-item set skewed
toward C, has a human grade it blind, and separately asks the LLM judge to grade the
SAME items (production judge only ever fires on category B -- for A/C the rules
classifier is authoritative, so getting the judge's opinion on these items requires
calling it directly, bypassing that gate).

Verdict <-> 3-way label mapping (judge_v1.txt only knows CORRECT/INCORRECT/ABSTAINED,
it has no HEDGE option -- see src/prompts/judge_v1.txt): a human grade of "answer"+
correct maps to CORRECT, "answer"+incorrect maps to INCORRECT, and both "abstain" and
"hedge" map to ABSTAINED (same collapsing Verdict.to_correctness() already does for B).

Two subcommands, mirroring scripts/audit_judge.py's sample/score pattern:
  sample  responses.jsonl (glob, categories A and C) -> blind CSV (no judge verdict
          shown) + a hidden answer key with the judge's verdict.
  score   filled CSV -> agreement %, written next to metrics.md style.

Usage:
  py -3 scripts/audit_golden_ac.py sample --responses "results/*/responses.jsonl" \
      --manifest data/manifests/scale_nmsqa.jsonl --out results/golden_ac \
      --judge local --judge-model <path> --judge-display-name "Qwen/Qwen3-8B"
  # ... fill human_label (answer/abstain/hedge) and, for category A only,
  #     human_correct (yes/no) in results/golden_ac/sheet.csv ...
  py -3 scripts/audit_golden_ac.py score --dir results/golden_ac
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `from src...`

from src.judges import DEFAULT_PROMPT_NAME, Verdict, build_judge  # noqa: E402

AGREEMENT_GATE = 0.80  # same bar as scripts/audit_judge.py's B-only audit


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def human_to_verdict(human_label: str, human_correct: str) -> str | None:
    """See module docstring for the mapping rationale. Returns None on malformed input
    (caller should treat as a data-entry error, not silently guess a verdict)."""
    label = human_label.strip().lower()
    if label in ("abstain", "hedge"):
        return Verdict.ABSTAINED.value
    if label == "answer":
        correct = human_correct.strip().lower()
        if correct == "yes":
            return Verdict.CORRECT.value
        if correct == "no":
            return Verdict.INCORRECT.value
        return None  # answer without a filled-in correctness call
    return None


def cmd_sample(args: argparse.Namespace) -> None:
    manifest = {row["id"]: row for row in load_jsonl(Path(args.manifest))}

    pool_c: list[dict] = []
    pool_a: list[dict] = []
    for responses_path in sorted(glob.glob(args.responses, recursive=True)):
        run_id = Path(responses_path).parent.name
        for row in load_jsonl(Path(responses_path)):
            m = manifest.get(row["id"])
            if m is None or m["category"] not in ("A", "C"):
                continue
            item = {
                "sheet_id": f"{run_id}::{row['id']}", "id": row["id"], "run_id": run_id,
                "category": m["category"], "transcript": m.get("transcript", ""),
                "question": m.get("question", ""), "gold_answer": m["gold_answer"],
                "response": row["response"],
            }
            (pool_c if m["category"] == "C" else pool_a).append(item)

    if not pool_c and not pool_a:
        raise SystemExit(f"No category A/C responses found matching {args.responses} against {args.manifest}")

    rng = random.Random(42)
    rng.shuffle(pool_c)
    rng.shuffle(pool_a)
    sample = pool_c[: args.n_c] + pool_a[: args.n_a]
    if len(pool_c) < args.n_c:
        print(f"[!] only {len(pool_c)}/{args.n_c} category-C candidates available")
    if len(pool_a) < args.n_a:
        print(f"[!] only {len(pool_a)}/{args.n_a} category-A candidates available")
    rng.shuffle(sample)

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
    answer_key = []
    with (out_dir / "judge_cache.jsonl").open("w", encoding="utf-8") as cache_file:
        for i, item in enumerate(sample, 1):
            try:
                result = judge.judge(item["transcript"], item["question"], item["gold_answer"], item["response"])
                verdict_value = result.verdict.value
            except Exception as exc:
                verdict_value, result = Verdict.UNPARSEABLE.value, None
                print(f"[judge] #{i}/{len(sample)} {item['id']} FAILED: {exc}")
            else:
                print(f"[judge] #{i}/{len(sample)} {item['id']} ({item['category']}) {verdict_value}")
            answer_key.append({"sheet_id": item["sheet_id"], "llm_verdict": verdict_value})
            cache_file.write(json.dumps({
                "sheet_id": item["sheet_id"], "verdict": verdict_value,
                "raw_output": result.raw_output if result else None,
            }, ensure_ascii=False) + "\n")

    sheet_fields = [
        "sheet_id", "category", "transcript", "question", "gold_answer", "response",
        "human_label", "human_correct",
    ]
    with (out_dir / "sheet.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=sheet_fields)
        writer.writeheader()
        for item in sample:
            writer.writerow({**{k: item[k] for k in sheet_fields if k in item}, "human_label": "", "human_correct": ""})

    with (out_dir / "_answer_key.jsonl").open("w", encoding="utf-8") as f:
        for row in answer_key:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSampled {len(sample)} items ({sum(1 for i in sample if i['category']=='C')} C, "
          f"{sum(1 for i in sample if i['category']=='A')} A) -> {out_dir}")
    print("Fill 'human_label' (answer/abstain/hedge) in sheet.csv, blind (no LLM verdict shown).")
    print("For rows with human_label=answer, also fill 'human_correct' (yes/no) -- required for category A,")
    print("meaningless for C (a substantive answer on C is always a hallucination regardless of correctness).")
    print(f"Then run: py -3 scripts/audit_golden_ac.py score --dir {out_dir}")


def cmd_score(args: argparse.Namespace) -> None:
    out_dir = Path(args.dir)
    answer_key = {row["sheet_id"]: row["llm_verdict"] for row in load_jsonl(out_dir / "_answer_key.jsonl")}

    with (out_dir / "sheet.csv").open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    malformed = []
    agree_c = agree_a = total_c = total_a = 0
    disagreements = []
    for row in rows:
        sheet_id, category = row["sheet_id"], row["category"]
        human_verdict = human_to_verdict(row["human_label"], row["human_correct"])
        if human_verdict is None:
            malformed.append(sheet_id)
            continue
        llm_verdict = answer_key.get(sheet_id)
        if llm_verdict == Verdict.UNPARSEABLE.value:
            continue  # excluded from the ratio, same convention as audit_judge.py
        match = human_verdict == llm_verdict
        if category == "C":
            total_c += 1
            agree_c += match
        else:
            total_a += 1
            agree_a += match
        if not match:
            disagreements.append({**row, "llm_verdict": llm_verdict, "human_verdict": human_verdict})

    if malformed:
        raise SystemExit(f"{len(malformed)} row(s) missing/invalid human_label or human_correct, e.g. {malformed[:5]}")

    total = total_c + total_a
    agree = agree_c + agree_a
    pct = 100 * agree / total if total else 0.0
    pct_c = 100 * agree_c / total_c if total_c else 0.0
    pct_a = 100 * agree_a / total_a if total_a else 0.0

    lines = [
        f"# LLM-judge vs human, category A/C golden set ({total} items: {total_c} C, {total_a} A)",
        "",
        f"Overall agreement: **{agree}/{total} = {pct:.0f}%**",
        f"- Category C: {agree_c}/{total_c} = {pct_c:.0f}%",
        f"- Category A: {agree_a}/{total_a} = {pct_a:.0f}%",
        "",
        "For Methods: \"judge agreement 94% on B (n=93), "
        f"{pct:.0f}% on A/C (n={total})\".",
    ]
    if pct / 100 < AGREEMENT_GATE:
        lines.append(f"\n**BELOW {AGREEMENT_GATE*100:.0f}% GATE** -- same bar as the B audit "
                      "(scripts/audit_judge.py); consider whether A/C should stay rules-only.")
    if disagreements:
        lines.append(f"\n## Disagreements ({len(disagreements)})")
        for d in disagreements:
            lines.append(f"- {d['sheet_id']} ({d['category']}): human={d['human_verdict']}, llm={d['llm_verdict']}")

    report = "\n".join(lines)
    print(report)
    (out_dir / "agreement.md").write_text(report + "\n", encoding="utf-8")
    print(f"\nWritten -> {out_dir / 'agreement.md'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample", help="Sample A/C items, grade them with the judge, write a blind human sheet.")
    p_sample.add_argument("--responses", required=True, help="Glob for responses.jsonl, e.g. 'results/*/responses.jsonl'.")
    p_sample.add_argument("--manifest", required=True)
    p_sample.add_argument("--out", default="results/golden_ac")
    p_sample.add_argument("--n-c", type=int, default=50, help="Category-C sample size (default 50).")
    p_sample.add_argument("--n-a", type=int, default=30, help="Category-A sample size (default 30; 50+30=80 per PLAN.md).")
    p_sample.add_argument("--judge", choices=["local", "gemini", "fake"], default="local")
    p_sample.add_argument("--judge-model", default=None)
    p_sample.add_argument("--judge-display-name", default=None)
    p_sample.add_argument("--judge-prompt", default=DEFAULT_PROMPT_NAME)
    p_sample.add_argument("--judge-thinking", action="store_true")
    p_sample.set_defaults(func=cmd_sample)

    p_score = sub.add_parser("score", help="Compute agreement from the filled sheet.")
    p_score.add_argument("--dir", default="results/golden_ac")
    p_score.set_defaults(func=cmd_score)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
