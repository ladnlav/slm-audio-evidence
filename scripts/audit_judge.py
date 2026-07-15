"""Audit the LLM judge against human grading (ROLE_M3 Task 4).

Three modes:

  0. compare (PREFERRED — use this first): the pilot's category-B answers were
     already hand-graded by M1 (results/<run_id>/responses_judged.jsonl,
     judge == "manual-M1", 93 items, committed 2026-07-13 — see docs/decisions.md).
     That grading is real ground truth, frozen before any LLM verdict existed, so
     there is no blind-sampling step to redo: run the LLM judge into a SEPARATE
     --out directory (never overwrite the committed file — manual grades are
     irreplaceable) and diff its cache against manual-M1 directly.
  1. sample / 2. score: the fallback for whenever ground truth does NOT already
     exist (a future run, a different category). Mirrors
     make_verification_sheet.py -> freeze_pilot.py: sample pulls every
     category-B item the LLM judge graded, takes a random sample, writes a
     BLIND CSV (no LLM verdict shown) for the primary grader plus a smaller
     recheck CSV for a second grader; score reads the filled sheets back and
     computes agreement.

Usage:
  # 0. compare against existing manual-M1 ground truth (no new human grading needed)
  py -3 -m src.run_eval --judge local \
      --responses results/qwen2audio_plain_20260712/responses.jsonl \
      --out results/qwen2audio_plain_20260712/llm_audit   # <- separate --out, on purpose
  # repeat per run_id, then:
  py -3 scripts/audit_judge.py compare --judged "results/*/responses_judged.jsonl"

  # 1/2. blind sampling, only when no ground truth exists yet
  py -3 scripts/audit_judge.py sample --judged "results/*/responses_judged.jsonl" \
      --manifest data/manifests/pilot.jsonl --out results/judge_audit
  # ... fill human_verdict (correct/incorrect/abstained) in the two CSVs ...
  py -3 scripts/audit_judge.py score --dir results/judge_audit

Gate (ROLE_M3): if agreement < 80%, do not report LLM-judge numbers on slides —
fall back to rules + manual grading and log that decision in docs/decisions.md.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `from src.judges...` in cmd_compare

AGREEMENT_GATE = 0.80


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def cmd_compare(args: argparse.Namespace) -> None:
    """Diff a fresh LLM-judge cache against the committed manual-M1 ground truth.

    No blind resampling: the human grades were frozen (2026-07-13, decisions.md)
    before any LLM verdict existed, so comparing against them directly cannot
    bias the human side. Reads ground truth from --judged's responses_judged.jsonl
    (judge == "manual-M1") and fresh verdicts from <run_dir>/<judge-subdir>/judge_cache.jsonl
    — a SEPARATE directory from --judged on purpose, so this never depends on
    (or risks) overwriting the committed manual grades.
    """
    from src.judges.base import Verdict  # local import: keeps this script runnable without torch/transformers

    rows: list[dict] = []
    for judged_path in sorted(glob.glob(args.judged, recursive=True)):
        judged_path = Path(judged_path)
        run_id = judged_path.parent.name
        cache_path = judged_path.parent / args.judge_subdir / "judge_cache.jsonl"
        cache = {row["id"]: row for row in load_jsonl(cache_path)}
        if not cache:
            continue
        for row in load_jsonl(judged_path):
            if row.get("judge") != "manual-M1":
                continue
            cached = cache.get(row["id"])
            if cached is None:
                continue  # this id wasn't (re-)judged by the fresh LLM run — skip rather than guess
            rows.append({
                "run_id": run_id, "id": row["id"],
                "human_correct": row["correct"],
                "llm_verdict": cached["verdict"], "judge_name": cached["judge_name"],
            })

    if not rows:
        raise SystemExit(
            f"No overlap between manual-M1 ground truth ({args.judged}) and a fresh judge cache "
            f"(<run_dir>/{args.judge_subdir}/judge_cache.jsonl). Run e.g.:\n"
            f"  py -3 -m src.run_eval --judge local --responses results/<run_id>/responses.jsonl "
            f"--out results/<run_id>/{args.judge_subdir}\n"
            "for each run_id first — into a SEPARATE --out, never over the committed responses_judged.jsonl."
        )

    agree = 0
    unparseable = 0
    confusion: Counter = Counter()
    for r in rows:
        llm_correct = Verdict(r["llm_verdict"]).to_correctness()
        if llm_correct is None:  # UNPARSEABLE — excluded from the agreement ratio, reported separately
            unparseable += 1
            continue
        confusion[(r["human_correct"], r["llm_verdict"])] += 1
        agree += llm_correct == r["human_correct"]

    scored = len(rows) - unparseable
    pct = 100 * agree / scored if scored else 0.0
    judge_names = {r["judge_name"] for r in rows}

    lines = [
        f"# LLM-judge vs manual-M1 ground truth ({len(rows)} overlapping items, {unparseable} UNPARSEABLE excluded)",
        f"Judge backend(s): {', '.join(sorted(judge_names))}",
        "",
        f"Agreement: **{agree}/{scored} = {pct:.0f}%**",
    ]
    if pct / 100 < AGREEMENT_GATE:
        lines.append(f"\n**BELOW {AGREEMENT_GATE*100:.0f}% GATE** — per ROLE_M3 Task 4, do not report LLM-judge "
                      "numbers; fall back to rules + manual grading and log this in docs/decisions.md.")
    lines.append("\n## Confusion (human correct/incorrect x LLM verdict)")
    for (human, verdict), n in sorted(confusion.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- human={human}, llm={verdict}: {n}")
    if unparseable:
        lines.append(f"\n{unparseable} item(s) the judge answered ambiguously (UNPARSEABLE) — not counted either way.")

    report = "\n".join(lines)
    print(report)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare_manual_m1.md").write_text(report + "\n", encoding="utf-8")
    print(f"\nWritten -> {out_dir / 'compare_manual_m1.md'} (paste the % into metrics.md / the results slide).")


def cmd_sample(args: argparse.Namespace) -> None:
    manifest = {row["id"]: row for row in load_jsonl(Path(args.manifest))}

    pool: list[dict] = []
    for judged_path in sorted(glob.glob(args.judged, recursive=True)) if isinstance(args.judged, str) else args.judged:
        judged_path = Path(judged_path)
        run_id = judged_path.parent.name
        cache = {
            (row["id"], row["judge_name"]): row["verdict"]
            for row in load_jsonl(judged_path.parent / "judge_cache.jsonl")
        }
        for row in load_jsonl(judged_path):
            judge_tag = row.get("judge", "")
            if not judge_tag.startswith("llm-"):
                continue  # audit is about the LLM judge specifically; rules have their own dev-set
            llm_verdict = cache.get((row["id"], judge_tag))
            if llm_verdict is None:
                continue  # cache missing/stale — skip rather than guess
            m = manifest.get(row["id"], {})
            pool.append({
                "sheet_id": f"{run_id}::{row['id']}",
                "id": row["id"],
                "run_id": run_id,
                "category": row["category"],
                "transcript": m.get("transcript", ""),
                "question": m.get("question", ""),
                "gold_answer": row["gold_answer"],
                "response": row["response"],
                "llm_verdict": llm_verdict,  # goes to the hidden answer key only
            })

    if not pool:
        raise SystemExit("No llm-judged category-B rows found. Run src/run_eval.py --judge local|gemini first.")

    rng = random.Random(42)
    rng.shuffle(pool)
    sample = pool[: args.n]
    recheck = sample[: args.recheck]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sheet_fields = ["sheet_id", "category", "transcript", "question", "gold_answer", "response", "human_verdict"]

    def write_sheet(rows: list[dict], path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=sheet_fields)
            writer.writeheader()
            for r in rows:
                writer.writerow({**{k: r[k] for k in sheet_fields if k != "human_verdict"}, "human_verdict": ""})

    write_sheet(sample, out_dir / "sheet_primary.csv")
    write_sheet(recheck, out_dir / "sheet_recheck.csv")

    # Hidden from graders on purpose: real LLM verdicts, read only by `score`.
    with (out_dir / "_answer_key.jsonl").open("w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps({"sheet_id": r["sheet_id"], "llm_verdict": r["llm_verdict"]}) + "\n")

    print(f"Sampled {len(sample)} items ({len(recheck)} also in the recheck sheet) -> {out_dir}")
    print("Fill 'human_verdict' with correct / incorrect / abstained in both CSVs (blind — no LLM verdict shown),")
    print("sheet_primary.csv by the main grader, sheet_recheck.csv by a second grader. Then run:")
    print(f"  py -3 scripts/audit_judge.py score --dir {out_dir}")


def _read_sheet(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    empty = [r["sheet_id"] for r in rows if not r["human_verdict"].strip()]
    if empty:
        raise SystemExit(f"{path}: {len(empty)} rows still have an empty human_verdict, e.g. {empty[:5]}")
    return {r["sheet_id"]: r["human_verdict"].strip().upper() for r in rows}


def cmd_score(args: argparse.Namespace) -> None:
    out_dir = Path(args.dir)
    answer_key = {row["sheet_id"]: row["llm_verdict"] for row in load_jsonl(out_dir / "_answer_key.jsonl")}
    primary = _read_sheet(out_dir / "sheet_primary.csv")

    agree = sum(1 for sid, verdict in primary.items() if answer_key.get(sid) == verdict)
    total = len(primary)
    pct = 100 * agree / total if total else 0.0

    lines = [f"# LLM-judge audit ({total} items)", "", f"LLM vs human agreement: **{agree}/{total} = {pct:.0f}%**"]
    if pct / 100 < AGREEMENT_GATE:
        lines.append(f"\n**BELOW {AGREEMENT_GATE*100:.0f}% GATE** — per ROLE_M3 Task 4, do not report LLM-judge "
                      "numbers; fall back to rules + manual grading and log this in docs/decisions.md.")

    recheck_path = out_dir / "sheet_recheck.csv"
    if recheck_path.exists():
        recheck = _read_sheet(recheck_path)
        hh_agree = sum(1 for sid, verdict in recheck.items() if primary.get(sid) == verdict)
        hh_total = len(recheck)
        hh_pct = 100 * hh_agree / hh_total if hh_total else 0.0
        lines.append(f"\nHuman-human agreement on recheck subset: {hh_agree}/{hh_total} = {hh_pct:.0f}% "
                      "(bounds how meaningful LLM-human disagreement is).")

    report = "\n".join(lines)
    print(report)
    (out_dir / "agreement.md").write_text(report + "\n", encoding="utf-8")
    print(f"\nWritten -> {out_dir / 'agreement.md'} (paste the % into metrics.md / the results slide).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_compare = sub.add_parser("compare", help="Compare a fresh LLM-judge cache against existing manual-M1 ground truth (preferred; no blind resampling).")
    p_compare.add_argument("--judged", required=True, help="Glob for the COMMITTED responses_judged.jsonl files, e.g. 'results/*/responses_judged.jsonl'.")
    p_compare.add_argument("--judge-subdir", default="llm_audit", help="Subdirectory under each run dir holding the fresh judge_cache.jsonl (default: llm_audit; must match the --out used for the audit run).")
    p_compare.add_argument("--out", default="results/judge_audit")
    p_compare.set_defaults(func=cmd_compare)

    p_sample = sub.add_parser("sample", help="Build blind grading sheets from llm-judged responses.")
    p_sample.add_argument("--judged", required=True, help="Glob for responses_judged.jsonl files, e.g. 'results/*/responses_judged.jsonl'.")
    p_sample.add_argument("--manifest", default="data/manifests/pilot.jsonl")
    p_sample.add_argument("--out", default="results/judge_audit")
    p_sample.add_argument("--n", type=int, default=30, help="Primary sample size (ROLE_M3 default: 30).")
    p_sample.add_argument("--recheck", type=int, default=10, help="Subset also sent to a second grader (default: 10).")
    p_sample.set_defaults(func=cmd_sample)

    p_score = sub.add_parser("score", help="Compute agreement from filled sheets.")
    p_score.add_argument("--dir", default="results/judge_audit")
    p_score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
