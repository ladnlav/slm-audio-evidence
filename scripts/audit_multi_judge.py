"""Compare several judge-model candidates against manual-M1 in one pass (ROLE_M4 task 1's
model ladder -- see docs/decisions.md 2026-07-21).

This script runs no models itself. Each candidate is one judge config (model + thinking
on/off) already graded into its own results/<run_id>/llm_audit_<judge_name>/ by run_evaluation()
-- see notebooks/kaggle_run.ipynb's candidate loop, which also writes --manifest (a JSON list
of {judge_name, model_id, backend, thinking, seconds}, one entry per candidate it ran).

Produces, under --out (default results/judge_audit_multi/):
  comparison.md / comparison.csv     One table, every candidate side by side, sorted best
                                      agreement first -- this is what goes in the PR description.
  <judge_name>/compare_manual_m1.md,
  <judge_name>/disagreements.jsonl   Per-candidate detail: exactly which items it got wrong
                                      vs manual-M1, full text (same shape as `audit_judge.py
                                      compare`'s own output).
  thinking_traces/<judge_name>/      ONLY for candidates with "thinking": true in the manifest:
                                      one file per judged item with the full raw judge output
                                      (the <think>...</think> block), for every item -- not
                                      just disagreements. Kept out of the tables above on
                                      purpose: these are for spot-checking reasoning quality,
                                      not part of the comparison read first.

Usage:
  py -3 scripts/audit_multi_judge.py --manifest results/judge_audit_multi/run_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

from audit_judge import AGREEMENT_GATE, compute_agreement, load_jsonl, _truncate


def extract_thinking_traces(judge_subdir: str, judged_glob: str, out_dir: Path) -> int:
    """Full raw_output for every judged item under this candidate's judge_subdir -- one file
    per item, so a <think> block is readable without grepping judge_cache.jsonl by hand.
    """
    written = 0
    for judged_path in sorted(glob.glob(judged_glob, recursive=True)):
        run_dir = Path(judged_path).parent
        run_id = run_dir.name
        cache_path = run_dir / judge_subdir / "judge_cache.jsonl"
        for row in load_jsonl(cache_path):
            item_path = out_dir / f"{run_id}__{row['id']}.md"
            item_path.write_text(
                f"# {run_id} / {row['id']}\n\n"
                f"Verdict: **{row['verdict']}**\n\n"
                "## Raw judge output\n\n"
                f"```\n{row['raw_output']}\n```\n",
                encoding="utf-8",
            )
            written += 1
    return written


def build_candidate_report(judge_name: str, result: dict) -> str:
    pct = 100 * result["agree"] / result["scored"] if result["scored"] else 0.0
    lines = [
        f"# {judge_name} vs manual-M1 ({result['scored']} scored, {result['unparseable']} unparseable)",
        "",
        f"Agreement: **{result['agree']}/{result['scored']} = {pct:.0f}%**",
        "",
    ]
    if pct / 100 < AGREEMENT_GATE:
        lines.append(f"**BELOW {AGREEMENT_GATE*100:.0f}% GATE** — per ROLE_M3 Task 4, do not report this "
                      "candidate's numbers; fall back to rules + manual grading and log this in docs/decisions.md.")
        lines.append("")
    lines += [
        f"## Disagreements ({len(result['disagreements'])} of {result['scored']})",
        "",
    ]
    for r in result["disagreements"]:
        human_label = "correct" if r["human_correct"] else "incorrect"
        lines.append(f"### {r['id']} — {r['run_id']} (category {r['category']})")
        if r["question"]:
            lines.append(f"- Question: {_truncate(r['question'])}")
        lines.append(f"- Gold: {_truncate(r['gold_answer'])}")
        lines.append(f"- Response: {_truncate(r['response'])}")
        lines.append(f"- **Human (manual-M1): {human_label}** vs **LLM: {r['llm_verdict']}**")
        if r["llm_raw_output"] != r["llm_verdict"]:
            lines.append(f"  (raw judge output: {_truncate(r['llm_raw_output'], 150)})")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True,
                        help="JSON list of {judge_name, model_id, backend, thinking, seconds} -- "
                             "written by the notebook's candidate loop, one entry per model tested.")
    parser.add_argument("--judged", default="results/*/responses_judged.jsonl",
                        help="Glob for the committed responses_judged.jsonl files (manual-M1 ground truth).")
    parser.add_argument("--data-manifest", default="data/manifests/pilot.jsonl",
                        help="Eval-set manifest, for question text in disagreement rows (pass '' to skip).")
    parser.add_argument("--out", default="results/judge_audit_multi")
    args = parser.parse_args()

    candidates = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if not candidates:
        raise SystemExit(f"{args.manifest} is empty -- nothing to compare.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    table_rows = []
    for cand in candidates:
        judge_name = cand["judge_name"]
        judge_subdir = f"llm_audit_{judge_name}"
        result = compute_agreement(args.judged, judge_subdir, args.data_manifest)

        if not result["rows"]:
            print(f"[!] {judge_name}: no overlap with manual-M1 under {judge_subdir}/ -- skipping "
                  "(did this candidate actually run?)")
            continue

        cand_dir = out_dir / judge_name
        cand_dir.mkdir(parents=True, exist_ok=True)
        (cand_dir / "compare_manual_m1.md").write_text(build_candidate_report(judge_name, result), encoding="utf-8")
        with (cand_dir / "disagreements.jsonl").open("w", encoding="utf-8") as f:
            for r in result["disagreements"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        n_traces = 0
        if cand.get("thinking"):
            trace_dir = out_dir / "thinking_traces" / judge_name
            trace_dir.mkdir(parents=True, exist_ok=True)
            n_traces = extract_thinking_traces(judge_subdir, args.judged, trace_dir)

        pct = 100 * result["agree"] / result["scored"] if result["scored"] else 0.0
        seconds = cand.get("seconds")
        # Only category-B "answer" items actually trigger an LLM call (src/run_eval.py) --
        # everything else in the timed loop is resolved by the near-instant rule-based path, so
        # seconds/scored is a good proxy for per-call judge latency, not just an average over
        # every item run_evaluation() touched.
        sec_per_item = seconds / result["scored"] if seconds is not None and result["scored"] else None
        table_rows.append({
            "judge_name": judge_name,
            "model_id": cand.get("model_id", ""),
            "backend": cand.get("backend", ""),
            "thinking": bool(cand.get("thinking", False)),
            "agreement_pct": round(pct, 1),
            "agree": result["agree"],
            "scored": result["scored"],
            "unparseable": result["unparseable"],
            "n_disagreements": len(result["disagreements"]),
            "seconds": seconds,
            "sec_per_item": round(sec_per_item, 2) if sec_per_item is not None else None,
            "thinking_traces_written": n_traces,
        })

    if not table_rows:
        raise SystemExit("No candidate produced any overlapping rows -- nothing to write. "
                          "Check that the notebook loop actually ran run_evaluation() for each one.")

    table_rows.sort(key=lambda r: -r["agreement_pct"])

    header = ["Judge", "Model", "Thinking", "Agreement", "Scored", "Unparseable", "Disagreements", "Time (s)", "s/item"]
    md = [
        "# Judge-model comparison vs manual-M1 (93 gold B labels)",
        "",
        "Sorted by agreement, best first. Per-model failures: `<judge_name>/compare_manual_m1.md` "
        "+ `disagreements.jsonl` in this same folder. Full reasoning traces for thinking-enabled "
        "candidates: `thinking_traces/<judge_name>/`. `s/item` = Time (s) / Scored -- only "
        "category-B \"answer\" items actually call the judge, so this approximates per-call latency.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for r in table_rows:
        time_str = f"{r['seconds']:.0f}" if r.get("seconds") is not None else "?"
        sec_per_item_str = f"{r['sec_per_item']:.2f}" if r.get("sec_per_item") is not None else "?"
        md.append("| " + " | ".join([
            r["judge_name"],
            r["model_id"] or r["backend"],
            "yes" if r["thinking"] else "no",
            f"{r['agreement_pct']:.0f}% ({r['agree']}/{r['scored']})",
            str(r["scored"]),
            str(r["unparseable"]),
            str(r["n_disagreements"]),
            time_str,
            sec_per_item_str,
        ]) + " |")
    comparison_md = "\n".join(md) + "\n"
    (out_dir / "comparison.md").write_text(comparison_md, encoding="utf-8")

    with (out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)

    print(comparison_md)
    print(f"Written -> {out_dir / 'comparison.md'} / comparison.csv (the top-level table)")
    print(f"        -> {out_dir}/<judge_name>/compare_manual_m1.md + disagreements.jsonl (per-model failures)")
    print(f"        -> {out_dir}/thinking_traces/<judge_name>/ (full reasoning traces, thinking candidates only)")


if __name__ == "__main__":
    main()
