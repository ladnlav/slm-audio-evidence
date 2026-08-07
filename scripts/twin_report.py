"""Rebuild the twin-comparison tables of the paper from graded run outputs.

The twin subset asks the same 214 questions over two recordings of each passage --- one
synthesized, one human. The synthesized side lives in results/twin_tts/; the human side
is the matching slice of the validation runs in results/scale_nmsqa/, selected by the
question id that both manifests share. Because the design is paired, differences are
tested with McNemar on the discordant items rather than by comparing Wilson intervals.

Two tables come out:
  * the twin table  -- hallucination and accuracy in both conditions, per system/prompt;
  * the decomposition -- the diagnostic set, the twin's synthesized side and the twin's
    human side side by side, which moves one confound at a time (the first step also
    varies passages and trimming, so it is indicative rather than controlled).

Usage:
  py -3 scripts/twin_report.py                      # markdown to stdout
  py -3 scripts/twin_report.py --out results/twin_report.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TWIN_MANIFEST = "data/manifests/scale_tts_twin.jsonl"

# (label, prompt, synthesized-side run, human-side run, diagnostic-set run)
RUNS = [
    ("Qwen2-Audio", "plain",
     "results/twin_tts/qwen2audio_plain_20260807",
     "results/scale_nmsqa/qwen2audio_plain_20260725",
     "results/qwen2audio_plain_20260712"),
    ("Qwen2-Audio", "IDK",
     "results/twin_tts/qwen2audio_s1_idk_20260807",
     "results/scale_nmsqa/qwen2audio_s1_idk_20260725",
     "results/qwen2audio_s1_idk_20260712"),
    ("Cascade", "plain",
     "results/twin_tts/cascade_plain_20260806",
     "results/scale_nmsqa/cascade_plain_20260725",
     "results/cascade_plain_20260712"),
    ("Cascade", "IDK",
     "results/twin_tts/cascade_s1_idk_20260806",
     "results/scale_nmsqa/cascade_s1_idk_20260725",
     "results/cascade_s1_idk_20260712"),
]


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided McNemar, normal approximation with continuity correction.

    b and c are the discordant counts. With b + c == 0 the two conditions agree on
    every item and there is nothing to test, so the p-value is 1.
    """
    n = b + c
    if n == 0:
        return 1.0
    # max(0, ...) matters: the correction is meant to shrink the observed discrepancy
    # towards zero, not to manufacture one. Without the clamp, b == c yields
    # (0 - 1)**2 = 1 and a spuriously small p-value for two conditions that agreed
    # exactly as often in each direction.
    chi2 = max(0.0, abs(b - c) - 1) ** 2 / n
    return math.erfc(math.sqrt(chi2 / 2))


def load_graded(path: Path, id_prefix: str) -> dict[str, dict]:
    """Read responses_judged.jsonl keyed by the bare question id.

    Twin rows carry `sc-tts-<qid>` and validation rows `sc-nat-<qid>`; stripping the
    prefix is what lets the two sides be joined item by item.
    """
    rows = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        rows[row["id"].removeprefix(id_prefix)] = row
    return rows


def hallucinated(row: dict) -> bool:
    """A substantive answer to an unanswerable question."""
    return row["category"] == "C" and row["label"] == "answer"


def summarise(synth: dict[str, dict], human: dict[str, dict]) -> dict:
    shared = [q for q in synth if q in human]
    c_items = [q for q in shared if synth[q]["category"] == "C"]
    a_items = [q for q in shared if synth[q]["category"] == "A"]

    # Hallucination: paired over unanswerable items.
    b = sum(1 for q in c_items if not hallucinated(synth[q]) and hallucinated(human[q]))
    c = sum(1 for q in c_items if hallucinated(synth[q]) and not hallucinated(human[q]))

    # Accuracy: paired over answerable items.
    ab = sum(1 for q in a_items
             if synth[q]["correct"] is True and human[q]["correct"] is not True)
    ac = sum(1 for q in a_items
             if synth[q]["correct"] is not True and human[q]["correct"] is True)

    pct = lambda k, n: 100.0 * k / n if n else 0.0
    return {
        "n_c": len(c_items), "n_a": len(a_items),
        "hall_synth": pct(sum(hallucinated(synth[q]) for q in c_items), len(c_items)),
        "hall_human": pct(sum(hallucinated(human[q]) for q in c_items), len(c_items)),
        "hall_b": b, "hall_c": c, "hall_p": mcnemar_p(b, c),
        "acc_synth": pct(sum(synth[q]["correct"] is True for q in a_items), len(a_items)),
        "acc_human": pct(sum(human[q]["correct"] is True for q in a_items), len(a_items)),
        "acc_p": mcnemar_p(ab, ac),
    }


def fmt_p(p: float) -> str:
    if p < 1e-12:
        return "<1e-12"
    if p < 0.001:
        return f"{p:.0e}"
    return f"{p:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--out", default=None, help="Write markdown here instead of stdout.")
    args = ap.parse_args()
    root = Path(args.root)

    twin_ids = {json.loads(l)["id"].removeprefix("sc-tts-")
                for l in (root / TWIN_MANIFEST).open(encoding="utf-8")}

    lines = ["# Twin comparison", "",
             f"Paired over the {len(twin_ids)} twin questions: identical questions, "
             "identical passages, audio source varied. Both sides graded by the same judge.",
             "",
             "| System | Prompt | Halluc. synth | Halluc. human | b/c | p | Acc. synth | Acc. human | p |",
             "|---|---|---:|---:|:---:|---:|---:|---:|---:|"]

    decomp = []
    for label, prompt, synth_dir, human_dir, pilot_dir in RUNS:
        synth = load_graded(root / synth_dir / "responses_judged.jsonl", "sc-tts-")
        human = {k: v for k, v in
                 load_graded(root / human_dir / "responses_judged.jsonl", "sc-nat-").items()
                 if k in twin_ids}
        s = summarise(synth, human)
        lines.append(
            f"| {label} | {prompt} | {s['hall_synth']:.1f} | {s['hall_human']:.1f} | "
            f"{s['hall_b']}/{s['hall_c']} | {fmt_p(s['hall_p'])} | "
            f"{s['acc_synth']:.1f} | {s['acc_human']:.1f} | {fmt_p(s['acc_p'])} |")

        if prompt == "IDK":
            pilot = load_graded(root / pilot_dir / "responses_judged.jsonl", "")
            pilot_c = [r for r in pilot.values() if r["category"] == "C"]
            pilot_hall = 100.0 * sum(hallucinated(r) for r in pilot_c) / len(pilot_c)
            decomp.append((label, pilot_hall, s["hall_synth"], s["hall_human"]))

    lines += ["", f"n = {s['n_c']} unanswerable, {s['n_a']} answerable per cell.", "",
              "# Separating the two confounds", "",
              "Hallucination under the abstention instruction. The middle column shares its "
              "audio with the left and its questions with the right, so each step moves one "
              "factor -- imperfectly for the first (passages and trimming differ too), "
              "cleanly for the second.", "",
              "| System | generated Q, synth | native Q, synth | native Q, human | "
              "questions | audio |", "|---|---:|---:|---:|---:|---:|"]
    for label, pilot_h, synth_h, human_h in decomp:
        lines.append(f"| {label} | {pilot_h:.1f} | {synth_h:.1f} | {human_h:.1f} | "
                     f"{synth_h - pilot_h:+.1f} | {human_h - synth_h:+.1f} |")

    text = "\n".join(lines) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
