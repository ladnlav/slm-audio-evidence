"""Output-entropy baseline over k self-consistency samples (M4 Task 4, PLAN.md section 3 step 4).

The baseline the probe has to beat: instead of looking inside the model (M1's probe),
just sample k answers per question and measure how much they disagree. High disagreement
(high entropy) stands in for "the model is unsure here" without any judge call and
without touching the model's internals.

Clustering is done with rapidfuzz string similarity (src/judge.py's clean_text +
fuzz.partial_ratio, same threshold as category-A correctness) -- deliberately NOT an
extra LLM-judge call per pair, which would cost k*(k-1)/2 additional judge calls per
item on top of the ones scripts/soft_labels.py already spends. This is a placeholder
default, not a citation: the exact recipe should be checked against paper 25 ("Walking
Through Uncertainty", M4's Wed-22.07 reading) before these numbers go in the paper --
see the module-level TODO below.

Two subcommands:
  compute  responses.jsonl (needs a 'samples' list per id, e.g. --strategy s4_consistency)
           -> entropy_baseline.jsonl (id, n_samples, n_clusters, entropy, entropy_normalized)
  auroc    entropy_baseline.jsonl + a labels file (e.g. scripts/soft_labels.py's
           soft_labels.jsonl) -> a single AUROC number. NOT fold-aware -- this is a raw
           preview so a number exists to eyeball; the real reportable AUROC needs M1's
           cross-validation protocol/fold split (PLAN.md: "same folds as the probes"),
           still pending as of writing.

Usage:
  py -3 scripts/entropy_baseline.py compute --responses results/<run_id>/responses.jsonl \
      --out results/<run_id>/entropy_baseline.jsonl
  py -3 scripts/entropy_baseline.py auroc --entropy results/<run_id>/entropy_baseline.jsonl \
      --labels results/<run_id>/soft_labels/soft_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `from src...`

from rapidfuzz import fuzz  # noqa: E402

from src.judge import clean_text  # noqa: E402

CLUSTER_THRESHOLD = 85  # same threshold src/judge.py uses for category-A correctness


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def cluster_answers(samples: list[str], threshold: int = CLUSTER_THRESHOLD) -> list[int]:
    """Greedy single-link clustering: each sample joins the first existing cluster whose
    representative (its first member) matches above `threshold`, else starts a new one.
    Returns one cluster id per sample, same order as input.
    """
    reps: list[str] = []
    assignment: list[int] = []
    for raw in samples:
        text = clean_text(raw)
        matched = next(
            (cid for cid, rep in enumerate(reps) if fuzz.partial_ratio(text, rep) >= threshold), None
        )
        if matched is None:
            reps.append(text)
            matched = len(reps) - 1
        assignment.append(matched)
    return assignment


def shannon_entropy(assignment: list[int]) -> float:
    """Bits of entropy over the cluster-size distribution. 0 = all k samples agreed
    (one cluster); max = every sample its own cluster (k distinct clusters)."""
    k = len(assignment)
    counts = Counter(assignment)
    return -sum((c / k) * math.log2(c / k) for c in counts.values())


def normalized_entropy(entropy: float, k: int) -> float:
    """Scales to [0, 1] by the k-sample ceiling (log2(k)) -- makes items with a
    different k comparable. 0 if k<=1 (no disagreement possible to measure)."""
    return entropy / math.log2(k) if k > 1 else 0.0


def auroc(scores: list[float], labels: list[int]) -> float | None:
    """Mann-Whitney U / rank-sum AUROC, no sklearn dependency -- consistent with this
    project's existing hand-rolled stats (src/metrics.py's Wilson CI). `labels` must be
    0/1. Returns None if one class is empty (AUROC undefined)."""
    pairs = sorted(zip(scores, labels), key=lambda sl: sl[0])
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2  # 1-indexed, ties share the average rank
        for x in range(i, j + 1):
            ranks[x] = avg_rank
        i = j + 1

    n_pos = sum(1 for _, label in pairs if label == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum_pos = sum(r for r, (_, label) in zip(ranks, pairs) if label == 1)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def cmd_compute(args: argparse.Namespace) -> None:
    responses = load_jsonl(Path(args.responses))
    if not responses:
        raise SystemExit(f"No rows in {args.responses}")

    rows = []
    skipped = 0
    for row in responses:
        samples = row.get("samples")
        if not samples:
            skipped += 1
            continue
        assignment = cluster_answers(samples, threshold=args.threshold)
        n_clusters = len(set(assignment))
        entropy = shannon_entropy(assignment)
        rows.append({
            "id": row["id"], "n_samples": len(samples), "n_clusters": n_clusters,
            "entropy": entropy, "entropy_normalized": normalized_entropy(entropy, len(samples)),
        })

    if skipped:
        print(f"[!] {skipped} row(s) had no 'samples' field -- not from --strategy s4_consistency, skipped.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[+] {len(rows)} items -> {out_path}")


def bootstrap_ci(scores: list[float], labels: list[int], n: int, seed: int,
                 alpha: float = 0.05) -> tuple[float, float] | None:
    """Percentile bootstrap over items, matching scripts/train_probe.py.

    Resamples items rather than each class separately, so the interval reflects how few
    negatives a rerun of this experiment might contain -- the dominant source of
    uncertainty when the model behaves the same way on almost every item.
    """
    import random

    if n <= 0 or not (0 < sum(labels) < len(labels)):
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        idx = [rng.randrange(len(labels)) for _ in range(len(labels))]
        a = auroc([scores[i] for i in idx], [labels[i] for i in idx])
        if a is not None:
            vals.append(a)
    if not vals:
        return None
    vals.sort()
    lo = vals[int(len(vals) * alpha / 2)]
    hi = vals[min(len(vals) - 1, int(len(vals) * (1 - alpha / 2)))]
    return lo, hi


def cmd_auroc(args: argparse.Namespace) -> None:
    entropy_rows = {row["id"]: row for row in load_jsonl(Path(args.entropy))}
    label_rows = {row["id"]: row for row in load_jsonl(Path(args.labels))}
    common = sorted(set(entropy_rows) & set(label_rows))
    if not common:
        raise SystemExit(f"No overlapping ids between {args.entropy} and {args.labels}")

    scores = [entropy_rows[i][args.score_field] for i in common]
    labels = [1 if label_rows[i]["soft_label"] > args.threshold else 0 for i in common]
    n_pos = sum(labels)

    result = auroc(scores, labels)
    print(f"Items: {len(common)} (positive class 'soft_label > {args.threshold}': {n_pos}/{len(common)})")
    if result is None:
        print("AUROC undefined -- one class is empty in this sample (need more items or a different threshold).")
        return
    print(f"AUROC ({args.score_field} vs soft_label>{args.threshold}): {result:.3f}")

    ci = bootstrap_ci(scores, labels, args.bootstrap, args.seed)
    if ci:
        print(f"95% bootstrap CI: ({ci[0]:.3f}, {ci[1]:.3f})")
    print(
        "\nThe entropy score is not fitted to anything, so there are no folds to hold out -- "
        "this is the reportable number. For the comparison against the probes to be honest, "
        "both sides must use the same items and the same threshold on the soft label."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_compute = sub.add_parser("compute", help="Compute per-item entropy from k self-consistency samples.")
    p_compute.add_argument("--responses", required=True, help="s4_consistency responses.jsonl (has 'samples' per id).")
    p_compute.add_argument("--out", required=True)
    p_compute.add_argument("--threshold", type=int, default=CLUSTER_THRESHOLD,
                            help=f"rapidfuzz partial_ratio cluster threshold (default {CLUSTER_THRESHOLD}, "
                                 "same as category-A correctness in src/judge.py).")
    p_compute.set_defaults(func=cmd_compute)

    p_auroc = sub.add_parser("auroc", help="Raw (non-fold-aware) AUROC preview: entropy vs soft_label.")
    p_auroc.add_argument("--entropy", required=True, help="Output of the 'compute' subcommand.")
    p_auroc.add_argument("--labels", required=True, help="scripts/soft_labels.py's soft_labels.jsonl.")
    p_auroc.add_argument("--score-field", default="entropy_normalized", choices=["entropy", "entropy_normalized"])
    p_auroc.add_argument("--bootstrap", type=int, default=2000,
                         help="Resamples for the confidence interval; 0 disables it.")
    p_auroc.add_argument("--seed", type=int, default=0)
    p_auroc.add_argument("--threshold", type=float, default=0.0,
                          help="soft_label strictly above this counts as the positive class (default 0.0: "
                               "any hallucinated sample among k marks the item positive).")
    p_auroc.set_defaults(func=cmd_auroc)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
