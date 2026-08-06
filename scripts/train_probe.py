"""Train linear probes over captured representations and report AUROC (A2, PLAN.md section 3 step 3).

Inputs
  representations.npz  from scripts/capture_representations.py -- pooled prompt states,
                       one row per manifest item, several layers x poolings
  soft_labels.jsonl    from scripts/soft_labels.py -- per-item fraction of sampled answers
                       that were hallucinations (the probe's regression target)
  manifest             for the category of each item and the passage it belongs to

What it reports, and why each number is there:

  AUROC (overall)     the headline: can the probe rank risky prompts above safe ones?
  AUROC within C      the honest one. Category C has a far higher base rate of hallucination
                      than A, so a probe that merely learned "this is a C question" scores
                      well overall while being useless -- we already know the category from
                      the manifest. Ranking *within* C is what a deployed abstention gate
                      would actually have to do.
  base rates          so a suspiciously high AUROC can be traced to class imbalance.

Cross-validation is grouped by passage: every question about the same recording lands in the
same fold. Without that, the probe can memorise a passage from its A questions and be scored
on its C questions -- leakage that inflates AUROC and would not survive deployment.

Usage:
  py -3 scripts/train_probe.py \
      --representations results/probing/scale_nmsqa/representations.npz \
      --labels results/probing/<run>/soft_labels/soft_labels.jsonl \
      --manifest data/manifests/scale_nmsqa_final.jsonl \
      --out results/probing/scale_nmsqa/probes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Same grid as the reference implementation of paper 24 (their probes_sklearn.py): C is the
# INVERSE regularisation strength, so small C = heavy shrinkage. With ~376 items and 4096
# features, the useful part of this grid is its low end.
C_GRID = (0.01, 0.1, 0.3, 0.5, 0.8)
DEFAULT_FOLDS = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train linear probes per layer/pooling, report AUROC.")
    p.add_argument("--representations", required=True, help="representations.npz from capture_representations.py")
    p.add_argument("--labels", required=True, help="soft_labels.jsonl (id -> soft_label)")
    p.add_argument("--manifest", required=True, help="Manifest, for category and passage grouping.")
    p.add_argument("--out", required=True, help="Output directory for the AUROC table and per-item scores.")
    p.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Soft label above this counts as a positive when computing AUROC "
                        "(the label is a fraction; AUROC needs a binary ground truth).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def align_data(
    npz_path: str | Path, labels_path: str | Path, manifest_path: str | Path
) -> tuple[list[str], dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Join representations, labels and manifest on item id.

    Items missing from either side are dropped with a warning rather than silently zero-filled:
    a missing label is not "no hallucination", and a missing representation is not a zero vector.
    """
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    rep_ids = [str(i) for i in arrays.pop("ids")]

    labels = {r["id"]: float(r["soft_label"]) for r in load_jsonl(labels_path)}
    manifest = {r["id"]: r for r in load_jsonl(manifest_path)}

    keep_idx, ids = [], []
    for i, item_id in enumerate(rep_ids):
        if item_id in labels and item_id in manifest:
            keep_idx.append(i)
            ids.append(item_id)
    dropped = len(rep_ids) - len(ids)
    if dropped:
        print(f"[!] {dropped} item(s) had a representation but no label (or no manifest row) -- dropped")
    if not ids:
        raise ValueError("No items left after joining representations with labels")

    keep = np.array(keep_idx)
    features = {name: arr[keep] for name, arr in arrays.items()}
    y_soft = np.array([labels[i] for i in ids])
    categories = np.array([manifest[i]["category"] for i in ids])
    # Group by passage so that questions about one recording never straddle a fold boundary.
    groups = np.array([manifest[i].get("audio_path", manifest[i]["id"]) for i in ids])
    return ids, features, y_soft, categories, groups


def cross_val_scores(X: np.ndarray, y_soft: np.ndarray, groups: np.ndarray,
                     folds: int, seed: int) -> np.ndarray:
    """Out-of-fold probe scores, one per item. Trained on the soft label, as in paper 24."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n_groups = len(set(groups))
    n_splits = min(folds, n_groups)
    if n_splits < 2:
        raise ValueError(f"Need at least 2 passage groups for CV, got {n_groups}")

    scores = np.full(len(y_soft), np.nan)
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y_soft, groups):
        # Ridge, not LogisticRegression: the target is a fraction in [0,1], not a class. Fitting
        # the fraction directly is what soft-target supervision means -- thresholding it first
        # would throw away exactly the information paper 24 shows is worth keeping.
        model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=seed))])
        model.fit(X[train_idx], y_soft[train_idx])
        scores[test_idx] = model.predict(X[test_idx])
    return scores


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Rank-based AUROC; None when one class is absent (undefined, not 0.5)."""
    pos, neg = labels.astype(bool), ~labels.astype(bool)
    if not pos.any() or not neg.any():
        return None
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties, so identical scores cannot inflate the statistic
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(unique))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids, features, y_soft, categories, groups = align_data(
        args.representations, args.labels, args.manifest
    )
    y_bin = (y_soft > args.threshold).astype(int)
    is_c = categories == "C"

    print(f"items {len(ids)} | passages {len(set(groups))} | "
          f"positives {int(y_bin.sum())}/{len(y_bin)} ({y_bin.mean():.1%})")
    print(f"soft labels: min {y_soft.min():.2f} median {np.median(y_soft):.2f} max {y_soft.max():.2f} | "
          f"degenerate (exactly 0 or 1): {np.mean((y_soft == 0) | (y_soft == 1)):.0%}")
    if is_c.any():
        print(f"base rate: C {y_bin[is_c].mean():.1%} vs non-C {y_bin[~is_c].mean():.1%}")
    print()

    rows = []
    per_item_scores: dict[str, np.ndarray] = {}
    for name in sorted(features):
        X = features[name].astype(np.float32)
        scores = cross_val_scores(X, y_soft, groups, args.folds, args.seed)
        per_item_scores[name] = scores
        overall = auroc(scores, y_bin)
        within_c = auroc(scores[is_c], y_bin[is_c]) if is_c.sum() > 1 else None
        rows.append({
            "features": name,
            "auroc_overall": overall,
            "auroc_within_c": within_c,
            "n_items": len(ids),
        })
        fmt = lambda v: f"{v:.3f}" if v is not None else "  n/a"
        print(f"{name:24s} AUROC {fmt(overall)}   within-C {fmt(within_c)}")

    rows.sort(key=lambda r: (r["auroc_overall"] is None, -(r["auroc_overall"] or 0)))
    best = rows[0]
    print(f"\nbest overall: {best['features']} (AUROC {best['auroc_overall']:.3f})")

    (out_dir / "auroc.json").write_text(
        json.dumps({"rows": rows, "threshold": args.threshold, "folds": args.folds,
                    "n_passages": len(set(groups))}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    np.savez_compressed(out_dir / "probe_scores.npz", ids=np.array(ids),
                        y_soft=y_soft, category=categories, **per_item_scores)

    lines = ["| features | AUROC | AUROC within C |", "|---|---:|---:|"]
    for r in rows:
        f = lambda v: f"{v:.3f}" if v is not None else "n/a"
        lines.append(f"| `{r['features']}` | {f(r['auroc_overall'])} | {f(r['auroc_within_c'])} |")
    (out_dir / "auroc.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[+] {out_dir/'auroc.md'}, {out_dir/'auroc.json'}, {out_dir/'probe_scores.npz'}")


if __name__ == "__main__":
    main()
