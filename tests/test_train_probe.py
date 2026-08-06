"""Dry-run tests for scripts/train_probe.py — synthetic data, no GPU, no captured runs.

Two things here are worth guarding beyond "it runs":

1. **AUROC must be right.** It is the number the whole probing contribution rests on, and a
   subtly wrong implementation (ties mishandled, labels flipped) would quietly report a
   plausible-looking 0.6 instead of the truth. So it is checked against cases whose value is
   known by hand: perfect ranking, inverted ranking, all-ties.

2. **Grouping must actually prevent leakage.** With questions from one passage split across
   folds, a probe can memorise the passage and score far above what it would achieve in
   deployment. The test builds data where the passage identity alone predicts the label and
   asserts that grouped CV does NOT reward that.

Run: python tests/test_train_probe.py   (or: pytest tests/test_train_probe.py)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_probe import align_data, auroc, cross_val_scores  # noqa: E402


def _check(cases: list[tuple[bool, str]]) -> int:
    failures = 0
    for ok, description in cases:
        print(f"  [{'ok' if ok else 'FAIL'}] {description}")
        failures += 0 if ok else 1
    return failures


def test_auroc_known_values() -> int:
    """Hand-checkable cases, including the worked example from the toy-examples write-up."""
    print("test_auroc_known_values")
    return _check([
        (auroc(np.array([0.9, 0.4, 0.6, 0.3, 0.1]), np.array([1, 1, 0, 0, 0])) == 5 / 6,
         "worked example: 5 of 6 pairs ranked correctly -> 5/6"),
        (auroc(np.array([0.9, 0.8]), np.array([1, 0])) == 1.0, "perfect ranking -> 1.0"),
        (auroc(np.array([0.1, 0.9]), np.array([1, 0])) == 0.0, "inverted ranking -> 0.0"),
        (auroc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([1, 1, 0, 0])) == 0.5,
         "all scores tied -> 0.5, not 1.0 (ties get average ranks)"),
        (auroc(np.array([1.0, 2.0]), np.array([1, 1])) is None,
         "one class absent -> None, never a made-up 0.5"),
    ])


def test_grouping_blocks_leakage() -> int:
    """Passage identity predicts the label; grouped CV must not be fooled by it.

    Each passage gets a constant feature vector and a constant label. A probe that memorises
    passages scores perfectly when folds are random, and no better than chance when the whole
    passage is held out -- which is what GroupKFold must enforce.
    """
    print("test_grouping_blocks_leakage")
    rng = np.random.default_rng(0)
    n_passages, per_passage, dim = 8, 5, 6
    X, y, groups = [], [], []
    for p in range(n_passages):
        signature = rng.normal(size=dim) * 5  # identifies the passage, unrelated to risk
        label = float(p % 2)                  # label is a property of the passage only
        for _ in range(per_passage):
            X.append(signature + rng.normal(size=dim) * 0.01)
            y.append(label)
            groups.append(f"passage-{p}")
    X, y, groups = np.array(X), np.array(y), np.array(groups)

    grouped = cross_val_scores(X, y, groups, folds=4, seed=0)
    grouped_auroc = auroc(grouped, (y > 0.5).astype(int))
    # Same data, but folds ignore passages: every held-out item has near-identical twins in train.
    fake_groups = np.array([f"item-{i}" for i in range(len(y))])
    ungrouped = cross_val_scores(X, y, fake_groups, folds=4, seed=0)
    ungrouped_auroc = auroc(ungrouped, (y > 0.5).astype(int))

    print(f"      grouped AUROC {grouped_auroc:.3f} vs ungrouped {ungrouped_auroc:.3f}")
    return _check([
        (ungrouped_auroc > 0.9, "without grouping the probe memorises passages (AUROC > 0.9)"),
        (grouped_auroc < ungrouped_auroc, "grouping by passage lowers the score -- leakage is blocked"),
    ])


def test_probe_finds_real_signal() -> int:
    """Sanity: when the label really is a linear function of the features, AUROC should be high."""
    print("test_probe_finds_real_signal")
    rng = np.random.default_rng(1)
    n_passages, per_passage, dim = 10, 4, 8
    direction = rng.normal(size=dim)
    X, y, groups = [], [], []
    for p in range(n_passages):
        for _ in range(per_passage):
            x = rng.normal(size=dim)
            X.append(x)
            y.append(1 / (1 + np.exp(-x @ direction)))  # smooth risk in [0,1]
            groups.append(f"passage-{p}")
    X, y, groups = np.array(X), np.array(y), np.array(groups)
    scores = cross_val_scores(X, y, groups, folds=5, seed=0)
    value = auroc(scores, (y > 0.5).astype(int))
    print(f"      AUROC {value:.3f}")
    return _check([(value > 0.8, "recovers a genuinely linear signal (AUROC > 0.8)")])


def test_align_drops_unmatched_items() -> int:
    """An item without a label must be dropped, not treated as risk-free."""
    print("test_align_drops_unmatched_items")
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        npz, labels, manifest = tmp / "r.npz", tmp / "l.jsonl", tmp / "m.jsonl"
        np.savez(npz, ids=np.array(["a", "b", "c"]), layer_8_last=np.arange(9, dtype=np.float32).reshape(3, 3))
        # 'c' has no label; 'b' has no manifest row
        labels.write_text("\n".join(json.dumps(r) for r in [
            {"id": "a", "soft_label": 0.9}, {"id": "b", "soft_label": 0.1}]) + "\n", encoding="utf-8")
        manifest.write_text("\n".join(json.dumps(r) for r in [
            {"id": "a", "category": "C", "audio_path": "p1.wav"},
            {"id": "c", "category": "A", "audio_path": "p2.wav"}]) + "\n", encoding="utf-8")

        ids, features, y_soft, categories, groups = align_data(npz, labels, manifest)
        return _check([
            (ids == ["a"], f"only the fully-matched item survives (got {ids})"),
            (features["layer_8_last"].shape == (1, 3), "features sliced to the surviving rows"),
            (float(y_soft[0]) == 0.9, "label follows the id, not the row order"),
            (list(categories) == ["C"] and list(groups) == ["p1.wav"], "category and group carried over"),
        ])


def main() -> int:
    failures = (
        test_auroc_known_values()
        + test_grouping_blocks_leakage()
        + test_probe_finds_real_signal()
        + test_align_drops_unmatched_items()
    )
    print("\nOK" if failures == 0 else f"\n{failures} FAILURE(S)")
    return failures


def test_auroc(): assert test_auroc_known_values() == 0
def test_leakage(): assert test_grouping_blocks_leakage() == 0
def test_signal(): assert test_probe_finds_real_signal() == 0
def test_align(): assert test_align_drops_unmatched_items() == 0


if __name__ == "__main__":
    sys.exit(main())
