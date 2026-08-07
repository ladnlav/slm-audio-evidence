"""Tests for the twin-comparison report.

The paired statistic is the part worth pinning down: McNemar looks only at the items
where the two conditions disagree, so a large difference in rates with few discordant
pairs is not the same evidence as a small difference with many.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from twin_report import mcnemar_p, summarise  # noqa: E402


def test_no_discordant_pairs_is_not_evidence():
    """Both conditions agree on every item -- nothing to test."""
    assert mcnemar_p(0, 0) == 1.0


def test_symmetric_discordance_is_not_evidence():
    assert mcnemar_p(10, 10) == pytest.approx(1.0, abs=1e-9)


def test_one_sided_discordance_is_strong():
    """The cascade's IDK row: 72 items one way, 3 the other."""
    assert mcnemar_p(72, 3) < 1e-12


def test_moderate_discordance():
    """The cascade's plain row: 33 against 4, reported in the paper as 4e-06."""
    assert mcnemar_p(33, 4) == pytest.approx(4.2e-06, rel=0.1)


def test_direction_does_not_change_the_p_value():
    """Two-sided: swapping which condition wins leaves the evidence unchanged."""
    assert mcnemar_p(33, 4) == mcnemar_p(4, 33)


def _row(qid, category, label, correct):
    return {"id": qid, "category": category, "label": label, "correct": correct}


def test_summarise_counts_hallucinations_and_accuracy():
    # Two unanswerable items: the first is hallucinated only in the human condition,
    # the second in neither. One answerable item, correct only under synthesis.
    synth = {
        "q1": _row("sc-tts-q1", "C", "abstain", None),
        "q2": _row("sc-tts-q2", "C", "abstain", None),
        "q3": _row("sc-tts-q3", "A", "answer", True),
    }
    human = {
        "q1": _row("sc-nat-q1", "C", "answer", None),
        "q2": _row("sc-nat-q2", "C", "abstain", None),
        "q3": _row("sc-nat-q3", "A", "answer", False),
    }
    s = summarise(synth, human)
    assert (s["n_c"], s["n_a"]) == (2, 1)
    assert s["hall_synth"] == 0.0
    assert s["hall_human"] == 50.0
    assert (s["hall_b"], s["hall_c"]) == (1, 0)
    assert s["acc_synth"] == 100.0
    assert s["acc_human"] == 0.0


def test_summarise_ignores_items_missing_from_one_side():
    """A question present in only one condition is not a pair and must be dropped."""
    synth = {"q1": _row("sc-tts-q1", "C", "answer", None),
             "q2": _row("sc-tts-q2", "C", "answer", None)}
    human = {"q1": _row("sc-nat-q1", "C", "abstain", None)}
    s = summarise(synth, human)
    assert s["n_c"] == 1
    assert s["hall_synth"] == 100.0
    assert s["hall_human"] == 0.0


def test_hedge_is_not_a_hallucination():
    """Hedges are reported separately and count as neither answer nor abstention."""
    synth = {"q1": _row("sc-tts-q1", "C", "hedge", None)}
    human = {"q1": _row("sc-nat-q1", "C", "answer", None)}
    s = summarise(synth, human)
    assert s["hall_synth"] == 0.0
    assert s["hall_human"] == 100.0
