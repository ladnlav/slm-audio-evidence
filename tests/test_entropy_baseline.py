"""Dry-run tests for scripts/entropy_baseline.py — no torch, no API key, no network,
no LLM judge calls (clustering is pure rapidfuzz string similarity).
Run: python tests/test_entropy_baseline.py
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from entropy_baseline import auroc, cluster_answers, normalized_entropy, shannon_entropy  # noqa: E402


def _check(description: str, ok: bool) -> int:
    print(f"  [{'OK' if ok else 'FAIL'}] {description}")
    return 0 if ok else 1


def test_cluster_answers() -> int:
    failures = 0
    failures += _check(
        "all-identical samples -> 1 cluster",
        len(set(cluster_answers(["blue jacket", "blue jacket", "blue jacket"]))) == 1,
    )
    failures += _check(
        "all-different samples -> k distinct clusters",
        len(set(cluster_answers(["blue jacket", "no information", "possibly a hat"]))) == 3,
    )
    failures += _check(
        "near-duplicate phrasing joins the same cluster",
        len(set(cluster_answers(["blue jacket", "a blue jacket", "he wore a blue jacket"]))) == 1,
    )
    assignment = cluster_answers(["blue jacket", "blue jacket", "no information"])
    failures += _check(
        "the two identical samples share a cluster id, the odd one out doesn't",
        assignment[0] == assignment[1] and assignment[2] != assignment[0],
    )
    return failures


def test_entropy() -> int:
    failures = 0
    failures += _check("all-same-cluster entropy is 0", shannon_entropy([0, 0, 0, 0]) == 0.0)
    failures += _check(
        "4 distinct clusters -> entropy = log2(4) = 2.0",
        abs(shannon_entropy([0, 1, 2, 3]) - 2.0) < 1e-9,
    )
    failures += _check(
        "normalized_entropy of full spread (k distinct clusters) = 1.0",
        abs(normalized_entropy(math.log2(5), 5) - 1.0) < 1e-9,
    )
    failures += _check("normalized_entropy with k=1 is 0 (no disagreement possible)", normalized_entropy(0.0, 1) == 0.0)
    return failures


def test_auroc() -> int:
    failures = 0
    failures += _check(
        "perfect separation (higher score -> positive) gives AUROC 1.0",
        abs(auroc([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]) - 1.0) < 1e-9,
    )
    failures += _check(
        "perfectly inverted separation gives AUROC 0.0",
        abs(auroc([0.4, 0.3, 0.2, 0.1], [0, 0, 1, 1]) - 0.0) < 1e-9,
    )
    failures += _check(
        "all-tied scores give AUROC 0.5 (no discrimination)",
        abs(auroc([0.1, 0.1, 0.1, 0.1], [0, 1, 0, 1]) - 0.5) < 1e-9,
    )
    failures += _check(
        "single-class labels -> AUROC undefined (None)",
        auroc([0.1, 0.2, 0.3], [0, 0, 0]) is None,
    )
    return failures


def test_cli_end_to_end() -> int:
    failures = 0
    tmp = Path(tempfile.mkdtemp())
    try:
        responses = [
            {"id": "sure1", "samples": ["blue jacket"] * 5},          # unanimous -> entropy 0
            {"id": "unsure1", "samples": ["blue", "red", "green", "black", "white"]},  # 5 distinct -> max entropy
        ]
        responses_path = tmp / "responses.jsonl"
        with responses_path.open("w", encoding="utf-8") as f:
            for row in responses:
                f.write(json.dumps(row) + "\n")

        entropy_path = tmp / "entropy_baseline.jsonl"
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "entropy_baseline.py"), "compute",
             "--responses", str(responses_path), "--out", str(entropy_path)],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        ok = result.returncode == 0
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] compute subcommand exits 0 (stderr: {result.stderr[-500:] if not ok else ''})")
        if not ok:
            return failures

        rows = {json.loads(l)["id"]: json.loads(l) for l in entropy_path.read_text(encoding="utf-8").splitlines()}
        failures += _check("unanimous item has entropy 0", rows["sure1"]["entropy"] == 0.0)
        failures += _check("fully-split item has entropy_normalized 1.0", abs(rows["unsure1"]["entropy_normalized"] - 1.0) < 1e-9)

        labels_path = tmp / "soft_labels.jsonl"
        with labels_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "sure1", "soft_label": 0.0}) + "\n")
            f.write(json.dumps({"id": "unsure1", "soft_label": 0.8}) + "\n")

        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "entropy_baseline.py"), "auroc",
             "--entropy", str(entropy_path), "--labels", str(labels_path)],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        ok = result.returncode == 0
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] auroc subcommand exits 0 (stderr: {result.stderr[-500:] if not ok else ''})")
        failures += _check("auroc output reports AUROC 1.000 (high entropy correctly ranked as the hallucinating item)", "AUROC" in result.stdout and "1.000" in result.stdout)
        failures += _check("auroc output carries the non-fold-aware caveat", "NOT the" in result.stdout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return failures


def main() -> None:
    suites = [
        ("cluster_answers", test_cluster_answers),
        ("shannon_entropy / normalized_entropy", test_entropy),
        ("auroc", test_auroc),
        ("CLI end-to-end", test_cli_end_to_end),
    ]
    total_failures = 0
    for name, suite in suites:
        print(f"\n=== {name} ===")
        total_failures += suite()

    print("\n================ РЕЗУЛЬТАТ ================")
    if total_failures == 0:
        print("[+] Все проверки прошли.")
    else:
        print(f"[-] {total_failures} проверок провалено.")
        sys.exit(1)


if __name__ == "__main__":
    main()
