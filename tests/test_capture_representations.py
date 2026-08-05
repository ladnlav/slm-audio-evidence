"""Dry-run tests for scripts/capture_representations.py — no torch, no GPU, no weights.

The pooling function is where a silent bug would be most expensive: a wrong mask means the probe
trains on vectors that mix in padding, or on "audio" that is actually the question text, and
nothing downstream would flag it -- the AUROC would just be quietly worse and we would blame
the method. So the arithmetic is checked against hand-computed values, not just for shape.

Run: python tests/test_capture_representations.py   (or: pytest tests/test_capture_representations.py)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from capture_representations import POOLINGS, pool_hidden_states  # noqa: E402
from fakes.fake_qwen2_audio import FakeQwen2AudioCapture  # noqa: E402


def _check(cases: list[tuple[bool, str]]) -> int:
    failures = 0
    for ok, description in cases:
        print(f"  [{'ok' if ok else 'FAIL'}] {description}")
        failures += 0 if ok else 1
    return failures


def test_pooling_arithmetic() -> int:
    """Hand-computed case: 4 positions (2 audio, 1 text, 1 padding), d=2."""
    print("test_pooling_arithmetic")
    hidden = np.array([
        [1.0, 1.0],    # audio
        [3.0, 5.0],    # audio
        [10.0, 20.0],  # text  <- last real position
        [99.0, 99.0],  # padding, must be ignored everywhere
    ])
    attn = np.array([1, 1, 1, 0])
    audio = np.array([1, 1, 0, 0])
    got = pool_hidden_states(hidden, attn, audio)
    return _check([
        (np.allclose(got["last"], [10.0, 20.0]),
         "last = final NON-padding position, not the padded tail"),
        (np.allclose(got["mean_all"], [(1 + 3 + 10) / 3, (1 + 5 + 20) / 3]),
         "mean_all averages the 3 real positions only"),
        (np.allclose(got["mean_audio"], [2.0, 3.0]),
         "mean_audio averages the 2 audio positions"),
        (np.allclose(got["mean_text"], [10.0, 20.0]),
         "mean_text averages the text positions (audio excluded)"),
        (set(got) == set(POOLINGS), "all four poolings returned"),
    ])


def test_pooling_edge_cases() -> int:
    """Empty groups yield zeros, not NaN; bad input is rejected loudly."""
    print("test_pooling_edge_cases")
    hidden = np.array([[1.0, 2.0], [3.0, 4.0]])
    cases = []

    no_audio = pool_hidden_states(hidden, np.array([1, 1]), np.array([0, 0]))
    cases.append((np.allclose(no_audio["mean_audio"], [0.0, 0.0]) and not np.isnan(no_audio["mean_audio"]).any(),
                  "no audio positions -> zeros, never NaN"))
    cases.append((np.allclose(no_audio["mean_text"], [2.0, 3.0]),
                  "with no audio, mean_text equals mean_all"))

    all_audio = pool_hidden_states(hidden, np.array([1, 1]), np.array([1, 1]))
    cases.append((np.allclose(all_audio["mean_text"], [0.0, 0.0]),
                  "no text positions -> zeros"))

    # audio flagged on a padded position must not count as audio
    padded = pool_hidden_states(np.array([[1.0, 1.0], [7.0, 7.0]]), np.array([1, 0]), np.array([1, 1]))
    cases.append((np.allclose(padded["mean_audio"], [1.0, 1.0]),
                  "audio mask is intersected with the attention mask"))

    for bad_args, description in [
        ((np.zeros((2, 2)), np.array([0, 0]), np.array([0, 0])), "all-padding input raises"),
        ((np.zeros(2), np.array([1, 1]), np.array([0, 0])), "1-D hidden raises"),
        ((np.zeros((2, 2)), np.array([1, 1, 1]), np.array([0, 0])), "mask/seq length mismatch raises"),
    ]:
        try:
            pool_hidden_states(*bad_args)
            cases.append((False, description))
        except ValueError:
            cases.append((True, description))
    return _check(cases)


def test_end_to_end_with_fake_model() -> int:
    """Full CLI run on a stub model: files, shapes, ids and metadata."""
    print("test_end_to_end_with_fake_model")
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        manifest = tmp / "m.jsonl"
        items = [{"id": f"it-{i}", "audio_path": "unused.wav", "question": "q?",
                  "category": "C", "gold_answer": "UNANSWERABLE"} for i in range(5)]
        manifest.write_text("\n".join(json.dumps(r) for r in items) + "\n", encoding="utf-8")
        out = tmp / "cap"

        proc = subprocess.run(
            [sys.executable, "scripts/capture_representations.py", "--manifest", str(manifest),
             "--out", str(out), "--layers", "8", "16", "--fake", "--full-for", "2"],
            cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8",
        )
        if proc.returncode != 0:
            print(proc.stdout[-1500:], proc.stderr[-1500:])
            return _check([(False, "CLI run exited cleanly")])

        # np.load is lazy and keeps the archive open -- on Windows that blocks TemporaryDirectory
        # cleanup, so read what we need and close it explicitly.
        with np.load(out / "representations.npz", allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        expected_keys = {f"layer_{l}_{p}" for l in (8, 16) for p in POOLINGS} | {"ids"}
        full_files = sorted((out / "full").glob("*.npz")) if (out / "full").exists() else []
        return _check([
            (set(data) == expected_keys, f"npz holds ids + 2 layers x 4 poolings (got {len(data)} keys)"),
            (data["layer_8_last"].shape == (5, 4096), f"pooled array is (n_items, d) = {data['layer_8_last'].shape}"),
            (list(data["ids"]) == [i["id"] for i in items], "ids preserved in manifest order"),
            (data["layer_8_last"].dtype == np.float16, "stored as float16 to keep the file small"),
            (meta["n_items"] == 5 and meta["layers"] == [8, 16], "meta records item count and layers"),
            (meta["hidden_dim"] == 4096, "meta records hidden dim"),
            (meta["fake"] is True, "meta flags this as a fake run, so it can never be mistaken for real data"),
            (len(full_files) == 2, f"--full-for 2 wrote 2 per-token files (got {len(full_files)})"),
        ])


def test_layer_out_of_range_is_rejected() -> int:
    """Asking for layer 99 must fail loudly, not silently capture something else."""
    print("test_layer_out_of_range_is_rejected")
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        manifest = tmp / "m.jsonl"
        manifest.write_text(json.dumps({"id": "x", "audio_path": "u.wav", "question": "q?"}) + "\n",
                            encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "scripts/capture_representations.py", "--manifest", str(manifest),
             "--out", str(tmp / "o"), "--layers", "99", "--fake"],
            cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8",
        )
        return _check([
            (proc.returncode != 0, "exits non-zero"),
            ("99" in (proc.stderr + proc.stdout), "error message names the offending layer"),
        ])


def test_fake_model_shapes() -> int:
    """The stub itself must mimic the real model's contract, or the tests above prove nothing."""
    print("test_fake_model_shapes")
    fake = FakeQwen2AudioCapture(n_audio=12, n_text=7, n_pad=3)
    hidden, attn, audio = fake.forward_prompt("a.wav", "q?", "{question}")
    return _check([
        (len(hidden) == 33, f"33 hidden-state entries = embeddings + 32 layers (got {len(hidden)})"),
        (hidden[0].shape == (22, 4096), f"(seq, d) per layer, seq = 12 audio + 7 text + 3 pad (got {hidden[0].shape})"),
        (attn.sum() == 19, "attention mask marks the 3 padded positions as 0"),
        (audio.sum() == 12, "audio mask covers exactly the audio positions"),
        ((audio & (1 - attn)).sum() == 0, "no audio position is marked as padding"),
    ])


def main() -> int:
    failures = (
        test_pooling_arithmetic()
        + test_pooling_edge_cases()
        + test_fake_model_shapes()
        + test_end_to_end_with_fake_model()
        + test_layer_out_of_range_is_rejected()
    )
    print("\nOK" if failures == 0 else f"\n{failures} FAILURE(S)")
    return failures


def test_arithmetic(): assert test_pooling_arithmetic() == 0
def test_edges(): assert test_pooling_edge_cases() == 0
def test_fake_shapes(): assert test_fake_model_shapes() == 0
def test_e2e(): assert test_end_to_end_with_fake_model() == 0
def test_bad_layer(): assert test_layer_out_of_range_is_rejected() == 0


if __name__ == "__main__":
    sys.exit(main())
