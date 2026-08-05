"""Capture Qwen2-Audio prompt representations for probe training (A2, PLAN.md section 3 step 1).

What this does: one forward pass per item over (audio + question), WITHOUT generating anything,
saving pooled hidden states from several layers. That is the "pre-generation" part of
pre-generation probing (paper 24) -- the probe must see only what the model knows before it
starts writing an answer.

Why pooled and not full: a 30-second clip becomes ~750 audio positions, plus text. Keeping every
position for every layer would be ~18 GB over the scale set; the four pooled vectors below are
~25 MB total and are what a linear probe consumes anyway. Full per-token states are only needed
for the attention probe, and only for the layer that wins the linear sweep -- use --full-for N
to keep them for the first N items.

Four poolings per layer, all computed over prompt positions only:
  last        -- the final non-padding position (what the model would condition the 1st token on)
  mean_all    -- mean over every prompt position
  mean_audio  -- mean over audio positions only (input_ids == audio_token_index)
  mean_text   -- mean over text positions only (the question + chat template)
The audio/text split is free to compute and lets us ask afterwards *where* the signal lives --
in the acoustic evidence or in the question phrasing. That distinction is the whole point of
this paper, so it is worth carrying.

Usage:
  py -3 scripts/capture_representations.py \
      --manifest data/manifests/scale_nmsqa_final.jsonl \
      --out results/probing/scale_nmsqa \
      --layers 8 16 24 32

  # smoke test without a GPU (fake model, no weights downloaded):
  py -3 scripts/capture_representations.py --manifest <m> --out /tmp/cap --fake
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `from src...`

DEFAULT_LAYERS = (8, 16, 24, 32)
POOLINGS = ("last", "mean_all", "mean_audio", "mean_text")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture pooled prompt representations for probing.")
    p.add_argument("--manifest", required=True, help="JSONL manifest (id, audio_path, question).")
    p.add_argument("--out", required=True, help="Output directory; writes representations.npz + meta.json.")
    p.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS),
                   help="Decoder layers to keep. Index 0 = embeddings, 32 = last layer of Qwen2-Audio. "
                        "Capture several at once: which one carries the signal is an empirical "
                        "question answered after the fact, not a design choice made up front.")
    p.add_argument("--prompt", default="plain.txt", help="Prompt file from src/prompts/.")
    p.add_argument("--limit", type=int, default=None, help="Only the first N items (debugging).")
    p.add_argument("--full-for", type=int, default=0, metavar="N",
                   help="Also keep FULL per-token states for the first N items (attention probe).")
    p.add_argument("--no-8bit", action="store_true", help="Load in fp16 instead of 8-bit.")
    p.add_argument("--fake", action="store_true",
                   help="Dry run with a stub model: no weights, no GPU. Shapes and files are real.")
    return p.parse_args()


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def pool_hidden_states(
    hidden: np.ndarray,
    attention_mask: np.ndarray,
    audio_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reduce one layer's per-token states to four vectors. Pure function -- unit-tested.

    Args:
        hidden:         (seq_len, d) states for one item, one layer.
        attention_mask: (seq_len,) 1 for real positions, 0 for padding.
        audio_mask:     (seq_len,) 1 for audio positions (a subset of the real ones).

    Padding is excluded everywhere: a mean over padded positions would silently shrink toward
    whatever the pad embedding is, and 'last' would land on padding rather than on the last
    real token. If a group is empty (e.g. no audio positions), its vector is all-zeros -- an
    honest "nothing here" rather than a NaN that would poison downstream training.
    """
    if hidden.ndim != 2:
        raise ValueError(f"hidden must be (seq_len, d), got {hidden.shape}")
    if attention_mask.shape[0] != hidden.shape[0] or audio_mask.shape[0] != hidden.shape[0]:
        raise ValueError("mask length must match hidden's seq_len")

    real = attention_mask.astype(bool)
    if not real.any():
        raise ValueError("attention_mask is all zeros -- nothing to pool")
    audio = audio_mask.astype(bool) & real
    text = real & ~audio

    def _mean(sel: np.ndarray) -> np.ndarray:
        if not sel.any():
            return np.zeros(hidden.shape[1], dtype=hidden.dtype)
        return hidden[sel].mean(axis=0)

    last_idx = int(np.nonzero(real)[0][-1])
    return {
        "last": hidden[last_idx],
        "mean_all": _mean(real),
        "mean_audio": _mean(audio),
        "mean_text": _mean(text),
    }


def build_model(fake: bool, load_in_8bit: bool):
    """Real Qwen2-Audio, or a stub with the same call surface for --fake runs."""
    if fake:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
        from fakes.fake_qwen2_audio import FakeQwen2AudioCapture

        return FakeQwen2AudioCapture()

    from src.models.qwen2_audio import Qwen2AudioModel

    base = Qwen2AudioModel(load_in_8bit=load_in_8bit)
    return _CaptureAdapter(base)


class _CaptureAdapter:
    """Wraps Qwen2AudioModel to expose a forward pass instead of generation.

    Deliberately reuses the wrapper's own processor and chat-template handling: if the capture
    path built its prompt differently from src/models/qwen2_audio.py's answer(), the probe would
    be trained on representations the model never actually sees at inference time.
    """

    def __init__(self, base) -> None:
        self.base = base
        self.audio_token_index = int(
            getattr(base.model.config, "audio_token_index", 151646)
        )

    def forward_prompt(self, audio_path: str, question: str, prompt_template: str):
        import librosa
        import torch

        from src.models.base import render_prompt

        prompt = render_prompt(prompt_template, question)
        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": prompt},
        ]}]
        text = self.base.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audio, _ = librosa.load(audio_path, sr=self.base.sample_rate, mono=True)
        inputs = self.base._pack_inputs(text, audio)
        inputs = {k: (v.to(self.base.model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        with torch.no_grad():
            out = self.base.model(**inputs, output_hidden_states=True, return_dict=True)
        hidden = [h[0].float().cpu().numpy() for h in out.hidden_states]  # per layer, (seq, d)
        input_ids = inputs["input_ids"][0].cpu().numpy()
        attn = inputs.get("attention_mask")
        attn = attn[0].cpu().numpy() if attn is not None else np.ones_like(input_ids)
        audio_mask = (input_ids == self.audio_token_index).astype(np.int64)
        return hidden, attn, audio_mask


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = read_jsonl(args.manifest)
    if args.limit:
        items = items[: args.limit]
    prompt_template = (Path("src/prompts") / args.prompt).read_text(encoding="utf-8").strip()

    model = build_model(args.fake, load_in_8bit=not args.no_8bit)

    ids: list[str] = []
    # pooled[layer][pooling] -> list of (d,) vectors, stacked at the end
    pooled: dict[int, dict[str, list[np.ndarray]]] = {
        l: {p: [] for p in POOLINGS} for l in args.layers
    }
    full_dir = out_dir / "full"
    n_audio_positions: list[int] = []
    t_start = time.perf_counter()

    for n, item in enumerate(items, 1):
        hidden, attn, audio_mask = model.forward_prompt(
            item["audio_path"], item["question"], prompt_template
        )
        n_layers_available = len(hidden) - 1  # hidden[0] is the embedding output
        for layer in args.layers:
            if layer > n_layers_available:
                raise ValueError(
                    f"--layers asks for {layer}, but the model exposes {n_layers_available} "
                    f"(hidden_states has {len(hidden)} entries incl. embeddings)"
                )
            vecs = pool_hidden_states(hidden[layer], attn, audio_mask)
            for p in POOLINGS:
                pooled[layer][p].append(vecs[p].astype(np.float16))

        if n <= args.full_for:
            full_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                full_dir / f"{item['id']}.npz",
                **{f"layer_{l}": hidden[l].astype(np.float16) for l in args.layers},
                attention_mask=attn.astype(np.int8),
                audio_mask=audio_mask.astype(np.int8),
            )

        ids.append(item["id"])
        n_audio_positions.append(int(audio_mask.sum()))
        if n % 25 == 0 or n == len(items):
            rate = n / (time.perf_counter() - t_start)
            print(f"[{n}/{len(items)}] {rate:.1f} items/s")

    arrays = {
        f"layer_{layer}_{p}": np.stack(pooled[layer][p])
        for layer in args.layers for p in POOLINGS
    }
    np.savez_compressed(out_dir / "representations.npz", ids=np.array(ids), **arrays)

    meta = {
        "manifest": str(args.manifest),
        "n_items": len(ids),
        "layers": list(args.layers),
        "poolings": list(POOLINGS),
        "prompt": args.prompt,
        "hidden_dim": int(next(iter(arrays.values())).shape[1]),
        "audio_positions": {
            "min": min(n_audio_positions), "max": max(n_audio_positions),
            "mean": round(float(np.mean(n_audio_positions)), 1),
        },
        "full_states_for": min(args.full_for, len(ids)),
        "fake": bool(args.fake),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    size_mb = (out_dir / "representations.npz").stat().st_size / 1e6
    print(f"\n[+] {len(ids)} items x {len(args.layers)} layers x {len(POOLINGS)} poolings "
          f"-> {out_dir/'representations.npz'} ({size_mb:.1f} MB)")
    print(f"[+] meta: {out_dir/'meta.json'}")
    if args.fake:
        print("[!] --fake: representations are random, useful only for pipeline checks.")


if __name__ == "__main__":
    main()
