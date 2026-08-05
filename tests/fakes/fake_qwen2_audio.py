"""Stub standing in for Qwen2-Audio in capture dry-runs: same call surface, no weights, no GPU.

Shapes mimic the real thing (33 hidden-state entries = embeddings + 32 decoder layers, d=4096
by default) so that a --fake run exercises exactly the code paths a real run does: layer
indexing, the audio/text split, padding handling, stacking and serialization. The numbers
themselves are deterministic noise -- the point is the plumbing, not the values.
"""

from __future__ import annotations

import numpy as np

N_LAYERS = 32
HIDDEN_DIM = 4096
AUDIO_TOKEN_INDEX = 151646


class FakeQwen2AudioCapture:
    def __init__(self, n_layers: int = N_LAYERS, hidden_dim: int = HIDDEN_DIM,
                 n_audio: int = 12, n_text: int = 7, n_pad: int = 0, seed: int = 0) -> None:
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_audio = n_audio
        self.n_text = n_text
        self.n_pad = n_pad
        self.audio_token_index = AUDIO_TOKEN_INDEX
        self.calls = 0
        self._rng = np.random.default_rng(seed)

    def forward_prompt(self, audio_path: str, question: str, prompt_template: str):
        self.calls += 1
        seq = self.n_audio + self.n_text + self.n_pad
        hidden = [
            self._rng.normal(size=(seq, self.hidden_dim)).astype(np.float32)
            for _ in range(self.n_layers + 1)  # +1: embeddings at index 0
        ]
        attn = np.ones(seq, dtype=np.int64)
        if self.n_pad:
            attn[-self.n_pad:] = 0
        audio_mask = np.zeros(seq, dtype=np.int64)
        audio_mask[: self.n_audio] = 1  # audio first, then the question text
        return hidden, attn, audio_mask
