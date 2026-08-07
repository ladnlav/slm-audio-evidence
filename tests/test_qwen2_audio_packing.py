"""Regression tests for Qwen2-Audio input packing.

These exist because of a silent failure on 2026-08-07: the processor in transformers
5.x does not raise on the old ``audios=`` kwarg, it warns and drops it. The old
try/except-TypeError probe therefore returned a text-only batch, the model answered
from the question alone, and 376 captured representations plus two inference runs were
ungrounded while looking entirely normal. The contract under test is: audio features
must be present in the returned batch, or packing must raise.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# src.models.qwen2_audio imports torch and librosa at module level, but the packing
# logic under test touches neither -- it only calls the processor. Stub whatever is
# missing so these tests run on a machine without the inference stack installed.
for _name in ("torch", "librosa"):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except ImportError:
            sys.modules[_name] = types.ModuleType(_name)

from src.models.qwen2_audio import Qwen2AudioModel  # noqa: E402


class _Processor:
    """Stand-in for AutoProcessor with a configurable set of accepted kwargs."""

    def __init__(self, accepts: tuple[str, ...], emits_audio: bool = True) -> None:
        self.accepts = accepts
        self.emits_audio = emits_audio
        self.calls: list[str] = []

    def __call__(self, text=None, return_tensors=None, padding=None, **kwargs):
        (name,) = kwargs  # exactly one of audio= / audios= per call site
        self.calls.append(name)
        if name not in self.accepts:
            raise TypeError(f"unexpected keyword argument {name!r}")
        batch = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
        if self.emits_audio:
            batch["input_features"] = [[0.0]]
        return batch


class _IgnoringProcessor(_Processor):
    """The real 5.x behaviour: unknown kwargs are ignored, not rejected."""

    def __call__(self, text=None, return_tensors=None, padding=None, **kwargs):
        (name,) = kwargs
        self.calls.append(name)
        batch = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
        if name in self.accepts and self.emits_audio:
            batch["input_features"] = [[0.0]]
        return batch


def _model(processor) -> Qwen2AudioModel:
    m = object.__new__(Qwen2AudioModel)  # bypass __init__: no weights are needed here
    m.processor = processor
    return m


def test_uses_new_kwarg_when_supported():
    proc = _Processor(accepts=("audio",))
    out = _model(proc)._pack_inputs("hi", [0.0])
    assert "input_features" in out
    assert proc.calls == ["audio"]


def test_falls_back_to_old_kwarg_on_typeerror():
    proc = _Processor(accepts=("audios",))
    out = _model(proc)._pack_inputs("hi", [0.0])
    assert "input_features" in out
    assert proc.calls == ["audio", "audios"]


def test_rejects_silently_ignored_kwarg():
    """The exact 2026-08-07 failure: no exception, no audio, plausible-looking batch."""
    proc = _IgnoringProcessor(accepts=("audio",), emits_audio=False)
    with pytest.raises(RuntimeError, match="no audio features"):
        _model(proc)._pack_inputs("hi", [0.0])
    # Both names were attempted before giving up -- neither yielded audio.
    assert proc.calls == ["audio", "audios"]


def test_ignoring_processor_still_works_when_one_name_lands():
    proc = _IgnoringProcessor(accepts=("audios",))
    out = _model(proc)._pack_inputs("hi", [0.0])
    assert "input_features" in out


def test_error_names_the_processor_class():
    proc = _IgnoringProcessor(accepts=(), emits_audio=False)
    with pytest.raises(RuntimeError, match="_IgnoringProcessor"):
        _model(proc)._pack_inputs("hi", [0.0])
