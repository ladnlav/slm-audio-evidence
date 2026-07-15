from __future__ import annotations

from .base import DEFAULT_PROMPT_NAME, JudgeResult, LLMJudge, Verdict, parse_verdict, render_judge_prompt

__all__ = [
    "DEFAULT_PROMPT_NAME",
    "JudgeResult",
    "LLMJudge",
    "Verdict",
    "parse_verdict",
    "render_judge_prompt",
    "build_judge",
]


def build_judge(backend: str, **kwargs) -> LLMJudge:
    """Factory, so callers (run_eval.py, notebooks) depend on a backend *name*, not a class.
    Imports stay inside each branch so `--judge none` never needs torch or google-generativeai.
    """
    if backend == "local":
        from .local_hf import LocalHFJudge

        return LocalHFJudge(**kwargs)
    if backend == "tiered":
        # Fast (no-think) pass on everything, slow (thinking) pass only on responses that
        # look verbosity-bias-prone -- see tiered.py. For 1000+-item datasets, not the ~90-item
        # pilot (there the plain 'local' thinking judge is affordable and simpler).
        from .tiered import build_local_tiered_judge

        return build_local_tiered_judge(**kwargs)
    if backend == "gemini":
        from .gemini import GeminiJudge

        return GeminiJudge(**kwargs)
    if backend == "fake":
        from .fake import FakeJudge

        return FakeJudge(**kwargs)
    raise ValueError(f"Unknown judge backend: {backend!r} (expected 'local', 'tiered', 'gemini', or 'fake')")
