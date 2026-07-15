from __future__ import annotations

from .base import LLMJudge


class FakeJudge(LLMJudge):
    """Deterministic canned backend — no GPU, no API key, no network.

    Not a real judge: exists only so tests/CI and run_eval.py's wiring (caching,
    provenance tagging, error handling) can be exercised without Colab or a
    Gemini key. Real evaluation must use 'local' or 'gemini'.
    """

    def __init__(self, canned: str = "CORRECT") -> None:
        self.name = "llm-fake-v1"
        self.calls = 0
        self._canned = canned

    def _generate(self, prompt: str) -> str:
        self.calls += 1
        return self._canned
