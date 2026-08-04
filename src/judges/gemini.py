from __future__ import annotations

import os
import re
import time

from .base import DEFAULT_PROMPT_NAME, LLMJudge

# Free tier, verified 2026-07-22 (https://ai.google.dev/gemini-api/docs/rate-limits,
# https://www.aifreeapi.com/en/posts/gemini-api-free-tier-rate-limits): gemini-2.5-flash-lite is
# the only small Gemini model whose free-tier daily cap (1,000 RPD) actually covers a ~1000-item
# judge run in one day -- gemini-2.5-flash is only 250 RPD (hit in practice: ResourceExhausted
# after a handful of items), gemini-2.0-flash is worse (5 RPM) and deprecated besides. RPM is also
# more forgiving here (15 vs 10).
MODEL_ID = "gemini-2.5-flash-lite"
_FREE_TIER_RPM = 15  # pace calls this far apart so we avoid the 429 instead of just retrying it


class GeminiJudge(LLMJudge):
    """API fallback for when no local GPU is available (project decision: open-source first,
    docs/decisions.md). Needs GEMINI_API_KEY in the environment — never hardcode the key.

    Pinned to gemini-2.5-flash-lite, not a "smarter" sibling used as a fallback on quota
    exhaustion: this project judges ~1000 responses per run (pilot + scale set), and every other
    free-tier Gemini model caps out well under that. A per-item fallback to a different model was
    considered and rejected -- verdicts from two different judge models inside one results/
    directory would be an uncomparable mix, worse than a single consistent (if slightly weaker)
    judge throughout. If a run genuinely needs more than flash-lite's 1,000 RPD, split it across
    days or move to a paid tier -- do not silently swap models mid-run.
    """

    def __init__(self, model_id: str = MODEL_ID, max_retries: int = 3, prompt_name: str = DEFAULT_PROMPT_NAME) -> None:
        super().__init__(prompt_name=prompt_name)
        import google.generativeai as genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set — export it before using --judge gemini.")
        genai.configure(api_key=api_key)
        self.name = f"llm-{model_id}-v1"
        self.max_retries = max_retries
        self._model = genai.GenerativeModel(model_id)
        self._min_interval = 60.0 / _FREE_TIER_RPM
        self._last_call = 0.0

    def _throttle(self) -> None:
        wait = self._min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    def _generate(self, prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = self._model.generate_content(prompt)
                self._last_call = time.time()
                return (resp.text or "").strip()
            except Exception as exc:  # network hiccup / rate limit / safety block — retry with backoff
                self._last_call = time.time()
                last_error = exc
                print(f"  [gemini attempt {attempt + 1}/{self.max_retries}] {exc!r}")
                # ResourceExhausted spells out the real wait ("Please retry in 55.4s") -- honor
                # that instead of the generic 2**attempt backoff (1s/2s/4s), which is far too
                # short to clear a per-minute/per-day quota window.
                retry_match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc))
                time.sleep(float(retry_match.group(1)) + 1 if retry_match else 2**attempt)
        # repr(last_error), not just str() -- run_eval.py's except-block only prints str(exc) on
        # the RuntimeError raised here, and `raise ... from last_error` alone is invisible there
        # (it only shows up in a full traceback, which nothing prints). Folding the real cause
        # into this message is what actually reaches the console log and judge_cache.jsonl.
        raise RuntimeError(
            f"Gemini judge failed after {self.max_retries} attempts: {last_error!r}"
        ) from last_error
