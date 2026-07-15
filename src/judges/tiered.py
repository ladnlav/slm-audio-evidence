from __future__ import annotations

from .base import DEFAULT_PROMPT_NAME, JudgeResult, LLMJudge

# response/gold word-count ratio above which the fast judge's verdict gets a second opinion
# from the slow one. Targets the verbosity-bias pattern found 2026-07-15 (docs/decisions.md):
# every real disagreement so far involved a response several times longer than gold.
DEFAULT_LENGTH_RATIO_THRESHOLD = 3.0



class TieredJudge(LLMJudge):
    """Runs a fast (no-think) pass on every item; only escalates to a slow (thinking) pass
    when the response looks verbosity-bias-prone (much longer than gold). At pilot scale
    (~90 category-B items) always running the thinking judge cost 20-40+ minutes; on a
    1000+-item, multi-run dataset that stops being affordable. Escalating only the risky
    ~10-20% keeps most of the accuracy gain from thinking at a fraction of the compute --
    see docs/decisions.md 2026-07-15 (throughput discussion).

    Wraps a single LocalHFJudge-like instance and toggles its `enable_thinking` /
    `max_new_tokens` between calls instead of loading the model twice -- an int8 8B model
    loaded twice would roughly double VRAM and load time, which a 16 GB Kaggle GPU can't
    spare. This only works because everything here runs strictly sequentially (no threads);
    do not share one `judge` instance across concurrent TieredJudge calls.
    """

    def __init__(
        self,
        judge: LLMJudge,
        length_ratio_threshold: float = DEFAULT_LENGTH_RATIO_THRESHOLD,
    ) -> None:
        self.judge_backend = judge
        self.length_ratio_threshold = length_ratio_threshold
        self.prompt_name = judge.prompt_name
        self.prompt_version = judge.prompt_version
        self.name = f"tiered-{judge.name.rsplit('-', 1)[0]}"  # e.g. tiered-llm-qwen3-8b
        self.escalated_count = 0
        self.total_count = 0

    def _generate(self, prompt: str) -> str:
        raise NotImplementedError("TieredJudge overrides judge() directly -- _generate() is unused.")

    def _looks_risky(self, gold: str, response: str) -> bool:
        gold_words = max(len(gold.split()), 1)
        response_words = len(response.split())
        return (response_words / gold_words) >= self.length_ratio_threshold

    def judge(self, transcript: str, question: str, gold: str, response: str) -> JudgeResult:
        self.total_count += 1
        self.judge_backend.set_thinking(False)
        result = self.judge_backend.judge(transcript, question, gold, response)

        if self._looks_risky(gold, response):
            self.escalated_count += 1
            self.judge_backend.set_thinking(True)
            result = self.judge_backend.judge(transcript, question, gold, response)

        return JudgeResult(verdict=result.verdict, raw_output=result.raw_output, judge_name=self.name)


def build_local_tiered_judge(
    model_id: str | None = None,
    prompt_name: str = DEFAULT_PROMPT_NAME,
    length_ratio_threshold: float = DEFAULT_LENGTH_RATIO_THRESHOLD,
    **local_kwargs,
) -> TieredJudge:
    from .local_hf import LocalHFJudge

    kwargs = dict(local_kwargs)
    if model_id:
        kwargs["model_id"] = model_id
    backend = LocalHFJudge(prompt_name=prompt_name, **kwargs)  # mode gets overridden per call
    return TieredJudge(backend, length_ratio_threshold=length_ratio_threshold)
