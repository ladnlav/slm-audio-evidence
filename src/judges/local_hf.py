from __future__ import annotations

import re
from typing import Any

import torch

from .base import DEFAULT_PROMPT_NAME, LLMJudge

_THINK_CLOSE = "</think>"
_VERDICT_RE = re.compile(r"\b(CORRECT|INCORRECT|ABSTAINED)\b", re.IGNORECASE)


class _StopOnVerdict:
    """Stops generation as soon as a verdict word appears after </think>, instead of always
    running to max_new_tokens. Only looks past </think> -- the model may mention a verdict word
    while still reasoning (e.g. weighing "is this CORRECT or not"), and stopping on that would
    cut off its actual conclusion. Before </think> closes, keeps generating regardless.
    """

    def __init__(self, tokenizer, prompt_len: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0, self.prompt_len :], skip_special_tokens=True)
        _, _, after_think = text.partition(_THINK_CLOSE)
        return bool(after_think) and bool(_VERDICT_RE.search(after_think))

# Qwen3-8B (2025 generation) — swapped in 2026-07-15 to isolate the model variable from the
# judge_v2.txt prompt fix (both address the same verbosity-bias finding; testing them one at a
# time tells us which one actually helped). Previously Qwen2.5-7B-Instruct, reused from
# src/models/cascade.py's config for expedience, never evaluated as a judge on its own merits.
# See docs/decisions.md for the full audit and the reasoning behind this choice.
MODEL_ID = "Qwen/Qwen3-8B"


class LocalHFJudge(LLMJudge):
    """Runs the rubric through a local instruct LLM — no API key, no per-call cost or network.

    Primary judge backend per project decision (open-source first, docs/decisions.md).
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        load_in_8bit: bool = True,
        max_new_tokens: int | None = None,
        prompt_name: str = DEFAULT_PROMPT_NAME,
        enable_thinking: bool = True,
    ) -> None:
        super().__init__(prompt_name=prompt_name)
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # -v2 (thinking) vs -v1 (no-think): distinct judge_name so judge_cache.jsonl entries
        # never collide across the behavior change (cache key includes judge_name, see
        # run_eval.py). docs/decisions.md 2026-07-15, judge_v2 verbosity-bias post-mortem: a
        # one-sentence prompt addition tried to fix over-crediting of long answers but couldn't
        # -- at max_new_tokens=64 with thinking off, the judge had no room to check whether the
        # gold facts were actually present, so it started pattern-matching on length instead
        # (92%->89%). Reverted the prompt to judge_v1.txt; giving the judge room to reason
        # (enable_thinking=True) is the alternative lever. Both modes stay available -- see
        # src/judges/tiered.py, which runs -v1 on everything and escalates only the
        # verbosity-bias-prone items to -v2, since -v2 is far slower per call.
        self._model_id = model_id
        self.set_thinking(enable_thinking, max_new_tokens=max_new_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        kwargs: dict[str, Any] = {"device_map": "auto"}
        if load_in_8bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    def set_thinking(self, enable_thinking: bool, max_new_tokens: int | None = None) -> None:
        """Swap generation mode on an already-loaded instance -- keeps enable_thinking,
        max_new_tokens, and name (the judge_cache.jsonl cache-key component) consistent with
        each other, instead of setting enable_thinking by hand and leaving a stale name behind
        (the -v1/-v2 suffix would then no longer match what actually ran). Use this together
        with LLMJudge.set_prompt() to A/B a rubric+mode on one loaded model without a second
        weights load -- see docs/decisions.md 2026-07-16 (ad-hoc cell OOM'd loading a second copy).
        """
        self.enable_thinking = enable_thinking
        self.name = f"llm-{self._model_id.split('/')[-1].lower()}-{'v2' if enable_thinking else 'v1'}"
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else (512 if enable_thinking else 64)

    def _generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        # enable_thinking=True (self.enable_thinking): Qwen3 emits its native <think>...</think>
        # block before the verdict word instead of answering cold -- needs max_new_tokens raised
        # well past 64 or the reply gets truncated mid-thought with no verdict word at all, which
        # parse_verdict correctly reports as UNPARSEABLE rather than guessing. Harmless no-op
        # kwarg either way on chat templates that don't recognize it (e.g. Qwen2.5).
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.enable_thinking
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        from transformers import StoppingCriteriaList

        # Caps the common case (verdict reached well under max_new_tokens) without capping
        # thinking itself -- max_new_tokens stays the hard ceiling for cases that never emit
        # a clean verdict (surfaces as UNPARSEABLE via parse_verdict, not a silent guess).
        stopping = StoppingCriteriaList([_StopOnVerdict(self.tokenizer, inputs["input_ids"].shape[1])])
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # deterministic verdicts, same convention as plain/S1 inference
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping,
            )
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
