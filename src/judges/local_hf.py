from __future__ import annotations

from typing import Any

import torch

from .base import DEFAULT_PROMPT_NAME, LLMJudge

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
        max_new_tokens: int = 512,
        prompt_name: str = DEFAULT_PROMPT_NAME,
    ) -> None:
        super().__init__(prompt_name=prompt_name)
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # -v2: reasoning enabled (see _generate). Distinct from the old -v1 no-think backend so
        # judge_cache.jsonl entries never collide across the behavior change (cache key includes
        # judge_name, see run_eval.py). docs/decisions.md 2026-07-15, judge_v2 verbosity-bias
        # post-mortem: a one-sentence prompt addition tried to fix over-crediting of long answers
        # but couldn't -- at max_new_tokens=64 with thinking off, the judge had no room to check
        # whether the gold facts were actually present, so it started pattern-matching on length
        # instead (92%->89%). Reverted the prompt to judge_v1.txt; giving the judge room to reason
        # is the alternative lever, tried here instead of more prompt wording.
        self.name = f"llm-{model_id.split('/')[-1].lower()}-v2"
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        kwargs: dict[str, Any] = {"device_map": "auto"}
        if load_in_8bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    def _generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        # enable_thinking=True: let Qwen3 emit its native <think>...</think> block before the
        # verdict word instead of answering cold. Needs max_new_tokens raised well past the old
        # 64 (thinking runs long) or the reply gets truncated mid-thought with no verdict word at
        # all, which parse_verdict correctly reports as UNPARSEABLE rather than guessing. Harmless
        # no-op kwarg on chat templates that don't recognize it (e.g. Qwen2.5).
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # deterministic verdicts, same convention as plain/S1 inference
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
