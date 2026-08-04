from __future__ import annotations

import re
from typing import Any

import torch

from .base import DEFAULT_PROMPT_NAME, LLMJudge, _THINK_CLOSE_MARKERS

_VERDICT_RE = re.compile(r"\b(CORRECT|INCORRECT|ABSTAINED)\b", re.IGNORECASE)


class _StopOnVerdict:
    """Stops generation as soon as a verdict word appears after the reasoning block closes,
    instead of always running to max_new_tokens. Only looks past the closing marker -- the
    model may mention a verdict word while still reasoning (e.g. weighing "is this CORRECT or
    not"), and stopping on that would cut off its actual conclusion. Before the block closes,
    keeps generating regardless.

    2026-07-22: was hardcoded to "</think>" (Qwen3's convention) -- Ministral-3-8B-Reasoning
    uses "[THINK]...[/THINK]" instead (confirmed via its real chat_template.jinja on the Hub),
    so this never fired, generation always ran to the full max_new_tokens ceiling (~118s/item,
    near-identical every time -- the tell), and parse_verdict (src/judges/base.py) scanned the
    entire reasoning instead of just the final answer, causing near-universal UNPARSEABLE. Now
    checks every known closing marker, not just Qwen's.
    """

    def __init__(self, tokenizer, prompt_len: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0, self.prompt_len :], skip_special_tokens=True)
        after_think = ""
        for marker in _THINK_CLOSE_MARKERS:
            if marker in text:
                after_think = text.rsplit(marker, 1)[-1]
                break
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
        load_in_4bit: bool = False,
        max_new_tokens: int | None = None,
        prompt_name: str = DEFAULT_PROMPT_NAME,
        enable_thinking: bool = True,
        display_name: str | None = None,
    ) -> None:
        super().__init__(prompt_name=prompt_name)
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # display_name: only needed when model_id is a local path (e.g. a Kaggle flat-cache dir
        # standing in for a Hub id, see notebooks/kaggle_run.ipynb's FLAT_CACHE_DIRS) rather than
        # a real Hub id -- without it, judge.name (below) would derive from the local dir's own
        # folder name ("qwen3-8b-flat-cache") instead of the actual model identity ("qwen3-8b"),
        # silently giving the exact same model a different judge_cache.jsonl cache key / output
        # dir depending on whether it was loaded fresh from the Hub or from a saved flat cache.
        # 2026-07-23 (docs/decisions.md): caught in practice -- a flat-cache Qwen3-8B run got
        # filed under judge_name "llm-qwen3-8b-flat-cache-v1", not "llm-qwen3-8b-v1".
        self._display_name = display_name or model_id

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
        # 2026-07-23 (docs/decisions.md): Qwen3.6-27B (~54 GB bf16, ~27 GB at 8-bit) failed with
        # "Some modules are dispatched on the CPU or the disk" on 2xT4 (32 GB combined) -- 8-bit
        # weights alone leave almost no headroom for KV-cache/activations, so accelerate's
        # device_map="auto" tries to offload the overflow to CPU/disk, which bitsandbytes'
        # 8-bit path refuses without an explicit fp32 CPU-offload opt-in. 4-bit halves the
        # quantized footprint (~13-14 GB for this model) instead of fighting the offload path --
        # takes precedence over load_in_8bit if both are somehow passed.
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = torch.float16

        # mistral3 (Ministral-3's published checkpoints): transformers dropped
        # Mistral3ForConditionalGeneration._checkpoint_conversion_mapping between 4.57.6 and the
        # main/5.x line (confirmed by diffing modeling_mistral3.py across both) -- the checkpoint's
        # real safetensors keys ("language_model.model.layers...", "language_model.lm_head...")
        # no longer get auto-renamed to what the class expects ("model.language_model.layers...",
        # top-level "lm_head..."). from_pretrained does NOT raise on this -- it silently loads
        # with those layers randomly initialized (only visible via its own printed LOAD REPORT,
        # every key MISSING) and would otherwise still "succeed" as far as this class is
        # concerned. `key_mapping` is still an accepted from_pretrained kwarg (dict[str, str],
        # regex -> replacement) on every transformers version -- passing the exact mapping the
        # class used to apply automatically fixes this regardless of version. Harmless to pass
        # for any other model_type; only mistral3 checkpoints will ever match these patterns.
        # docs/decisions.md 2026-07-22 (verified against the real checkpoint index.json + a diff
        # of modeling_mistral3.py, not guessed).
        from transformers import AutoConfig

        if getattr(AutoConfig.from_pretrained(model_id), "model_type", None) == "mistral3":
            kwargs["key_mapping"] = {
                "^language_model.model": "model.language_model",
                "^vision_tower": "model.vision_tower",
                "^multi_modal_projector": "model.multi_modal_projector",
                "^language_model.lm_head": "lm_head",
            }

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except ValueError as exc:
            if "Unrecognized configuration class" not in str(exc):
                raise  # a real error (e.g. VRAM/device_map) -- retrying via a different Auto
                # class would only fail again with the same message, masking the actual cause.
            # Some 2026-generation checkpoints (e.g. Ministral-3-8B's mistral3 architecture,
            # confirmed via HF's own auto-mapping: registered under image-text-to-text, not
            # causal-LM, even for the text-only instruct/reasoning variants) raise this specific
            # error -- not an OOM or a missing-package error, just the wrong Auto* class. Retry
            # once before giving up; docs/decisions.md 2026-07-22.
            from transformers import AutoModelForImageTextToText

            self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)

    def set_thinking(self, enable_thinking: bool, max_new_tokens: int | None = None) -> None:
        """Swap generation mode on an already-loaded instance -- keeps enable_thinking,
        max_new_tokens, and name (the judge_cache.jsonl cache-key component) consistent with
        each other, instead of setting enable_thinking by hand and leaving a stale name behind
        (the -v1/-v2 suffix would then no longer match what actually ran). Use this together
        with LLMJudge.set_prompt() to A/B a rubric+mode on one loaded model without a second
        weights load -- see docs/decisions.md 2026-07-16 (ad-hoc cell OOM'd loading a second copy).
        """
        self.enable_thinking = enable_thinking
        self.name = f"llm-{self._display_name.split('/')[-1].lower()}-{'v2' if enable_thinking else 'v1'}"
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
