from __future__ import annotations

from typing import Any

import librosa
import torch

from .base import SpeechQAModel, render_prompt

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"


class Qwen2AudioModel(SpeechQAModel):
    name = "qwen2audio"

    def __init__(self, load_in_8bit: bool = True) -> None:
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        kwargs: dict[str, Any] = {"device_map": "auto"}
        if load_in_8bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = torch.float16
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
        self.sample_rate = int(self.processor.feature_extractor.sampling_rate)  # 16000

    # Keys under which the processor returns the encoded waveform. If none of these is
    # present, the batch is text-only however plausible the generated answers look.
    AUDIO_FEATURE_KEYS = ("input_features", "audio_values", "input_audio_features")

    def _pack_inputs(self, text: str, audio) -> dict[str, Any]:
        """Encode text + waveform, verifying the audio actually made it into the batch.

        transformers renamed this kwarg across versions (``audios=`` -> ``audio=``). The
        obvious probe -- pass the old name and catch TypeError -- is wrong on 5.x: the
        processor does not raise, it logs "Keyword argument `audios` is not a valid
        argument for this processor and will be ignored" and returns a text-only batch.
        The model then answers from the question alone and produces fluent, entirely
        ungrounded output that no downstream metric flags as broken. So we try each name
        and require audio features in the result, failing loudly when there are none.
        """
        last_error: Exception | None = None
        for kwarg in ("audio", "audios"):
            try:
                inputs = self.processor(
                    text=text, **{kwarg: [audio]}, return_tensors="pt", padding=True
                )
            except TypeError as exc:  # genuinely unsupported name on this version
                last_error = exc
                continue
            if any(key in inputs for key in self.AUDIO_FEATURE_KEYS):
                return inputs
        raise RuntimeError(
            "Qwen2-Audio processor returned no audio features under either 'audio' or "
            "'audios'; the batch would be text-only and every result silently ungrounded. "
            f"Processor: {type(self.processor).__name__}, transformers may have renamed "
            f"the argument again. Last TypeError: {last_error}"
        )

    def answer(self, audio_path, question, prompt_template, gen_kwargs=None):
        gen_kwargs = {"max_new_tokens": 256, "do_sample": False, **(gen_kwargs or {})}
        prompt = render_prompt(prompt_template, question)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        inputs = self._pack_inputs(text, audio)
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
