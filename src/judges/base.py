"""LLM-judge contract: rubric in, one verdict out.

Mirrors src/models/base.py's SpeechQAModel — swap the backend (_generate) and
everything upstream (run_eval.py, scripts/audit_judge.py) keeps working. Prompt
rendering and verdict parsing live here, not in the backends, so a backend swap
(local model <-> Gemini) or a rubric rewrite can never change how a verdict is
interpreted without also changing the shared, tested code path.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
DEFAULT_PROMPT_NAME = "judge_v1.txt"

_VERDICT_RE = re.compile(r"\b(CORRECT|INCORRECT|ABSTAINED)\b")


class Verdict(str, Enum):
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    ABSTAINED = "ABSTAINED"
    UNPARSEABLE = "UNPARSEABLE"  # backend reply had 0 or >1 distinct verdict words

    def to_correctness(self) -> bool | None:
        """Same True/False/None contract as judge.check_correctness's `correct` field."""
        return {"CORRECT": True, "INCORRECT": False, "ABSTAINED": False}.get(self.value)


@dataclass(frozen=True)
class JudgeResult:
    verdict: Verdict
    raw_output: str
    judge_name: str


def _prompt_version(prompt_name: str) -> str:
    """filename + content hash, so an in-place edit to the rubric can never silently
    keep matching stale judge_cache.jsonl entries -- only renaming the file could
    hide that before (see docs/decisions.md 2026-07-15, judge_v2 rollout).
    """
    text = (PROMPTS_DIR / prompt_name).read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"{prompt_name}@{digest}"


def render_judge_prompt(
    transcript: str, question: str, gold: str, response: str, prompt_name: str = DEFAULT_PROMPT_NAME
) -> str:
    template = (PROMPTS_DIR / prompt_name).read_text(encoding="utf-8")
    return (
        template.replace("{transcript}", transcript)
        .replace("{question}", question)
        .replace("{gold}", gold)
        .replace("{response}", response)
    )


def parse_verdict(raw_output: str) -> Verdict:
    """Strict on purpose: an ambiguous reply must surface as UNPARSEABLE (-> pending-manual),
    never get silently guessed. Models rarely reply with exactly one word despite the rubric
    asking for it, so this scans the whole reply and only accepts a single distinct verdict word.

    If a <think>...</think> block is present (Qwen3-style reasoning), only the text after the
    last </think> is scanned. Found 2026-07-15 (docs/decisions.md, fourth audit): reasoning
    routinely name-checks multiple verdict words as hypotheses ("is this correct or incorrect?")
    before settling on one, so scanning the whole reply made 60/93 real thinking-judge replies
    look ambiguous even though every one of them gave a single clear answer after </think>.
    """
    text = raw_output.rsplit("</think>", 1)[-1] if "</think>" in raw_output else raw_output
    matches = {m.group(1) for m in _VERDICT_RE.finditer(text.upper())}
    if len(matches) == 1:
        return Verdict(matches.pop())
    return Verdict.UNPARSEABLE


class LLMJudge(ABC):
    """One method to implement per backend: send the rendered rubric, return raw text.

    Subclasses accept **kwargs and must forward prompt_name to super().__init__() so
    the rubric version is swappable per instance (A/B testing judge_v1 vs judge_v2)
    without touching backend code.
    """

    name: str = "base"

    def __init__(self, prompt_name: str = DEFAULT_PROMPT_NAME) -> None:
        self.prompt_name = prompt_name
        self.prompt_version = _prompt_version(prompt_name)

    @abstractmethod
    def _generate(self, prompt: str) -> str:
        """Send `prompt` (the rendered rubric) to the backend; return its raw text reply."""

    def judge(self, transcript: str, question: str, gold: str, response: str) -> JudgeResult:
        prompt = render_judge_prompt(transcript, question, gold, response, self.prompt_name)
        raw = self._generate(prompt)
        return JudgeResult(verdict=parse_verdict(raw), raw_output=raw, judge_name=self.name)
