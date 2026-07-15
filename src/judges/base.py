"""LLM-judge contract: rubric in, one verdict out.

Mirrors src/models/base.py's SpeechQAModel — swap the backend (_generate) and
everything upstream (run_eval.py, scripts/audit_judge.py) keeps working. Prompt
rendering and verdict parsing live here, not in the backends, so a backend swap
(local model <-> Gemini) or a rubric rewrite can never change how a verdict is
interpreted without also changing the shared, tested code path.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "judge_v1.txt"
PROMPT_VERSION = "judge_v1.txt"  # bump the file AND this string together (cache key, provenance)

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


def render_judge_prompt(transcript: str, question: str, gold: str, response: str) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
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
    """
    matches = {m.group(1) for m in _VERDICT_RE.finditer(raw_output.upper())}
    if len(matches) == 1:
        return Verdict(matches.pop())
    return Verdict.UNPARSEABLE


class LLMJudge(ABC):
    """One method to implement per backend: send the rendered rubric, return raw text."""

    name: str = "base"

    @abstractmethod
    def _generate(self, prompt: str) -> str:
        """Send `prompt` (the rendered rubric) to the backend; return its raw text reply."""

    def judge(self, transcript: str, question: str, gold: str, response: str) -> JudgeResult:
        prompt = render_judge_prompt(transcript, question, gold, response)
        raw = self._generate(prompt)
        return JudgeResult(verdict=parse_verdict(raw), raw_output=raw, judge_name=self.name)
