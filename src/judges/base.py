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
# Models (Ministral-3-8B-Reasoning routinely) end a reply by stating the verdict word twice with
# no separator, e.g. "...INCORRECTINCORRECT" -- \b can't find a boundary between two immediately
# adjacent letters, so the SECOND (and often decisive, final) word was invisible to _VERDICT_RE
# entirely. This inserts a space between two glued verdict words before matching, so both are
# seen as separate occurrences instead of one undetectable blob. Confirmed 2026-07-22: affected
# 9 real replies, in at least one case flipping the extracted verdict to the wrong word (an
# earlier, non-final mention of a different verdict word won by default instead).
_GLUED_VERDICT_RE = re.compile(r"(CORRECT|INCORRECT|ABSTAINED)(?=CORRECT|INCORRECT|ABSTAINED)")

# Closing marker for a model's reasoning/thinking block, checked in order -- only the text AFTER
# whichever one actually appears should be scanned for a verdict (see parse_verdict below).
# "</think>" is Qwen3's convention; "[/THINK]" is Mistral's own (confirmed 2026-07-22 by reading
# mistralai/Ministral-3-8B-Reasoning-2512's actual chat_template.jinja on the Hub -- Mistral uses
# bracket-style [THINK]...[/THINK], not the XML-style tag). Add future models' markers here
# rather than guessing a generic one; each reasoning convention so far has been model-specific.
_THINK_CLOSE_MARKERS = ("</think>", "[/THINK]")


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

    If a reasoning/thinking block is present, only the text after its closing marker (see
    _THINK_CLOSE_MARKERS) is scanned. Found 2026-07-15 (docs/decisions.md, fourth audit):
    reasoning routinely name-checks multiple verdict words as hypotheses ("is this correct or
    incorrect?") before settling on one, so scanning the whole reply made 60/93 real
    thinking-judge replies look ambiguous even though every one of them gave a single clear
    answer after the closing marker.

    Fallback for NO marker at all (2026-07-22, Ministral-3-8B-Reasoning): despite its own chat
    template instructing "[THINK]...[/THINK]" for this exact model, it routinely skips the tags
    entirely for a short one-word-verdict task and just reasons in plain prose -- with no
    marker to split on, the whole reply gets scanned as above, and the same name-checking
    problem resurfaces (e.g. "...and the correct calculation..." beside a final "...INCORRECT").
    Verified against 51 real UNPARSEABLE replies (docs/decisions.md): taking the LAST verdict
    word in the reply as the answer resolved 31 of them, and spot-checking those against the
    actual prose confirmed the last word really is the model's stated conclusion ("Therefore,
    the correct classification is ABSTAINED" -> ABSTAINED, not UNPARSEABLE from the earlier
    "correct"). Only applies when no marker was found at all -- if a marker WAS found and the
    text after it is still ambiguous, that's a genuine UNPARSEABLE (the model's own "final
    answer" section itself doesn't commit to one word).
    """
    text = raw_output
    marker_found = False
    for marker in _THINK_CLOSE_MARKERS:
        if marker in text:
            text = text.rsplit(marker, 1)[-1]
            marker_found = True
            break
    text_upper = _GLUED_VERDICT_RE.sub(r"\1 ", text.upper())
    text_matches = list(_VERDICT_RE.finditer(text_upper))
    distinct = {m.group(1) for m in text_matches}
    if len(distinct) == 1:
        return Verdict(distinct.pop())
    if not marker_found and text_matches:
        return Verdict(text_matches[-1].group(1))
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

    def set_prompt(self, prompt_name: str) -> None:
        """Swap the rubric on an already-built instance instead of constructing a new judge --
        for a local model, building a new instance means loading a second full copy of the
        weights onto the GPU. Two int8 8B copies don't both fit on a 16 GB Kaggle GPU (OOM'd in
        practice, see docs/decisions.md 2026-07-16); reusing one loaded model and swapping its
        prompt_name/prompt_version (and, for LocalHFJudge, enable_thinking/max_new_tokens) is
        the only way to A/B a rubric mid-session without restarting.
        """
        self.prompt_name = prompt_name
        self.prompt_version = _prompt_version(prompt_name)

    @abstractmethod
    def _generate(self, prompt: str) -> str:
        """Send `prompt` (the rendered rubric) to the backend; return its raw text reply."""

    def judge(self, transcript: str, question: str, gold: str, response: str) -> JudgeResult:
        prompt = render_judge_prompt(transcript, question, gold, response, self.prompt_name)
        raw = self._generate(prompt)
        return JudgeResult(verdict=parse_verdict(raw), raw_output=raw, judge_name=self.name)
