"""Dry-run test for the verdict-stopping logic in src/judges/local_hf.py.

Kept separate from tests/test_judges.py (which stays torch-free) since local_hf.py
imports torch at module level. Skips cleanly if torch isn't installed. No model
download -- _StopOnVerdict only needs something with a .decode(ids) -> str method,
so a stub stands in for the real tokenizer.
Run: python tests/test_local_hf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch  # noqa: F401
except ImportError:
    print("[skip] torch not installed -- test_local_hf.py needs it, skipping.")
    sys.exit(0)

from src.judges.local_hf import _StopOnVerdict


class _StubTokenizer:
    """Decodes a list-of-one-token-per-char id list back to text -- just enough
    to drive _StopOnVerdict without a real vocabulary or model.
    """

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(chr(i) for i in ids.tolist())


def _ids(text: str):
    return torch.tensor([ord(c) for c in text])


STOP_CASES = [
    ("still reasoning, no </think> yet", "Let me think about whether this is CORRECT", False),
    ("</think> but no verdict word yet", "reasoning...</think>The answer is", False),
    ("</think> then CORRECT", "reasoning...</think>The verdict is CORRECT", True),
    ("</think> then INCORRECT", "reasoning...</think>INCORRECT", True),
    ("verdict word only inside thinking, not stopped", "is it CORRECT? unsure...", False),
]


def test_stop_on_verdict() -> int:
    failures = 0
    tok = _StubTokenizer()
    for description, generated_text, expected in STOP_CASES:
        full = "PROMPT" + generated_text  # prompt_len chars belong to the prompt, not the reply
        stopper = _StopOnVerdict(tok, prompt_len=len("PROMPT"))
        got = stopper(_ids(full).unsqueeze(0), scores=None)
        ok = got == expected
        failures += not ok
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] stop={got} (expected {expected}) — {description}")
    return failures


def main() -> None:
    print("=== _StopOnVerdict ===")
    failures = test_stop_on_verdict()

    print("\n================ РЕЗУЛЬТАТ ================")
    if failures:
        print(f"[-] {failures} проверок не прошло.")
        sys.exit(1)
    print("[+] Все проверки прошли.")


if __name__ == "__main__":
    main()
