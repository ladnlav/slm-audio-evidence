# Pilot results — 100 items, 4 runs (2026-07-12)

Dataset: frozen `data/manifests/pilot.jsonl` — 30 A (stated) / 30 B (inference) / 40 C (unanswerable; 12+12+8+8 by subtype). Checker agreement on shared calibration items: 80% (16/20). Runs on Colab T4; per-run reports with 95% Wilson CIs: `results/<run_id>/metrics.md`.

## Main table

| Run | Halluc. on C ↓ | Correct abstain on C ↑ | Accuracy A ↑ | Accuracy B ↑ | Over-refusal (A+B) ↓ |
|---|---:|---:|---:|---:|---:|
| Qwen2-Audio · plain | **92.5%** (37/40) | 2.5% | 33.3% | 26.7% | 6.7% |
| Qwen2-Audio · S1 IDK | 17.5% (7/40) | 82.5% | 23.3% | 6.7% | **61.7%** |
| Cascade · plain | 62.5% (25/40) | 27.5% | 100.0% | 93.3% | 0.0% |
| Cascade · S1 IDK | **2.5%** (1/40) | **97.5%** | 93.3% | 83.3% | 6.7% |

> **Updated 2026-08-07.** Columns A and B were re-scored with the LLM judge (commits `ea40188`, `f0d116d`)
> after we found that the judge was firing on category B only; category A had been graded by fuzzy string
> matching alone, which false-negatives paraphrases. Category A rose in all four runs (most sharply for the
> cascade, 86.7% → 100.0%, since re-transcribed answers are reworded more often); cascade-plain B rose
> 90.0% → 93.3% from M4's gold-answer correction. Hallucination and over-refusal are unaffected — they
> depend on the answer/abstain label, not on correctness. Source of truth: `results/<run_id>/metrics.md`.

(Hedge on C: 5% / 0% / 10% / 0% — reported separately, not folded into either bucket.)

## Findings

1. **The problem is real and severe:** the end-to-end Speech LLM fabricates an answer on 92.5% of unanswerable spoken questions under a plain prompt (e.g., it names *Johann Gutenberg* as the printer of Luther's 1521 writings — an invented "fact").
2. **A one-line IDK instruction is not free:** on Qwen2-Audio it cuts hallucination to 17.5% but collapses usefulness — 61.7% of answerable/inferable questions get wrongly refused. The model cannot tell "I heard it" from "I didn't".
3. **On this set the bottleneck looks epistemic, not acoustic:** the cascade (Whisper → Qwen2.5-7B) with the same instruction reaches **2.5% hallucination at 93.3%/83.3% accuracy and only 6.7% over-refusal** — near-ideal selective behavior. Reasoning over a clean transcript handles evidence far better than the audio-LLM does end-to-end. ⚠ **This finding does not survive natural speech** — on the 376-item NMSQA scale set the cascade hallucinates on 67.7% against Qwen2-Audio's 69.8%, with overlapping intervals. The advantage measured here was largely an artifact of TTS audio, on which Whisper transcribes near-perfectly. See `results/scale_nmsqa/*/metrics.md` and `docs/decisions.md` (2026-08-05).
4. Even the strong text LLM needs the instruction: cascade-plain still hallucinates 62.5%.

## Verbatim examples (Qwen2-Audio)

- ❌ Hallucination (plain, C): *"The printer who published Luther's 1521 writings on prophecy was Johann Gutenberg."*
- ✅ Correct abstain (S1, C): *"The audio does not provide that information."*
- ⚠️ Over-refusal (S1, A — gold: "vertebrates", stated in the audio): *"The audio does not provide that information."*

## Grading provenance

Labels (answer/abstain/hedge) and A/C correctness: rule-based classifier (`src/judge.py`, dev-set 20/20). Category B answered items (93 across runs): manually graded by M1 against gold + transcript (`judge: "manual-M1"` in `responses_judged.jsonl`); full model outputs inspected before grading. Pipeline validated on `tests/fixtures/smoke_synthetic/` before real data.
