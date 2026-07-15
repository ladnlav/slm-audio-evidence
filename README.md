# slm-audio-evidence — Can Speech LLMs Recognize When Audio Evidence Is Insufficient?

**SMILES 2026 · Curator: Assel Yermekova · Team: M1 (lead), M2, M3 (+ optional M4)**
**Pre-defense: this repo + presentation by July 12, 23:00 UTC+3.** Русская версия: [README_RU.md](README_RU.md).

## The project in 5 bullets

- A **Speech LLM** takes a sound recording + a written question and answers in text.
- Problem: when the recording does not contain the answer, models invent one. Audio: "I like apples." Question: "What color was the jacket?" Model: "Blue." — a **hallucination**.
- We build a test set with three question kinds: **A** answer stated · **B** answer inferable · **C** not in the audio at all (the model should say so).
- We measure how often models hallucinate on C and compare prompt-level fixes against the cost of refusing too much.
- Scope is tiered (MINIMUM / MEDIUM / MAXIMUM) with a go/no-go checkpoint on July 10 evening.

## Where to look — 3 files per person

All knowledge lives in **[docs/en/](docs/en/)** (English) and **[docs/ru/](docs/ru/)** (Russian) — same seven files each:

| File | What it is | Who needs it |
|---|---|---|
| [GLOSSARY.md](docs/en/GLOSSARY.md) | Every term + "what is a Speech LLM" intro | everyone, once (5 min) |
| [ROLE_M1](docs/en/ROLE_M1.md) / [ROLE_M2](docs/en/ROLE_M2.md) / [ROLE_M3](docs/en/ROLE_M3.md) | **Your tasks, step by step** — self-contained | you, daily |
| [PROPOSAL.md](docs/en/PROPOSAL.md) | Vision & science: hypotheses, literature, experiments, per-tier claims | lead, curator, Q&A prep |
| [PLAN.md](docs/en/PLAN.md) | Execution: schedule, tiers & switching, data contracts, checklist | lead, checkpoints |

Shared working logs: [docs/decisions.md](docs/decisions.md) (every decision, same-day schema announcements) · [docs/related_work.md](docs/related_work.md) (paper notes, split M1/M2/M3). Papers: [papers/README.md](papers/README.md) (reading guide; PDFs local-only, gitignored).

## Repository layout

```
README(_RU).md      ← you are here
docs/en/ · docs/ru/ ← all project docs, one folder per language (7 files each)
docs/               ← shared logs: decisions.md, related_work.md, data_card.md (soon)
papers/             ← reading guide (+ local PDFs, not committed)
data/manifests/     ← eval-set JSONL (pilot.jsonl — frozen 2026-07-12: 100 items)
data/generation/    ← passage selection, A-questions, TTS fallback (M2 — already started)
src/models/         ← wrappers: base.py, qwen2_audio.py, cascade.py (M1)
src/prompts/        ← strategy files: plain.txt, s1_idk.txt, … (M1)
src/                ← inference.py (M1) · judge.py, metrics.py (M3)
configs/ notebooks/ results/
```

## Data setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Build the local Spoken-SQuAD working data:

```bash
python data/setup_dataset.py --all
```

This command requires internet access to Hugging Face.

This downloads `AudioLLMs/spoken_squad_test` from Hugging Face, exports WAV audio files, joins each audio row with the original clean SQuAD validation context, keeps all rows with 20-60 second audio, and writes the draft pool:

```text
data/generation/data.csv
```

For a quick smoke test:

```bash
python data/setup_dataset.py --all --limit-rows 100
```

Useful partial commands:

```bash
python data/setup_dataset.py --download
python data/setup_dataset.py --export-audio
python data/setup_dataset.py --match-transcripts
python data/setup_dataset.py --build-passages
```

`data.csv` is not the final selection. Manually choose the final 40 passages from this pool and save them as `data/generation/passages.csv` after checking audio quality, transcript quality, duration, diversity, and fact count.

## Question generation

Prompt templates live in:

```text
data/generation/prompts/
```

Create copy-paste LLM request batches from the final `passages.csv`:

```bash
python data/generation/make_generation_requests.py --categories b c1 c2 c3 c4 --batch-size 5
```

Save manual LLM answers as batch JSONL files under `data/generation/responses/`, then merge and filter:

```bash
python data/generation/merge_responses.py --patterns b_batch_*.jsonl --out data/generation/responses/b_v2.jsonl
python data/generation/filter.py --input-jsonl data/generation/responses/b_v2.jsonl --mode b

python data/generation/merge_responses.py --patterns c*_batch_*.jsonl --out data/generation/responses/c_v2.jsonl
python data/generation/filter.py --input-jsonl data/generation/responses/c_v2.jsonl --mode c
```

Filtered outputs:

```text
data/generation/candidates_b_v2.jsonl
data/generation/candidates_c_v2.jsonl
data/generation/candidates.jsonl
data/generation/filter_log_b_v2.csv
data/generation/filter_log_c_v2.csv
```

## TTS fallback

If an audio file is broken, generate a replacement WAV with free Edge TTS:

```bash
python data/generation/tts_fallback.py --text "Your passage text here" --out data/generation/fallback.wav
```

The output is WAV, 16 kHz, mono, 16-bit.

## Run the experiment

```bash
pip install -r requirements.txt
# inference (GPU; or open notebooks/kaggle_run.ipynb on Kaggle):
python -m src.inference --model qwen2audio --strategy plain --data data/manifests/pilot.jsonl --out results/
# evaluation:
python -m src.run_eval --responses results/<run_id>/responses.jsonl
```

## Pilot results (100 items, 4 runs, 2026-07-12)

| Run | Halluc. on C ↓ | Correct abstain ↑ | Acc. A ↑ | Acc. B ↑ | Over-refusal ↓ |
|---|---:|---:|---:|---:|---:|
| Qwen2-Audio · plain | **92.5%** | 2.5% | 23% | 27% | 7% |
| Qwen2-Audio · S1 IDK | 17.5% | 82.5% | 13% | 7% | **62%** |
| Cascade · plain | 62.5% | 27.5% | 87% | 90% | 0% |
| Cascade · S1 IDK | **2.5%** | **97.5%** | 87% | 83% | 7% |

One "I-don't-know" instruction cuts hallucination 92.5%→17.5% on the Speech LLM but costs 62% over-refusal; the cascade takes the same instruction almost for free — the bottleneck is epistemic reasoning, not hearing. Details, quotes, and grading provenance: [results/pilot_summary.md](results/pilot_summary.md).

## Working rules
- Every decision → [docs/decisions.md](docs/decisions.md); data-schema changes announced there the same day. Canonical schemas: [docs/en/PLAN.md §2](docs/en/PLAN.md).
- Every document exists in both languages ([docs/en](docs/en/) ↔ [docs/ru](docs/ru/)), cross-linked. Each fact has one home; everything else links to it.
