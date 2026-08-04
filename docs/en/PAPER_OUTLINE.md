# PAPER_OUTLINE — skeleton, owners, requirements

Русская версия: [../ru/PAPER_OUTLINE.md](../ru/PAPER_OUTLINE.md) · Plan: [PLAN.md](PLAN.md)

## Hard requirements (school rules)

- Template: **Zapiski POMI** (LaTeX, in [`paper/`](../../paper/)); compile on **Overleaf** (nobody needs local LaTeX).
- **≤ 10 pages** main text; appendices unlimited. Non-anonymous.
- **Every author registered on OpenReview before submission**; mentors listed on the paper.
- Mandatory **Contribution section**: who did what, per member.
- Submit via OpenReview by **Sun Aug 2, 21:00 MSK**; our internal deadline **Sat Aug 1 evening**.

## Working title

*Can Speech LLMs Recognize When Audio Evidence Is Insufficient? A Diagnostic Study and a Pre-Generation Probing Mitigation* — second half drops under plan-minimum.

## Skeleton and owners

Tags: **[MIN]** = stays under plan-minimum · **[A2]** = only if probing succeeds (PLAN §5).

| § | Section (target length) | Owner → editor | Content |
|---|---|---|---|
| — | Abstract (0.5 p) [MIN] | M1 | written last, Jul 31 |
| 1 | Introduction (1 p) [MIN] | M3 → M1 | **draft ready:** [`paper/introduction.tex`](../../paper/introduction.tex). Problem; the hearing-vs-reasoning diagnostic question; the TTS pilot result; **its refutation on natural speech** (the cascade's edge was a TTS artifact, IDK-prompting does not scale); 3 contribution bullets |
| 2 | Related Work (1–1.5 p) [MIN] | M3 → M1 | **draft ready:** [`paper/related_work.tex`](../../paper/related_work.tex) — 2 subsections: unanswerability benchmarks in audio QA (01, 03) · mitigation for audio LLMs (02, 26, 27) and positioning vs 25 (*output-level* uncertainty there, *pre-generation* here). Bibliography: [`paper/references.bib`](../../paper/references.bib), 5 entries flagged `TODO verify title` |
| 3 | Task & Diagnostic Set (1.5 p) [MIN] | M2 → M3 | task definition (answer or abstain); A/B/C + C1–C4 taxonomy; pilot-100 construction, 80% agreement, freeze; scale-set construction: native SQuAD 2.0 questions over NMSQA natural speech (48 passages, 40 speakers, ~500 A/C) + the TTS twin subset |
| 4 | Methods (1.5–2 p) | M4 (probing subsection with M1) → M1 | systems (Qwen2-Audio, cascade) [MIN]; prompts plain/S1 [MIN]; judge pipeline + 94% agreement + audit table [MIN]; probing: capture → soft targets → linear/attention probes → threshold-abstain [A2] |
| 5 | Experiments & Results (2–2.5 p) | M2+M4 (+M1: probe tables) → M1 | pilot table (4 runs, Wilson CIs) [MIN]; scale-set validation table (native SQuAD 2.0 on natural speech) + "TTS vs natural speech" table on the twin subset [MIN]; AUROC table probes vs entropy [A2]; operating-points plot probe→abstain vs plain/S1 [A2] |
| 6 | Discussion (0.5–1 p) [MIN] | M1 | what the cascade gap means; why pre-generation matters; two-team landscape |
| 7 | Limitations (0.5 p) [MIN] | M1 | 100 pilot items/wide CIs; one model family; LLM judge; English-only; pilot on TTS audio (scale set on natural speech); **Qwen2-Audio's 30-second window**: pilot passages are 31–55 s, our check found 96% of A answers inside the audible span, no such check is possible for B — residual risk |
| 8 | Conclusion & Future Work (0.3 p) [MIN] | M1 | under minimum: probing *is* the future work |
| 9 | **Contribution statement** [MIN] | each → M1 | 2–4 sentences per member, collected by Jul 30 |
| — | Acknowledgements [MIN] | M1 | mentors; Amina/Zaytsev if consulted |
| — | References [MIN] | M3 | BibTeX collected *during* the reading conveyor — save entries as you read |
| App | Appendices | owners of §§ | prompt texts verbatim; per-subtype C breakdown; graded-answer examples; judge prompt; extra probe tables |

## What already exists → where it goes

| Artifact | Goes to |
|---|---|
| [results/pilot_summary.md](../../results/pilot_summary.md) (4-run table, quotes) | §5 pilot table + §3 examples |
| [data_card.md](../data_card.md) | §3 + appendix |
| M4's judge audit (94%, model comparison) | §4 + appendix |
| [related_work.md](../related_work.md) bullets | §2 |
| Pre-defense slides figures (pipeline diagram) | §1/§3 figure 1 |
| decisions.md freeze/grading entries | §3 provenance sentences |

## Writing rules

- Numbers only from files in `results/` — no numbers from memory or chat.
- Terms exactly as in [GLOSSARY.md](GLOSSARY.md); M3 runs the final consistency pass.
- Every claim about prior work carries a citation; M3 verifies against the PDF, not the abstract.
- Write sections in the template from day one (no Google-Docs detour): `paper/` on Overleaf, M1 owns the project, M2 fixes build breakages.
