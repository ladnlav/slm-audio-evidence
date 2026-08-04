"""Small fixed (run_id, id) set for fast judge iteration -- runs in ~1-2 min instead of
the full ~93-item pilot audit, so a prompt or model change can be checked without paying
for a full Kaggle run each time. See docs/decisions.md 2026-07-15 (throughput discussion).

HARD: every category-B item that any past audit (no-think judge_v1, judge_v2, or the
reasoning-enabled judge) disagreed with manual-M1 on. These are the cases most likely to
flip when the prompt or model changes -- the ones worth actually looking at.

EASY: a random sample of items every past audit agreed on. Regression control -- if a
change breaks one of these, something is wrong even if the hard-set numbers improved.

Only use this subset to compare judge configurations against each other and against
manual-M1. It is NOT a substitute for the full audit before reporting final numbers --
23 items both over- and under-represents failure modes relative to the full pilot.
"""

DEV_SUBSET: dict[str, list[str]] = {
    "cascade_plain_20260712": [
        "sq-3032-B2", "sq-3647-B2", "sq-4398-B1",  # hard
        "sq-0229-B3", "sq-1628-B1", "sq-2005-B3", "sq-2571-B1", "sq-3203-B3",  # easy control
    ],
    "cascade_s1_idk_20260712": [
        "sq-2571-B1", "sq-3647-B2",  # hard
        "sq-0229-B3", "sq-1122-B1", "sq-1628-B3",  # easy control
    ],
    "qwen2audio_plain_20260712": [
        "sq-0005-B2", "sq-2005-B3", "sq-3555-B2", "sq-3647-B2", "sq-5092-B2", "sq-5207-B2",  # hard
        "sq-0525-B2", "sq-4471-B1",  # easy control
    ],
    "qwen2audio_s1_idk_20260712": [
        "sq-4398-B1", "sq-5207-B2",  # hard, no easy control sampled for this run
    ],
}
