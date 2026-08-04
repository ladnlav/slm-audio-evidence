# Judge-model comparison vs manual-M1 (93 gold B labels)

Sorted by agreement, best first. Per-model failures: `<judge_name>/compare_manual_m1.md` + `disagreements.jsonl` in this same folder. Full reasoning traces for thinking-enabled candidates: `thinking_traces/<judge_name>/`. `s/item` = Time (s) / Scored -- only category-B "answer" items actually call the judge, so this approximates per-call latency.

| Judge | Model | Thinking | Agreement | Scored | Unparseable | Disagreements | Time (s) | s/item |
|---|---|---|---|---|---|---|---|---|
| llm-gemma-4-12b-it-v1 | /kaggle/input/datasets/pasheviakova/llm-gemma-4-12b-it-v1-flat-cache | no | 96% (89/93) | 93 | 0 | 4 | 158 | 1.70 |
| llm-qwen3-8b-v1 | /kaggle/input/datasets/pasheviakova/qwen3-8b-flat-cache | no | 95% (88/93) | 93 | 0 | 5 | 101 | 1.09 |
| llm-qwen3.6-27b-v1 | /kaggle/input/datasets/pasheviakova/llm-qwen3-6-27b-v1-flat-cache | no | 95% (88/93) | 93 | 0 | 5 | 288 | 3.09 |
| llm-ministral-3-8b-instruct-2512-bf16-v1 | /kaggle/input/datasets/pasheviakova/llm-ministral-3-8b-instruct-2512-v1-flat-cache | no | 92% (86/93) | 93 | 0 | 7 | 185 | 1.99 |
| llm-qwen3.5-9b-v1 | /kaggle/input/datasets/pasheviakova/llm-qwen3-5-9b-v1-flat-cache | no | 92% (86/93) | 93 | 0 | 7 | 175 | 1.89 |
| llm-qwen3-8b-v1_geval_v1 | /kaggle/input/datasets/pasheviakova/qwen3-8b-flat-cache | no | 88% (82/93) | 93 | 0 | 11 | 1467 | 15.78 |
| llm-ministral-3-8b-instruct-2512-bf16-v1_geval_v1 | /kaggle/input/datasets/pasheviakova/llm-ministral-3-8b-instruct-2512-v1-flat-cache | no | 84% (78/93) | 93 | 0 | 15 | 4877 | 52.44 |
