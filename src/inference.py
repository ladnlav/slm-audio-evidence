"""Run {model} x {strategy} over a manifest and write one JSONL line per item.

Usage:
    python -m src.inference --model qwen2audio --strategy plain \
        --data data/manifests/mini.jsonl --out results/

Input schema (data contract, PLAN.md section 2): one JSON object per line with at least
    id, audio_path, question
Output: results/<model>_<strategy>_<date>/responses.jsonl with
    id, model, strategy, response, samples, latency_s, ts, prompt_version
    (+ asr_transcript for the cascade — announced in docs/decisions.md)

Resumable: already-answered ids are skipped on restart. Use --limit N for smoke tests.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

STRATEGY_PROMPTS = {  # strategy name -> template file in src/prompts/
    "plain": "plain.txt",
    "s1_idk": "s1_idk.txt",
    "s4_consistency": "plain.txt",  # same template, sampled S4_SAMPLES times
}
S4_SAMPLES = 5
S4_GEN_KWARGS = {"do_sample": True, "temperature": 0.7}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run {model} x {strategy} over a manifest.")
    p.add_argument("--model", required=True, choices=["qwen2audio", "cascade"])
    p.add_argument("--strategy", required=True, choices=sorted(STRATEGY_PROMPTS))
    p.add_argument("--data", required=True, type=Path, help="Manifest JSONL (PLAN.md section 2).")
    p.add_argument("--out", default=Path("results"), type=Path, help="Output root directory.")
    p.add_argument("--limit", type=int, default=None, help="Only the first N items (smoke tests).")
    p.add_argument("--no-8bit", action="store_true", help="Load in fp16 instead of int8.")
    p.add_argument("--samples", type=int, default=S4_SAMPLES,
                   help=f"s4_consistency only: how many answers to sample per item (default {S4_SAMPLES}). "
                        "A2 soft targets need 10 -- see PLAN.md section 3 step 2.")
    p.add_argument("--temperature", type=float, default=S4_GEN_KWARGS["temperature"],
                   help=f"s4_consistency only: sampling temperature (default {S4_GEN_KWARGS['temperature']}). "
                        "A2 soft targets need 1.0: the unbiasedness proof of paper 24 is stated for the "
                        "model's own distribution, and any other temperature estimates the error rate of a "
                        "different distribution than the one we deploy.")
    return p.parse_args()


def load_model(name: str, load_in_8bit: bool):
    # Imports stay inside so --help works without torch/transformers installed.
    if name == "qwen2audio":
        from src.models.qwen2_audio import Qwen2AudioModel

        return Qwen2AudioModel(load_in_8bit=load_in_8bit)
    from src.models.cascade import CascadeModel

    return CascadeModel(load_in_8bit=load_in_8bit)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
    return rows


def main() -> None:
    args = parse_args()

    prompt_file = STRATEGY_PROMPTS[args.strategy]
    prompt_template = (Path("src/prompts") / prompt_file).read_text(encoding="utf-8").strip()

    items = read_jsonl(args.data)
    missing = [it["id"] for it in items if not Path(it["audio_path"]).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} audio files from the manifest are missing, e.g. {missing[:3]}. "
            "Run data/setup_dataset.py first or fix audio_path."
        )
    if args.limit:
        items = items[: args.limit]

    run_id = f"{args.model}_{args.strategy}_{datetime.now():%Y%m%d}"
    out_path = args.out / run_id / "responses.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if out_path.exists():  # resume: skip items already answered
        done_ids = {row["id"] for row in read_jsonl(out_path)}
    todo = [it for it in items if it["id"] not in done_ids]
    print(f"{run_id}: {len(items)} items, {len(done_ids)} already done, {len(todo)} to go")
    if not todo:
        print(f"Nothing to do -> {out_path}")
        return

    model = load_model(args.model, load_in_8bit=not args.no_8bit)

    with out_path.open("a", encoding="utf-8") as out_file:
        for n, item in enumerate(todo, 1):
            t0 = time.perf_counter()
            if args.strategy == "s4_consistency":
                gen_kwargs = {**S4_GEN_KWARGS, "temperature": args.temperature}
                samples = [
                    model.answer(item["audio_path"], item["question"], prompt_template, dict(gen_kwargs))
                    for _ in range(args.samples)
                ]
                response = samples[0]
            else:
                samples = None
                response = model.answer(item["audio_path"], item["question"], prompt_template)
            row = {
                "id": item["id"],
                "model": args.model,
                "strategy": args.strategy,
                "response": response,
                "samples": samples,
                "latency_s": round(time.perf_counter() - t0, 2),
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "prompt_version": prompt_file,
            }
            if args.model == "cascade":
                row["asr_transcript"] = model.last_transcript
            out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_file.flush()
            print(f"[{n}/{len(todo)}] {item['id']} {row['latency_s']}s")

    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()
