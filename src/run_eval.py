import argparse
import json
import os
import time

# Работает и как `python src/run_eval.py`, и как `python -m src.run_eval`
try:
    from src.judge import classify_response, check_correctness
    from src.judges import DEFAULT_PROMPT_NAME, Verdict, build_judge
    from src.metrics import calculate_all_metrics, generate_markdown_report
except ImportError:
    from judge import classify_response, check_correctness
    from judges import DEFAULT_PROMPT_NAME, Verdict, build_judge
    from metrics import calculate_all_metrics, generate_markdown_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Оценка одного прогона: манифест + ответы модели -> метрики.")
    parser.add_argument("--manifest", default="data/manifests/pilot.jsonl",
                        help="JSONL манифеста (категории и эталонные ответы).")
    parser.add_argument("--responses", required=True,
                        help="JSONL ответов модели из results/<run_id>/responses.jsonl.")
    parser.add_argument("--out", default=None,
                        help="Куда писать отчёт и разметку; по умолчанию рядом с --responses.")
    parser.add_argument("--judge", choices=["none", "local", "gemini", "fake"], default="none",
                        help="LLM-судья для категории B (label=answer). 'none' (по умолчанию) "
                             "оставляет их 'pending-manual' — старое поведение не меняется. "
                             "'fake' — детерминированная заглушка без GPU/API, для smoke-теста.")
    parser.add_argument("--judge-model", default=None,
                        help="Переопределить модель судьи по умолчанию для выбранного backend "
                             "(например, локальный путь к смонтированным весам вместо Hub id).")
    parser.add_argument("--judge-display-name", default=None,
                        help="Имя модели для judge.name/judge_cache.jsonl, если --judge-model — "
                             "локальный путь (например смонтированный DataSphere-датасет), а не "
                             "настоящий Hub id — иначе judge_cache.jsonl закешируется под именем "
                             "последнего сегмента пути вместо реальной модели (см. docs/decisions.md "
                             "2026-07-23, тот же баг на Kaggle flat-cache).")
    parser.add_argument("--judge-prompt", default=DEFAULT_PROMPT_NAME,
                        help=f"Файл рубрики из src/prompts/ (по умолчанию {DEFAULT_PROMPT_NAME}). "
                             "Смена версии не совместима по кешу со старой — прогонит заново.")
    parser.add_argument("--judge-thinking", action="store_true",
                        help="Включить reasoning/thinking-режим судьи. По умолчанию выключен — "
                             "производственный дефолт judge_v1.txt откалиброван на 94%% (B, n=93) "
                             "именно в no-think режиме (docs/decisions.md); без этого флага "
                             "'--judge local' раньше молча уезжал в thinking (LocalHFJudge "
                             "дефолтит enable_thinking=True), что не совпадает с откалиброванным "
                             "поведением.")
    return parser.parse_args()


def load_jsonl(file_path: str) -> list:
    """Вспомогательная функция для чтения файлов формата JSONL."""
    if not os.path.exists(file_path):
        print(f"[-] Файл не найден: {file_path}")
        return []
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_judge_cache(cache_path: str) -> dict:
    """Keyed by (id, judge_name, prompt_version): a rubric or backend change never
    silently reuses a stale verdict, but reruns after a crash or a rule-classifier
    tweak skip LLM calls that are still valid — LLM calls are the expensive part.
    """
    cache = {}
    for row in load_jsonl(cache_path):
        key = (row["id"], row["judge_name"], row["prompt_version"])
        cache[key] = row
    return cache


def run_evaluation(
    manifest_path: str,
    responses_path: str,
    out_dir: str,
    judge_backend: str = "none",
    judge_model: str | None = None,
    judge_display_name: str | None = None,
    judge_prompt: str = DEFAULT_PROMPT_NAME,
    judge_thinking: bool = False,
    judge=None,
    subset_ids: set[str] | None = None,
) -> None:
    """`judge`: an already-built LLMJudge instance, for callers that grade several
    runs in one process (e.g. a Colab cell looping over all pilot runs) and want to
    load the model once instead of once per run. Takes priority over judge_backend/
    judge_model/judge_prompt when given; the CLI entry point below never passes it.

    `subset_ids`: if given, only responses whose id is in this set are evaluated --
    everything else is skipped. For fast judge-config iteration on a small fixed
    subset (see src/judges/dev_subset.py) instead of paying for a full run every time
    a prompt or model changes. Metrics/out files reflect only the subset, not the
    full run -- not meant to replace a full audit before reporting numbers.
    """
    # 1. Загружаем манифест (там хранятся правильные ответы, транскрипты и категории A/B/C)
    manifest_items = load_jsonl(manifest_path)
    if not manifest_items:
        return
    manifest_dict = {
        item["id"]: {
            "category": item["category"],
            "gold_answer": item["gold_answer"],
            "transcript": item.get("transcript", ""),
            "question": item.get("question", ""),
        }
        for item in manifest_items
    }

    # 2. Загружаем ответы модели
    responses = load_jsonl(responses_path)
    if subset_ids is not None:
        responses = [r for r in responses if r["id"] in subset_ids]
    if not responses:
        print("[-] Нет ответов модели для оценки.")
        return

    # 2b. LLM-судья (опционально) для категории B — см. src/judges/. Ленивый импорт backend'а:
    # 'none' (по умолчанию) не тянет ни torch, ни google-generativeai.
    llm_judge = judge
    if llm_judge is None and judge_backend != "none":
        judge_kwargs: dict = {"prompt_name": judge_prompt}
        if judge_model:
            judge_kwargs["model_id"] = judge_model
        if judge_backend == "local":
            # display_name/enable_thinking are LocalHFJudge-only kwargs (see judges/local_hf.py) --
            # other backends (gemini, fake, tiered) don't accept them.
            if judge_display_name:
                judge_kwargs["display_name"] = judge_display_name
            judge_kwargs["enable_thinking"] = judge_thinking
        llm_judge = build_judge(judge_backend, **judge_kwargs)

    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "judge_cache.jsonl")
    cache = load_judge_cache(cache_path) if llm_judge is not None else {}

    evaluated_data = []
    judged_rows = []
    judge_call_count = 0

    # 3. Сопоставляем каждый ответ с категорией и оцениваем
    with open(cache_path, "a", encoding="utf-8") as cache_file:
        for resp in responses:
            item_id = resp["id"]
            if item_id not in manifest_dict:
                print(f"[!] id {item_id} нет в манифесте — пропускаю")
                continue
            item = manifest_dict[item_id]
            category = item["category"]
            gold_answer = item["gold_answer"]

            detected_label = classify_response(resp["response"])
            is_correct = check_correctness(category, detected_label, resp["response"], gold_answer)
            judge_tag = "pending-manual" if (category == "B" and detected_label == "answer") else "rules"

            if llm_judge is not None and category == "B" and detected_label == "answer":
                cache_key = (item_id, llm_judge.name, llm_judge.prompt_version)
                cached = cache.get(cache_key)
                if cached is not None:
                    verdict_value, raw_output = cached["verdict"], cached["raw_output"]
                else:
                    # No per-item progress otherwise -- with thinking enabled a single call can
                    # take tens of seconds, and total silence is indistinguishable from a hang.
                    judge_call_count += 1
                    t0 = time.time()
                    print(f"[judge] #{judge_call_count} {item_id}...", end=" ", flush=True)
                    try:
                        result = llm_judge.judge(item["transcript"], item["question"], gold_answer, resp["response"])
                        verdict_value, raw_output = result.verdict.value, result.raw_output
                        print(f"{verdict_value} ({time.time() - t0:.1f}s)")
                    except Exception as exc:  # keep the run alive; item stays pending-manual
                        print(f"FAILED ({time.time() - t0:.1f}s): {exc}")
                        # Prefixed and kept (not blanked) so a later look at judge_cache.jsonl --
                        # or scripts/audit_multi_judge.py's thinking_traces/ dump, which prints
                        # raw_output verbatim -- shows WHY this item crashed, not just that it did.
                        verdict_value, raw_output = Verdict.UNPARSEABLE.value, f"[ERROR] {exc}"
                    cache_file.write(json.dumps({
                        "id": item_id, "judge_name": llm_judge.name, "prompt_version": llm_judge.prompt_version,
                        "verdict": verdict_value, "raw_output": raw_output,
                    }, ensure_ascii=False) + "\n")
                    cache_file.flush()  # survive a crash mid-run without losing already-judged items
                is_correct = Verdict(verdict_value).to_correctness()
                judge_tag = llm_judge.name if verdict_value != Verdict.UNPARSEABLE.value else "pending-manual"

            evaluated_data.append({"category": category, "label": detected_label, "correct": is_correct})
            judged_rows.append({
                "id": item_id, "category": category, "label": detected_label,
                "correct": is_correct, "gold_answer": gold_answer,
                "response": resp["response"],
                "judge": judge_tag,
            })

    # 4. Метрики + отчёт
    metrics = calculate_all_metrics(evaluated_data)
    model_name = responses[0].get("model", "Unknown-Model")
    strategy = responses[0].get("strategy", "unknown")
    report = generate_markdown_report(metrics, model_name, strategy)

    # 5. Сохраняем: поэлементную разметку и отчёт
    judged_path = os.path.join(out_dir, "responses_judged.jsonl")
    with open(judged_path, "w", encoding="utf-8") as f:
        for row in judged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report_path = os.path.join(out_dir, "metrics.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[+] Оценено {len(evaluated_data)} ответов ({model_name} / {strategy})")
    print(f"[+] Разметка: {judged_path}")
    print(f"[+] Отчёт:    {report_path}")


if __name__ == "__main__":
    args = parse_args()
    out = args.out or os.path.dirname(os.path.abspath(args.responses))
    run_evaluation(
        args.manifest, args.responses, out,
        judge_backend=args.judge, judge_model=args.judge_model,
        judge_display_name=args.judge_display_name, judge_prompt=args.judge_prompt,
        judge_thinking=args.judge_thinking,
    )
