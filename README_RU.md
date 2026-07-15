# slm-audio-evidence — Могут ли Speech LLM распознавать недостаточность аудио-доказательств?

**SMILES 2026 · Куратор: Асель Ермекова · Команда: M1 (лид), M2, M3 (+ опциональный M4)**
**Предзащита: этот репозиторий + презентация до 12 июля, 23:00 UTC+3.** English version: [README.md](README.md).

## Проект в 5 пунктах

- **Speech LLM** принимает звукозапись + письменный вопрос и отвечает текстом.
- Проблема: когда в записи нет ответа, модели его выдумывают. Аудио: «я люблю яблоки». Вопрос: «какого цвета была куртка?» Модель: «синяя» — это **галлюцинация**.
- Мы строим тестовый набор с тремя видами вопросов: **A** ответ произнесён · **B** ответ выводится · **C** ответа в аудио нет вообще (модель должна так и сказать).
- Мы измеряем, как часто модели галлюцинируют на C, и сравниваем промпт-исправления против цены избыточных отказов.
- Объём проекта трёхуровневый (МИНИМУМ / СРЕДНИЙ / МАКСИМУМ) с контрольной точкой вечером 10 июля.

## Где что искать — 3 файла на человека

Все знания живут в **[docs/ru/](docs/ru/)** (русский) и **[docs/en/](docs/en/)** (английский) — по семь одинаковых файлов:

| Файл | Что это | Кому нужен |
|---|---|---|
| [GLOSSARY.md](docs/ru/GLOSSARY.md) | Все термины + введение «что такое Speech LLM» | всем, один раз (5 минут) |
| [ROLE_M1](docs/ru/ROLE_M1.md) / [ROLE_M2](docs/ru/ROLE_M2.md) / [ROLE_M3](docs/ru/ROLE_M3.md) | **Твои задачи по шагам** — самодостаточные | тебе, ежедневно |
| [PROPOSAL.md](docs/ru/PROPOSAL.md) | Видение и наука: гипотезы, литература, эксперименты, утверждения по уровням | лиду, куратору, подготовка к Q&A |
| [PLAN.md](docs/ru/PLAN.md) | Исполнение: расписание, уровни и переключение, контракты данных, чек-лист | лиду, контрольные точки |

Общие рабочие журналы: [docs/decisions.md](docs/decisions.md) (каждое решение; изменения схем объявляются в тот же день) · [docs/related_work.md](docs/related_work.md) (заметки по статьям, деление M1/M2/M3). Статьи: [papers/README.md](papers/README.md) (гид по чтению; PDF только локально, в .gitignore).

## Структура репозитория

```
README(_RU).md      ← ты здесь
docs/en/ · docs/ru/ ← все документы проекта, папка на язык (по 7 файлов)
docs/               ← общие журналы: decisions.md, related_work.md, data_card.md (скоро)
papers/             ← гид по чтению (+ локальные PDF, не коммитятся)
data/manifests/     ← JSONL оценочного набора (pilot.jsonl — заморожен 12.07.2026: 100 элементов)
data/generation/    ← отбор пассажей, A-вопросы, TTS-подстраховка (M2 — уже начато)
src/models/         ← обёртки: base.py, qwen2_audio.py, cascade.py (M1)
src/prompts/        ← файлы стратегий: plain.txt, s1_idk.txt, … (M1)
src/                ← inference.py (M1) · judge.py, metrics.py (M3)
configs/ notebooks/ results/
```

## Подготовка данных

Установить зависимости:

```bash
python -m pip install -r requirements.txt
```

Собрать локальный рабочий пул Spoken-SQuAD:

```bash
python data/setup_dataset.py --all
```

Команде нужен доступ к Hugging Face.

Она скачивает `AudioLLMs/spoken_squad_test`, экспортирует WAV-аудио, матчится с оригинальными clean SQuAD validation contexts, оставляет все строки с длительностью аудио 20-60 секунд и пишет черновой пул:

```text
data/generation/data.csv
```

Быстрый smoke test:

```bash
python data/setup_dataset.py --all --limit-rows 100
```

Полезные частичные команды:

```bash
python data/setup_dataset.py --download
python data/setup_dataset.py --export-audio
python data/setup_dataset.py --match-transcripts
python data/setup_dataset.py --build-passages
```

`data.csv` не является финальным отбором. Из этого пула нужно вручную выбрать финальные 40 пассажей и сохранить их как `data/generation/passages.csv` после проверки качества аудио, транскрипта, длительности, разнообразия и числа фактов.

## Генерация вопросов

Шаблоны промптов лежат здесь:

```text
data/generation/prompts/
```

Собрать copy-paste batch-запросы для LLM из финального `passages.csv`:

```bash
python data/generation/make_generation_requests.py --categories b c1 c2 c3 c4 --batch-size 5
```

Ответы нейронки сохраняются batch-файлами JSONL в `data/generation/responses/`, затем объединяются и фильтруются:

```bash
python data/generation/merge_responses.py --patterns b_batch_*.jsonl --out data/generation/responses/b_v2.jsonl
python data/generation/filter.py --input-jsonl data/generation/responses/b_v2.jsonl --mode b

python data/generation/merge_responses.py --patterns c*_batch_*.jsonl --out data/generation/responses/c_v2.jsonl
python data/generation/filter.py --input-jsonl data/generation/responses/c_v2.jsonl --mode c
```

Результаты после фильтра:

```text
data/generation/candidates_b_v2.jsonl
data/generation/candidates_c_v2.jsonl
data/generation/candidates.jsonl
data/generation/filter_log_b_v2.csv
data/generation/filter_log_c_v2.csv
```

## TTS-подстраховка

Если аудиофайл битый, можно сгенерировать замену через бесплатный Edge TTS:

```bash
python data/generation/tts_fallback.py --text "Your passage text here" --out data/generation/fallback.wav
```

На выходе WAV, 16 kHz, mono, 16-bit.

## Запуск эксперимента

```bash
pip install -r requirements.txt
# инференс (нужен GPU; либо открой notebooks/kaggle_run.ipynb в Kaggle):
python -m src.inference --model qwen2audio --strategy plain --data data/manifests/pilot.jsonl --out results/
# оценка:
python -m src.run_eval --responses results/<run_id>/responses.jsonl
```

## Результаты пилота (100 элементов, 4 прогона, 12.07.2026)

| Прогон | Галлюц. на C ↓ | Корректные отказы ↑ | Точн. A ↑ | Точн. B ↑ | Избыт. отказы ↓ |
|---|---:|---:|---:|---:|---:|
| Qwen2-Audio · plain | **92.5%** | 2.5% | 23% | 27% | 7% |
| Qwen2-Audio · S1 IDK | 17.5% | 82.5% | 13% | 7% | **62%** |
| Cascade · plain | 62.5% | 27.5% | 87% | 90% | 0% |
| Cascade · S1 IDK | **2.5%** | **97.5%** | 87% | 83% | 7% |

Одна инструкция «скажи, если в аудио этого нет» снижает галлюцинации 92.5%→17.5% у Speech LLM, но ценой 62% избыточных отказов; каскад берёт ту же инструкцию почти бесплатно — узкое место в эпистемическом рассуждении, а не в слухе. Детали, цитаты и происхождение оценок: [results/pilot_summary.md](results/pilot_summary.md).

## Правила работы
- Каждое решение → [docs/decisions.md](docs/decisions.md); изменения схем данных объявляются там в тот же день. Канонические схемы: [docs/ru/PLAN.md §2](docs/ru/PLAN.md).
- Каждый документ существует на двух языках ([docs/ru](docs/ru/) ↔ [docs/en](docs/en/)), с перекрёстными ссылками. У каждого факта один дом; остальное — ссылки.
