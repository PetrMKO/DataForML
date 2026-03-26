# DataForML — Единый дата-пайплайн с Human-in-the-Loop

## Что это

ML-портфолио проект: автоматизированный сбор, чистка, разметка, умный отбор данных и обучение модели.
Пайплайн состоит из 4 агентов + шаг обучения модели, оркестрированных через **Prefect**.
Содержит явные Human-in-the-Loop точки, где человек проверяет и правит данные.

**Запуск одной командой:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py --domain "sentiment analysis" --ml_task classification
```

---

## Архитектура пайплайна

```
run_pipeline.py  ← точка входа, Prefect @flow
    │
    ├── [1/5] collect()          DataCollectionAgent   → data/raw/
    ├── [2/5] clean()            DataQualityAgent      → data/clean/
    ├── [3/5] auto_label()       AnnotationAgent       → data/annotated/
    │         ❗ HITL-1: human_review()               → review_queue_corrected.csv
    ├── [4/5] select()           ALAgent               → data/final/
    │         ❗ HITL-2: confirm_dataset()             → подтверждение перед обучением
    └── [5/5] train()            ModelTrainer          → models/ + reports/
```

### Агенты

| Шаг | Агент | Скилл | Выход |
|-----|-------|-------|-------|
| 1 | DataCollectionAgent | `/data-collection` | `data/raw/collected_*.parquet` |
| 2 | DataQualityAgent | `/data-quality` | `data/clean/cleaned_*.parquet` |
| 3 | AnnotationAgent | `/annotation` | `data/annotated/annotated_*.parquet` |
| 4 | ALAgent | `/active-learning` | `data/final/al_selected_*.parquet` |
| 5 | ModelTrainer | (встроен в pipeline) | `models/` + `reports/` |

---

## Быстрый старт

### 1. Запуск через скилл (интерактивный)

```
/data-pipeline
```

Claude спросит domain, modality, ml_task — и дальше ведёт пайплайн сам.

### 2. Запуск скриптом (воспроизводимый)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm

python run_pipeline.py --domain "sentiment analysis" --ml_task classification
```

### 3. Запуск с Prefect UI

```bash
prefect server start        # открыть http://localhost:4200
python run_pipeline.py ...  # запустить — видно прогресс в UI
```

---

## Human-in-the-Loop точки

### ❗ HITL-1: После авторазметки (Шаг 3)

Агент флагает примеры с `confidence < 0.85`:
1. Сохраняет `exports/review_queue.csv` — примеры для проверки
2. **Пауза** — ждёт файл `exports/review_queue_corrected.csv`
3. Человек открывает файл, правит колонку `label`, сохраняет
4. Пайплайн подхватывает исправленный файл и продолжается

### ❗ HITL-2: Перед обучением (между Шагом 4 и 5)

После AL-отбора агент показывает финальную статистику датасета:
- Размер, распределение классов, примеры
- Спрашивает: **"Датасет готов к обучению? [y/n]"**
- Только после подтверждения запускается обучение

---

## Структура файлов

```
DataForML/
├── CLAUDE.md
├── run_pipeline.py              ← единая точка входа (Prefect @flow)
├── config.yaml                  ← конфигурация источников и параметров
├── requirements.txt
│
├── agents/
│   ├── data_collection_agent.py
│   ├── data_quality_agent.py
│   ├── annotation_agent.py
│   └── al_agent.py
│
├── notebooks/
│   ├── eda.ipynb
│   ├── data_quality.ipynb
│   ├── annotation.ipynb
│   └── al_experiment.ipynb
│
├── data/
│   ├── raw/
│   ├── clean/
│   ├── annotated/
│   ├── final/
│   └── labeled/                 ← финальный датасет + data card
│
├── models/
│   └── model_*.pkl              ← обученная модель
│
├── reports/
│   ├── quality_report.json
│   ├── annotation_spec.md
│   ├── al_report.json
│   ├── learning_curve.png
│   ├── final_metrics.json       ← accuracy, F1 итоговой модели
│   └── pipeline_report.md       ← итоговый отчёт (5 разделов)
│
└── exports/
    ├── labelstudio_import.json
    ├── labelstudio_review.json
    ├── review_queue.csv          ← HITL-1: на проверку
    └── review_queue_corrected.csv ← HITL-1: после правки
```

---

## Итоговый отчёт (reports/pipeline_report.md)

Генерируется автоматически после Шага 5. Содержит 5 разделов:

1. **Описание задачи и датасета** — модальность, объём, классы, data card
2. **Что делал каждый агент** — решения и обоснования
3. **HITL-точки** — сколько примеров проверено, что исправлено
4. **Метрики на каждом этапе** — качество данных, разметки, AL, итоговая модель (accuracy/F1)
5. **Ретроспектива** — что сработало, что нет, что сделать иначе

---

## Управление пайплайном (через скилл)

| Команда | Действие |
|---------|----------|
| `pause` | Остановить после текущего шага |
| `status` | Показать прогресс и пути к файлам |
| `skip N` | Пропустить шаг N |
| `from N` | Начать с шага N (нужен путь к файлу) |
| `retry` | Повторить шаг после ошибки |

---

## Бонусы

- **+3 балла**: LLM-агент (Claude API) — Claude генерирует спецификацию разметки или объясняет ошибки. Активируется при передаче `--api-key`.
- **+2 балла**: Streamlit/Gradio дашборд для HITL-разметки — `python dashboard.py`

---

## Доступные скиллы

Скиллы вызываются командой `/имя` прямо в чат Claude Code (не в терминал).

| Скилл | Шаг | Описание | Входные данные |
|-------|-----|----------|----------------|
| `/data-pipeline` | 1–5 | Полный пайплайн с HITL — запрашивает `domain`, `ml_task`, `modality` и ведёт все шаги | — |
| `/data-collection` | 1/5 | Только сбор: создаёт `DataCollectionAgent`, `config.yaml`, `requirements.txt`, `eda.ipynb` | `domain` |
| `/data-quality` | 2/5 | Только чистка: находит и устраняет дубли, пропуски, выбросы, дисбаланс | путь к `data/raw/*.parquet` |
| `/annotation` | 3/5 | Только разметка: zero-shot/Whisper/YOLO, флагование low-confidence, экспорт в LabelStudio | путь к `data/clean/*.parquet` |
| `/active-learning` | 4/5 | Только AL-отбор: entropy vs random, learning curve, итоговый датасет | путь к `data/annotated/*.parquet` |

### Примеры использования

```
# Полный пайплайн — Claude спросит параметры:
/data-pipeline

# Только шаг 1 — для нового домена:
/data-collection

# Продолжить с шага 2 (данные уже собраны):
/data-quality

# Продолжить с шага 3:
/annotation

# Продолжить с шага 4:
/active-learning
```

### Файлы скиллов

| Скилл | Файл |
|-------|------|
| `/data-pipeline` | `.claude/skills/data-pipeline.md` |
| `/data-collection` | `.claude/skills/data-collection.md` |
| `/data-quality` | `.claude/skills/data-quality.md` |
| `/annotation` | `.claude/skills/annotation.md` |
| `/active-learning` | `.claude/skills/active-learning.md` |
