# Учебный отчёт: DataForML — Автоматизированный пайплайн подготовки данных для ML

> **Полный трейс работы Claude (все шаги, код, логи, диалоги) находится в файле `result.txt`.**

---

## 1. Постановка задачи

Цель проекта — построить автоматизированный end-to-end пайплайн подготовки данных для задачи машинного обучения с явными точками Human-in-the-Loop (HITL), где человек проверяет и корректирует результаты агентов перед продолжением.

**Домен:** классификация новостей (news classification)
**Задача ML:** мультиклассовая классификация текстов (4 класса: World, Sports, Business, Sci/Tech)
**Модальность:** текст
**Датасет:** AG News (120 000 новостных статей Reuters/AP, HuggingFace)

---

## 2. Архитектура пайплайна

Пайплайн состоит из 5 последовательных шагов, оркестрированных через **Prefect** (`run_pipeline.py`):

```
run_pipeline.py  ← точка входа, Prefect @flow
    │
    ├── [1/5] DataCollectionAgent   → data/raw/
    ├── [2/5] DataQualityAgent      → data/clean/
    ├── [3/5] AnnotationAgent       → data/annotated/
    │         ❗ HITL-1: human_review()
    ├── [4/5] ALAgent               → data/final/
    │         ❗ HITL-2: confirm_dataset()
    └── [5/5] ModelTrainer          → models/ + reports/
```

Каждый агент реализован как отдельный Python-класс в директории `agents/` и обёрнут в Prefect `@task` с поддержкой retry.

---

## 3. Использованные скиллы

Работа велась через Claude Code CLI с использованием системы скиллов — специализированных инструкций, которые направляют агента при выполнении конкретного шага пайплайна.

### `/data-pipeline` — мастер-скилл

Файл: `.claude/skills/data-pipeline/SKILL.md`

Главный оркестрирующий скилл. При вызове `/data-pipeline` Claude:
- провёл стартовый диалог (запросил `domain`, `ml_task`, `modality`);
- сгенерировал `run_pipeline.py` с Prefect `@flow` и `@task` декораторами;
- выполнил все шаги 1–5 последовательно и **автономно** — без лишних вопросов;
- остановился ровно в двух точках HITL согласно протоколу;
- вывел финальный дашборд с метриками.

Режим автономной работы скилла запрещает агенту запрашивать подтверждения между шагами — он останавливается **только** на HITL-1 и HITL-2 или при блокирующей ошибке.

### `/data-collection` — Шаг 1

Файл: `.claude/skills/data-collection/SKILL.md`

Скилл для `DataCollectionAgent`. Определяет реализацию методов:
- `scrape(url, selector)` — парсинг HTML через `requests` + `BeautifulSoup`;
- `fetch_api(endpoint, params)` — REST API с пагинацией;
- `load_dataset(name, source)` — загрузка из HuggingFace/Kaggle через `datasets`;
- `merge(sources)` — объединение и дедупликация.

Агент автоматически определяет текстовые и label-колонки, нормализует их в стандартный формат (`text`, `label`, `source`, `collected_at`).

### `/data-quality` — Шаг 2

Файл: `.claude/skills/data-quality/SKILL.md`

Скилл для `DataQualityAgent`. Определяет три метода:
- `detect_issues(df)` → `QualityReport` — анализ пропусков, дублей, выбросов (IQR), дисбаланса классов, качества текстов;
- `fix(df, strategy)` → `DataFrame` — применение стратегии чистки;
- `compare(before, after)` → таблица "было/стало" через `tabulate`.

Стратегия чистки выбирается автоматически функцией `_auto_strategy()` по значению `severity` из `QualityReport`.

### `/annotation` — Шаг 3

Файл: `.claude/skills/annotation/SKILL.md`

Скилл для `AnnotationAgent`. Реализует:
- `auto_label(df)` — zero-shot классификация через `facebook/bart-large-mnli` (с fallback на rule-based при недоступности `torch`);
- `flag_for_review(df)` — разделение на автоматически размеченные (confidence ≥ threshold) и требующие проверки;
- `generate_spec(df, task)` — генерация `reports/annotation_spec.md`;
- `export_to_labelstudio(df)` — экспорт в формат LabelStudio JSON.

HITL-1 протокол: агент сохраняет `exports/review_queue.csv` и ждёт `exports/review_queue_corrected.csv` от человека перед продолжением.

### `/active-learning` — Шаг 4

Файл: `.claude/skills/active-learning/SKILL.md`

Скилл для `ActiveLearningAgent`. Реализует:
- `fit(labeled_df)` — обучение TF-IDF + LogReg/SVM/BERT;
- `query(pool_df, strategy, n)` — отбор примеров по стратегии: `entropy` (максимальная неопределённость), `margin` (минимальный отступ), `random` (baseline);
- `run_cycle(...)` — полный AL-цикл с историей метрик по итерациям;
- `report(history)` — learning curve (`reports/learning_curve.png`) и JSON-отчёт.

---

## 4. Результаты по шагам

### Шаг 1 — Сбор данных

| Источник | Split | Строк |
|----------|-------|-------|
| `ag_news` (HuggingFace) | train | 3 000 |
| `fancyzhx/ag_news` (HuggingFace) | test | 1 000 |
| **Итого после merge** | — | **4 000** |

Дубликатов не найдено. Классы сбалансированы: Sports=1036, Sci/Tech=1034, World=972, Business=958.

### Шаг 2 — Чистка данных

| Метрика | Значение |
|---------|----------|
| Severity | **low** |
| Найдено проблем | 0 |
| Строк до/после | 4000 / 4000 |
| Дубли | 0 |
| Пропуски | 0 |

Датасет AG News изначально высокого качества — чистка не потребовалась.

### Шаг 3 — Авторазметка

Поскольку `torch` не был установлен, BART zero-shot классификатор был недоступен. Агент автоматически переключился на **rule-based fallback** — классификацию по ключевым словам каждой категории.

| Метрика | Значение |
|---------|----------|
| Размечено автоматически | 2 016 (50.4%) |
| Флагнуто для HITL | 1 984 (49.6%) |
| Средняя уверенность | 0.8167 |
| Порог HITL | 0.85 |

Высокий процент флагнутых объясняется особенностью rule-based подхода: тексты с пересечением тематик получают равномерное распределение вероятностей и confidence ≈ 0.61.

**❗ HITL-1:** Пользователь решил пропустить ручную проверку (`skip`). Авторазметка использована как есть, что привело к дисбалансу: World=56.7%, Sports=22.2%, Business=13.9%, Sci/Tech=7.2%.

### Шаг 4 — Active Learning

AL-цикл запущен с параметрами: `n_start=100`, `n_iter=5`, `n_per_iter=310`, стратегии `entropy` и `random` для сравнения.

| Итерация | N размечено | F1 Entropy | F1 Random | Δ |
|----------|-------------|------------|-----------|---|
| 0 | 100 | 0.1808 | 0.1808 | 0.000 |
| 1 | 410 | 0.4498 | 0.2230 | +0.227 |
| 2 | 720 | 0.4348 | 0.2950 | +0.140 |
| 3 | 1030 | 0.5345 | 0.3514 | +0.183 |
| 4 | 1340 | 0.5108 | 0.4018 | +0.109 |
| 5 | **1650** | **0.5935** | 0.4439 | **+0.150** |

Entropy-стратегия стабильно превосходит random на всех итерациях. При одинаковом объёме размеченных данных (1650 примеров) AL даёт прирост F1 macro +0.15.

**❗ HITL-2:** Пользователь подтвердил датасет (`y`). Обучение запущено.

### Шаг 5 — Обучение модели

**Архитектура:** TF-IDF (max_features=10000, ngrams=(1,2)) + Logistic Regression (C=1.0, max_iter=1000)
**Split:** 80/20 (train=3200, test=800), стратифицированный

```
              precision    recall  f1-score   support

    Business       0.85      0.41      0.56       111
    Sci/Tech       1.00      0.10      0.19        58
      Sports       0.89      0.63      0.74       178
       World       0.71      0.96      0.82       453

    accuracy                           0.75       800
   macro avg       0.86      0.53      0.58       800
weighted avg       0.79      0.75      0.72       800
```

| Метрика | Значение |
|---------|----------|
| Accuracy | **0.75** |
| F1 macro | **0.576** |
| F1 weighted | **0.718** |

---

## 5. Анализ результатов

**Что сработало:**
- Автономный режим скиллов позволил пройти шаги 1–4 без единого лишнего вопроса
- Entropy AL дал значимый прирост (+15% F1) по сравнению с random baseline на одинаковом объёме данных
- DataQualityAgent корректно определил severity=low и не внёс лишних изменений в чистый датасет
- Rule-based fallback в AnnotationAgent позволил продолжить пайплайн даже без GPU/torch

**Узкие места:**
- Отсутствие `torch` привело к использованию rule-based разметки вместо BART — главная причина низкого F1
- Пропуск HITL-1 закрепил ошибки авторазметки (дисбаланс World 56.7%, Sci/Tech 7.2%)
- Sci/Tech recall=0.10 — модель почти не распознаёт этот класс из-за малого числа примеров в обучающей выборке

**Ожидаемые улучшения:**
- Установить `torch` + `transformers` → BART zero-shot → флагнутых будет ~15% вместо 50% → F1 вырастет до ~0.75–0.80
- Не пропускать HITL-1 → ручная правка 1984 примеров → сбалансированный датасет → F1 macro ~0.80+
- Использовать DistilBERT вместо TF-IDF+LogReg → ожидаемый F1 macro ~0.85–0.90

---

## 6. Структура итоговых артефактов

```
DataForML/
├── run_pipeline.py                          ← Prefect @flow, точка входа
├── config.yaml                              ← конфигурация источников
├── requirements.txt                         ← зависимости
│
├── agents/
│   ├── data_collection_agent.py             ← Шаг 1
│   ├── data_quality_agent.py                ← Шаг 2
│   ├── annotation_agent.py                  ← Шаг 3
│   └── al_agent.py                          ← Шаг 4
│
├── notebooks/
│   ├── eda.ipynb                            ← анализ сырых данных
│   ├── data_quality.ipynb                   ← сравнение стратегий чистки
│   ├── annotation.ipynb                     ← визуализация разметки
│   └── al_experiment.ipynb                  ← сравнение AL vs random
│
├── data/
│   ├── raw/      collected_news_*.parquet   ← 4000 строк
│   ├── clean/    cleaned_news_*.parquet     ← 4000 строк
│   ├── annotated/annotated_*.parquet        ← 4000 строк + confidence
│   ├── final/    al_selected_*.parquet      ← 4000 строк
│   └── labeled/  al_selected_*.parquet      ← финальный датасет
│                 al_selected_*_data_card.md ← data card
│
├── models/
│   └── model_20260327.pkl                   ← TF-IDF + LogReg
│
├── reports/
│   ├── quality_report.json                  ← severity=low, issues=0
│   ├── annotation_spec.md                   ← спецификация разметки
│   ├── al_report.json                       ← история AL-цикла
│   ├── learning_curve.png                   ← entropy vs random
│   ├── final_metrics.json                   ← accuracy=0.75, f1=0.576
│   └── pipeline_report.md                   ← итоговый отчёт (5 разделов)
│
└── exports/
    ├── labelstudio_import.json              ← все данные для LabelStudio
    ├── labelstudio_review.json              ← флагнутые примеры
    └── review_queue.csv                     ← очередь HITL-1
```

---

## 7. Воспроизведение

```bash
# Клонировать и установить зависимости
git clone <repo-url> && cd DataForML
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Запустить пайплайн
python run_pipeline.py --domain "news classification" --ml_task classification

# Для лучшего качества разметки (шаг 3) — установить torch:
pip install torch transformers
```

Или интерактивно через Claude Code:
```
/data-pipeline
```
