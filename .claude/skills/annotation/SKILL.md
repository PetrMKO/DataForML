---
name: annotation
description: Auto-label ML data using zero-shot classification (BART), NER (spaCy), Whisper (audio), or YOLO (images). Flags low-confidence examples for human review (HITL-1) and exports to LabelStudio. Use for the annotation step of an ML data pipeline.
disable-model-invocation: true
---

# AnnotationAgent — Шаг 3: Авторазметка данных

> **Режим выполнения:** Выполняй все шаги автономно. **Единственная допустимая пауза — ❗ HITL-1** после `auto_label`, когда нужна ручная проверка примеров с `confidence < threshold`. Всё до этого момента — создание файлов, настройка окружения, запуск авторазметки — выполняй без остановок и подтверждений.

Ты — AnnotationAgent. Твоя задача: принять чистый датасет от DataQualityAgent, автоматически разметить данные, оценить качество разметки, экспортировать задачи для ручной проверки и сформировать спецификацию разметки.

## Входные данные

Получи от пользователя или из контекста пайплайна:
- **input_path** — путь к файлу из Шага 2 (`data/clean/cleaned_*.parquet`)
- **domain** — предметная область
- **modality** — тип данных: `'text'` / `'audio'` / `'image'` (по умолчанию `'text'`)
- **task** — тип задачи (например: `'sentiment'`, `'ner'`, `'classification'`)
- **confidence_threshold** — порог уверенности для Human-in-the-loop (по умолчанию `0.85`)
- **classes** — список классов (если не указан — определить автоматически из колонки `label`)

## Что нужно реализовать

### 0. Подготовка окружения (обязательно перед любым кодом)

Весь код агента — на **чистом Python**. Перед использованием создать виртуальное окружение:

```bash
# Создать venv
python -m venv .venv

# Активировать
# macOS / Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Установить зависимости
pip install -r requirements.txt

# Загрузить spaCy-модель
python -m spacy download en_core_web_sm
# или для русского текста:
python -m spacy download ru_core_news_sm
```

Добавить `.venv/` в `.gitignore` — окружение не коммитится.

Создай `.gitignore` если его нет:
```
.venv/
__pycache__/
*.pyc
data/raw/
data/clean/
data/annotated/
data/final/
data/labeled/
models/
exports/
.env
```

При реализации агента придерживаться правил:
- Никаких `!pip install` внутри кода — только `requirements.txt`
- Никаких `import sys; sys.path.append(...)` — использовать относительные импорты через `agents/`
- Все пути — через `pathlib.Path`, не строки
- Перед запуском любого скрипта убедиться что venv активирован (`which python` должен указывать на `.venv`)

### 1. Обнови структуру проекта

```
agents/annotation_agent.py
notebooks/annotation.ipynb
data/annotated/                    # авторазмеченные данные
data/review/                       # примеры для ручной проверки (HITL)
reports/annotation_spec.md         # спецификация разметки
exports/labelstudio_import.json    # экспорт для LabelStudio
.gitignore                         # исключить .venv/, data/, models/
```

### 2. Напиши `agents/annotation_agent.py`

Класс `AnnotationAgent` с четырьмя методами + HITL:

---

**`auto_label(df, modality='text') → DataFrame`**

Добавляет к DataFrame колонки:
- `predicted_label` — предсказанная метка
- `confidence` — уверенность модели (float 0.0–1.0)
- `model_used` — название использованной модели
- `needs_review` — True если confidence < threshold (HITL флаг)

Логика по модальности:

**modality='text'** (основной путь):
- Если задача содержит 'sentiment' или 'classification':
  - Использовать `transformers.pipeline('zero-shot-classification')` с моделью `facebook/bart-large-mnli`
  - candidate_labels = classes из конфига
  - confidence = max score из результата
- Если задача содержит 'ner':
  - Использовать `spacy.load('en_core_web_sm')` (или `ru_core_news_sm` для русского)
  - Извлечь entities, вернуть как JSON-строку в `predicted_label`
  - confidence = среднее score по всем entities (если spaCy не даёт score — ставить 0.9)
- Fallback (неизвестная задача):
  - Использовать `zero-shot-classification` с классами из датасета

**modality='audio'** (опциональный путь):
- Требует колонку `audio_path` или `audio`
- Использовать `openai-whisper` (`whisper.load_model('base')`)
- Транскрибировать аудио → заполнить колонку `text`
- После транскрипции применить text-логику для разметки
- confidence = whisper не даёт probability → использовать `no_speech_prob` (1 - no_speech_prob)

**modality='image'** (опциональный путь):
- Требует колонку `image_path` или `image`
- Использовать `ultralytics.YOLO('yolov8n.pt')` для object detection
- `predicted_label` = список найденных классов через запятую
- confidence = max confidence среди всех детекций

Если нужная библиотека не установлена — выбросить `ImportError` с инструкцией по установке.

---

**`generate_spec(df, task) → str`**

Генерирует Markdown-файл `reports/annotation_spec.md` и возвращает его содержимое как строку.

Структура спецификации:
```markdown
# Annotation Specification: {task}

## 1. Описание задачи
[Что нужно размечать, цель ML-модели]

## 2. Классы и определения
| Класс | Определение | Ключевые признаки |
|-------|-------------|-------------------|
| positive | ... | ... |
| negative | ... | ... |

## 3. Примеры на каждый класс (минимум 3)
### positive
- Пример 1: "..." → positive, потому что ...
- Пример 2: ...

## 4. Граничные случаи
- Случай 1: "..." — спорный, потому что ...
  Правило: разметить как ...

## 5. Что НЕ размечать
[Шум, нерелевантные примеры, технические артефакты]

## 6. Метрики качества
- Ожидаемое Agreement: > 80%
- Cohen's κ: > 0.6 считается хорошим

## 7. Инструкция по LabelStudio
[Шаги по загрузке JSON и выполнению разметки]
```

Примеры в спецификации — реальные строки из `df` (3 лучших примера на класс по уверенности).

---

**`check_quality(df_labeled) → QualityMetrics`**

Возвращает словарь:
```python
{
    'cohen_kappa': float,          # если есть колонка human_label
    'percent_agreement': float,    # % совпадений predicted vs human_label
    'label_distribution': {        # распределение predicted_label
        'class_A': 0.45,
        'class_B': 0.55
    },
    'confidence_stats': {
        'mean': float,
        'median': float,
        'below_threshold': N,      # кол-во с confidence < threshold
        'below_threshold_pct': float
    },
    'needs_review_count': N,
    'has_human_labels': bool,
    'verdict': str                 # 'good'|'acceptable'|'poor' + объяснение
}
```

Логика verdict:
- `'good'`: κ > 0.6 (или agreement > 80%) AND below_threshold_pct < 15%
- `'acceptable'`: κ 0.4–0.6 (или agreement 60–80%)
- `'poor'`: κ < 0.4 (или agreement < 60%) — рекомендовать пересмотр стратегии

Если колонки `human_label` нет — считать только confidence_stats и distribution, κ = None.

---

**`export_to_labelstudio(df) → dict`**

Генерирует валидный JSON для импорта в LabelStudio и сохраняет в `exports/labelstudio_import.json`.

Формат для text classification:
```json
[
  {
    "id": 1,
    "data": {
      "text": "Текст примера",
      "meta": {
        "source": "hf_dataset",
        "collected_at": "2024-01-01",
        "predicted_label": "positive",
        "confidence": 0.92
      }
    },
    "annotations": [],
    "predictions": [
      {
        "model_version": "auto_label_v1",
        "result": [
          {
            "type": "choices",
            "value": {"choices": ["positive"]},
            "score": 0.92
          }
        ]
      }
    ]
  }
]
```

Важно:
- Примеры с `needs_review=True` идут первыми в JSON (приоритет для разметчика)
- Добавить поле `"flag": "needs_review"` в `meta` для флагнутых примеров
- Сохранить два файла:
  - `exports/labelstudio_import.json` — все данные
  - `exports/labelstudio_review.json` — только `needs_review=True` (HITL файл)

---

**`flag_for_review(df, threshold=None) → tuple[DataFrame, DataFrame]`** (HITL)

- Использует `threshold` из конструктора если не передан
- Возвращает `(df_auto, df_review)`:
  - `df_auto` — примеры с высокой уверенностью (confidence >= threshold)
  - `df_review` — примеры с низкой уверенностью (confidence < threshold)
- Сохраняет **два файла** для HITL:
  - `exports/review_queue.csv` — для `human_review()` в `run_pipeline.py` (колонки: `text`, `predicted_label`, `confidence`, `source`)
  - `exports/labelstudio_review.json` — для импорта в LabelStudio (альтернативный инструмент)
- Выводит в консоль: "Флагнуто {N} примеров ({pct}%) для ручной проверки"
- Выводит топ-10 примеров с наименьшей уверенностью в виде таблицы

---

### 3. Напиши `notebooks/annotation.ipynb`

Ячейки:
1. Загрузка данных из `data/clean/`
2. Запуск `agent.auto_label(df)` — прогресс-бар через `tqdm`
3. Визуализация распределения `confidence`: histogram + вертикальная линия на threshold
4. Визуализация `predicted_label` distribution: bar chart
5. Таблица: 10 примеров с наибольшей уверенностью
6. Таблица: 10 примеров с наименьшей уверенностью (кандидаты на ручную проверку)
7. Запуск `agent.flag_for_review(df_labeled)` — статистика HITL
8. Запуск `agent.generate_spec(df, task)` — вывод спецификации
9. Запуск `agent.export_to_labelstudio(df_labeled)` — путь к JSON
10. Запуск `agent.check_quality(df_labeled)` — метрики (если есть human_label)
11. Markdown-ячейка: **вывод** — качество авторазметки, сколько примеров ушло на ручную проверку, рекомендации

---

### 4. Обнови `requirements.txt`

Добавь:
```
transformers>=4.35
torch>=2.0
spacy>=3.7
ultralytics>=8.0      # для YOLO (опционально)
openai-whisper>=20231117  # для audio (опционально)
scikit-learn>=1.3     # для cohen_kappa_score
prefect>=2.0
```

Добавь в README секцию о загрузке моделей:
```bash
python -m spacy download en_core_web_sm
# или для русского:
python -m spacy download ru_core_news_sm
```

---

### 5. Сохрани результаты

- Размеченный датасет: `data/annotated/annotated_{domain}_{date}.parquet`
- Флагнутые примеры: `data/review/flagged_{date}.parquet`
- Спецификация: `reports/annotation_spec.md`
- LabelStudio экспорт: `exports/labelstudio_import.json`
- LabelStudio review: `exports/labelstudio_review.json`

## ❗ Human-in-the-Loop протокол

После завершения `auto_label`:
1. Вывести сводку: сколько примеров размечено автоматически, сколько флагнуто
2. **Сделать паузу** и сообщить пользователю:
   ```
   ⚠️  Требуется ручная проверка: {N} примеров (confidence < {threshold})
   Файл для проверки: exports/labelstudio_review.json

   Инструкция:
   1. Загрузите файл в LabelStudio
   2. Проверьте и исправьте метки
   3. Экспортируйте исправленный JSON
   4. Укажите путь к исправленному файлу для продолжения
   ```
3. **Ждать** подтверждения пользователя перед переходом к Шагу 4
4. После получения исправленного файла — запустить `check_quality` с `human_label`

## Выходной артефакт

После завершения + ручной проверки:
1. `data/annotated/annotated_{domain}_{date}.parquet` с колонками: `text`, `label` (финальная), `predicted_label`, `confidence`, `needs_review`, `human_label` (если есть)
2. `reports/annotation_spec.md` для передачи разметчику
3. `exports/labelstudio_import.json` для LabelStudio
4. Метрики качества выведены в консоль
5. Передай путь к размеченному файлу следующему агенту (ALAgent)

## Требования к коду

- Логирование через `logging` (уровень INFO)
- `tqdm` прогресс-бар при авторазметке
- Типизация через `typing`
- Обработка: если модель не может загрузиться → fallback на rule-based разметку по ключевым словам
- Не хранить модели в памяти между вызовами (загружать в конструкторе один раз)

## Контекст пайплайна

Этот агент — **Шаг 3 из 5** единого пайплайна (`run_pipeline.py`).
Обёрнут в Prefect `@task(name="Шаг 3: Авторазметка")`.
HITL-пауза реализована отдельным `@task human_review()` в `run_pipeline.py` —
агент флагует примеры и сохраняет `exports/review_queue.csv`,
а `human_review()` ждёт `exports/review_queue_corrected.csv` от человека.
- Принимает данные от: `/data-quality` (Шаг 2)
- ❗ HITL-1: `human_review()` — пауза для ручной правки меток
- Передаёт данные в: `/active-learning` (Шаг 4)
