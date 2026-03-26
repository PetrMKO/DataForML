# DataForML

Автоматизированный пайплайн сбора и подготовки данных для ML-задач с поддержкой Human-in-the-Loop.

Запускается одной командой — сам собирает данные, чистит их, размечает и обучает базовую модель. В двух точках останавливается и просит человека проверить результат перед продолжением.

---

## Быстрый старт

```bash
# 1. Клонировать репозиторий и перейти в него
git clone <repo-url>
cd DataForML

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Установить зависимости
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Запустить пайплайн
python run_pipeline.py --domain "sentiment analysis" --ml_task classification
```

После старта пайплайн идёт сам. Потребуется ваше участие дважды — в точках Human-in-the-Loop (описаны ниже).

---

## Что делает пайплайн

Принимает описание задачи → возвращает обученную модель, размеченный датасет и отчёт.

```
[1/5] Сбор       → data/raw/
[2/5] Чистка     → data/clean/
[3/5] Разметка   → data/annotated/
      ❗ HITL-1: проверить примеры с низкой уверенностью
[4/5] AL-отбор   → data/final/
      ❗ HITL-2: подтвердить датасет перед обучением
[5/5] Обучение   → models/ + reports/
```

### Шаг 1 — Сбор данных (`DataCollectionAgent`)

Собирает данные из 2+ источников: HuggingFace датасеты, Kaggle, веб-скрапинг, API.
Унифицирует всё в единый формат: `text`, `label`, `source`, `collected_at`.

### Шаг 2 — Чистка (`DataQualityAgent`)

Находит и устраняет пропуски, дубли, выбросы, дисбаланс классов.
Сохраняет отчёт о качестве в `reports/quality_report.json`.

### Шаг 3 — Авторазметка (`AnnotationAgent`)

Размечает данные через zero-shot классификацию (text), Whisper (audio) или YOLO (image).
Примеры с уверенностью ниже порога (по умолчанию 0.85) флагуются для ручной проверки.

### ❗ HITL-1: Проверка разметки

Пайплайн останавливается и сохраняет `exports/review_queue.csv`.

```
Флагнуто: 47 примеров (confidence < 0.85)
Файл:     exports/review_queue.csv

Что делать:
1. Откройте файл в Excel / Google Sheets
2. Исправьте колонку predicted_label где нужно
3. Сохраните как exports/review_queue_corrected.csv
4. Введите путь к файлу (или "skip" чтобы пропустить)
```

### Шаг 4 — Active Learning (`ALAgent`)

Умный отбор наиболее информативных примеров для обучения.
Сравнивает стратегии `entropy` и `random`, строит learning curve.

### ❗ HITL-2: Подтверждение датасета

Перед обучением показывает финальную статистику датасета и спрашивает подтверждение.

### Шаг 5 — Обучение модели

TF-IDF + Logistic Regression (или DistilBERT если указать `--model bert`).
Сохраняет модель в `models/` и метрики в `reports/final_metrics.json`.

---

## Параметры запуска

```bash
python run_pipeline.py \
  --domain "название задачи"          # обязательно
  --ml_task classification            # classification | sentiment | ner | generation
  --modality text                     # text | audio | image (по умолчанию text)
  --config config.yaml                # путь к конфигу источников
  --confidence-threshold 0.85         # порог для HITL-1 (0.0–1.0)
```

Примеры:

```bash
# Классификация тональности отзывов
python run_pipeline.py --domain "product reviews sentiment" --ml_task sentiment

# NER для медицинских текстов
python run_pipeline.py --domain "medical NER" --ml_task ner --confidence-threshold 0.9

# Начать с шага 3 (данные уже собраны и очищены)
# Указать путь в интерактивном режиме через /data-pipeline → from 3
```

---

## Интерактивный режим (Claude Code Skills)

Вместо скрипта можно запустить через Claude Code CLI — скиллы ведут пайплайн интерактивно.

**Как вызвать:** введите `/команда` прямо в чат Claude Code (не в терминал).

### Запуск полного пайплайна

```
/data-pipeline
```

Claude спросит `domain`, `ml_task`, `modality` — и пойдёт сам. Остановится только в точках HITL.

### Запуск отдельных шагов

| Скилл | Шаг | Что делает | Нужно передать |
|-------|-----|------------|----------------|
| `/data-collection` | 1/5 | Сбор данных из 2+ источников | `domain` |
| `/data-quality` | 2/5 | Чистка: дубли, пропуски, выбросы | путь к `data/raw/*.parquet` |
| `/annotation` | 3/5 | Авторазметка + флагование low-confidence | путь к `data/clean/*.parquet` |
| `/active-learning` | 4/5 | AL-отбор информативных примеров | путь к `data/annotated/*.parquet` |

Примеры:

```
/data-collection        ← Claude спросит domain и создаст агента
/data-quality           ← Claude спросит путь к parquet и запустит чистку
/annotation             ← Claude спросит путь и запустит разметку
/active-learning        ← Claude спросит путь и запустит AL-отбор
```

### Команды управления во время работы

| Команда | Действие |
|---------|----------|
| `pause` | Остановить после текущего шага |
| `status` | Показать прогресс и пути к файлам |
| `skip N` | Пропустить шаг N |
| `from N` | Начать с шага N |
| `retry` | Повторить последний шаг |

---

## Структура выходных файлов

```
data/
├── raw/          собранные данные (после шага 1)
├── clean/        очищенные данные (после шага 2)
├── annotated/    размеченные данные (после шага 3)
├── final/        отобранные AL-стратегией (после шага 4)
└── labeled/      финальный датасет + data card

models/
└── model_YYYYMMDD.pkl     обученная модель + векторизатор

reports/
├── quality_report.json    отчёт по качеству данных
├── annotation_spec.md     спецификация разметки
├── al_report.json         история AL-цикла
├── learning_curve.png     график entropy vs random
├── final_metrics.json     accuracy, F1 итоговой модели
└── pipeline_report.md     итоговый отчёт (5 разделов)

exports/
├── labelstudio_import.json   все данные для LabelStudio
├── labelstudio_review.json   флагнутые примеры для LabelStudio
├── review_queue.csv          очередь для ручной проверки (HITL-1)
└── review_queue_corrected.csv исправленные метки (заполняет человек)
```

---

## Конфигурация источников (`config.yaml`)

```yaml
sources:
  - type: hf_dataset
    name: imdb
    split: train
    max_samples: 5000

  - type: scrape
    url: "https://example.com/reviews"
    selector: ".review-text"
    max_pages: 3

output:
  path: data/raw/
  format: parquet

api_keys:
  kaggle_username: ""
  kaggle_key: ""
```

Источники подбираются автоматически под `--domain` если не заданы явно.

---

---

## Зависимости

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # для английского NER
python -m spacy download ru_core_news_sm  # для русского NER (опционально)
```

Основные библиотеки: `pandas`, `scikit-learn`, `transformers`, `spacy`, `prefect`, `datasets`.

Опциональные (только если нужны audio/image): `openai-whisper`, `ultralytics`.

---

## Воспроизводимость

```bash
pip install -r requirements.txt
python run_pipeline.py --domain "your domain" --ml_task classification
```

Промежуточные результаты сохраняются в parquet на каждом шаге — пайплайн можно возобновить с любого места через `from N`.
