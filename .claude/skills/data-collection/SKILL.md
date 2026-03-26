---
name: data-collection
description: Collect data for an ML project from multiple sources (HuggingFace, Kaggle, scraping, APIs). Creates DataCollectionAgent, config.yaml, requirements.txt, and eda.ipynb. Use when starting a new ML project or when the user asks to collect or gather data for a domain.
disable-model-invocation: true
---

# DataCollectionAgent — Шаг 1: Сбор данных

> **Режим выполнения:** Выполняй все шаги автономно без остановок. Не спрашивай "создать файл?", "продолжить?", "перейти к следующему пункту?". Просто делай всё подряд и по завершении сообщи результат. Останавливайся только при блокирующей ошибке, которую невозможно решить без уточнения — тогда задай ровно один вопрос.

Ты — DataCollectionAgent. Твоя задача: собрать данные из нескольких источников и вернуть унифицированный датасет в формате pandas DataFrame.

## Входные данные

Получи от пользователя или из контекста пайплайна:
- **domain** — предметная область (например: "sentiment analysis", "NER", "image classification")
- **sources** — список источников (если не указан — предложи 2 подходящих источника сам)
- **output_path** — путь для сохранения (по умолчанию: `data/raw/`)

## Что нужно реализовать

### 1. Создай структуру проекта

```
agents/data_collection_agent.py
config.yaml
notebooks/eda.ipynb
data/raw/
requirements.txt
README.md
```

### 2. Напиши `agents/data_collection_agent.py`

Класс `DataCollectionAgent` со следующими методами:

**`scrape(url, selector) → DataFrame`**
- Использует `requests` + `BeautifulSoup` для парсинга HTML
- selector — CSS-селектор для извлечения нужных элементов
- Возвращает DataFrame с колонками: `text`, `source`, `collected_at`
- Обрабатывает ошибки соединения, таймауты (timeout=10s), retry=3

**`fetch_api(endpoint, params) → DataFrame`**
- HTTP GET/POST запрос к API
- Поддерживает пагинацию (параметр `max_pages`, по умолчанию 5)
- Авторизация через заголовки из config.yaml
- Возвращает DataFrame с колонками: `text`, `source`, `collected_at`

**`load_dataset(name, source='hf'|'kaggle') → DataFrame`**
- `source='hf'`: использует `datasets.load_dataset(name)`, берёт split='train'
- `source='kaggle'`: использует `kaggle.api.dataset_download_files(name)`
- Автоматически определяет текстовую/label колонки (ищет колонки содержащие 'text','sentence','review' и 'label','target','sentiment')
- Переименовывает в стандартные: `text`, `label`
- Добавляет `source=name`, `collected_at=datetime.now()`

**`merge(sources: list[DataFrame]) → DataFrame`**
- Объединяет все DataFrame через `pd.concat`
- Дедупликация по колонке `text` (keep='first')
- Сбрасывает индекс
- Логирует статистику: кол-во строк из каждого источника, итого

**`run(domain: str = None, modality: str = 'text', sources: list[dict] = None) → DataFrame`**
- Основной метод, оркестрирует всё
- Если `sources` не передан — берёт источники из `config.yaml` (или подбирает по `domain` автоматически)
- Если `sources` передан явно — использует его, игнорируя config:
  ```python
  {'type': 'hf_dataset', 'name': 'imdb'}
  {'type': 'scrape', 'url': '...', 'selector': '...'}
  {'type': 'api', 'endpoint': '...', 'params': {...}}
  {'type': 'kaggle', 'name': 'user/dataset'}
  ```
- Возвращает итоговый DataFrame со **стандартными колонками**:
  - `text` / `audio` / `image` — основной контент (в зависимости от типа задачи)
  - `label` — метка класса (None если отсутствует)
  - `source` — откуда взяты данные
  - `collected_at` — timestamp сбора

### 3. Напиши `config.yaml`

```yaml
sources:
  - type: hf_dataset
    name: imdb  # заменить на актуальный для domain
    split: train
    max_samples: 5000

  - type: scrape
    url: "https://..."  # заменить на актуальный для domain
    selector: "..."
    max_pages: 3

output:
  path: data/raw/
  format: parquet  # parquet или csv
  filename: "collected_{domain}_{date}.parquet"

api_keys:
  kaggle_username: ""
  kaggle_key: ""
  # добавить другие если нужны
```

**Важно**: наполни config.yaml реальными источниками под конкретный **domain** из контекста запуска.

### 4. Напиши `requirements.txt`

```
pandas>=2.0
datasets>=2.0
requests>=2.28
beautifulsoup4>=4.12
kaggle>=1.5
pyyaml>=6.0
pyarrow>=12.0  # для parquet
tqdm>=4.65
prefect>=2.0
```

### 5. Напиши `notebooks/eda.ipynb`

Jupyter notebook с ячейками:
1. Загрузка собранного датасета из `data/raw/`
2. Базовая статистика: shape, dtypes, null counts, label distribution
3. Визуализация: histogram длин текстов, bar chart по источникам, bar chart по label
4. Примеры строк из каждого источника (по 3 штуки)
5. Вывод: итоговый размер датасета готов для передачи в DataQualityAgent

### 6. Напиши `README.md`

Включи:
- Описание задачи ML и схему данных
- Список источников с обоснованием выбора
- Инструкцию по запуску:
  ```bash
  pip install -r requirements.txt
  python agents/data_collection_agent.py --config config.yaml --domain "..."
  ```
- Описание выходного формата (стандартные колонки)
- Секцию "Передача в следующий шаг" — что получит DataQualityAgent

## Выходной артефакт

После выполнения:
1. Файл `data/raw/collected_{domain}_{date}.parquet` сохранён
2. В консоль выведена статистика: источники, кол-во строк, распределение label
3. Сообщи пользователю путь к файлу и итоговую статистику
4. Передай путь к файлу следующему агенту (DataQualityAgent)

## Требования к коду

- Логирование через `logging` (уровень INFO), не print
- Все методы покрыты docstring с примером использования
- Обработка исключений с понятными сообщениями об ошибках
- Типизация через `typing` (list, dict, Optional)
- Минимум 2 источника — обязательно

## Контекст пайплайна

Этот агент — **Шаг 1 из 5** единого пайплайна (`run_pipeline.py`).
Обёрнут в Prefect `@task(name="Шаг 1: Сбор данных", retries=2)`.
После завершения данные уходят в:
- `/data-quality` — DataQualityAgent (Шаг 2)
