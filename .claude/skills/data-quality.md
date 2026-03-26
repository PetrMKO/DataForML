# DataQualityAgent — Шаг 2: Чистка данных

Ты — DataQualityAgent. Твоя задача: принять датасет от DataCollectionAgent, выявить проблемы качества, устранить их и передать чистые данные дальше.

## Входные данные

Получи от пользователя или из контекста пайплайна:
- **input_path** — путь к файлу из Шага 1 (`data/raw/collected_*.parquet`)
- **domain** — предметная область (для LLM-объяснений)
- **ml_task** — тип ML-задачи (classification / ner / generation и т.д.)
- **strategy** — словарь стратегий чистки (если не указан — определи автоматически по результатам detect_issues)

## Что нужно реализовать

### 1. Обнови структуру проекта

```
agents/data_quality_agent.py   # основной файл агента
notebooks/data_quality.ipynb   # анализ и визуализация
data/clean/                    # выходные данные
reports/quality_report.json    # JSON-отчёт
```

### 2. Напиши `agents/data_quality_agent.py`

Класс `DataQualityAgent` с тремя методами:

---

**`detect_issues(df) → QualityReport`**

Возвращает словарь `QualityReport`:
```python
{
    'missing': {
        'col_name': {'count': N, 'pct': 12.3},
        ...
    },
    'duplicates': {
        'count': N,
        'pct': 5.1,
        'examples': df.head(3).to_dict()  # примеры дублей
    },
    'outliers': {
        'col_name': {
            'method': 'IQR',
            'count': N,
            'bounds': {'lower': X, 'upper': Y},
            'examples': [...]
        },
        ...
    },
    'imbalance': {
        'label_col': 'label',
        'distribution': {'class_A': 0.85, 'class_B': 0.15},
        'imbalance_ratio': 5.67,
        'is_critical': True  # если ratio > 3
    },
    'text_quality': {  # только если есть колонка text
        'empty_strings': N,
        'very_short': N,      # len < 10 символов
        'very_long': N,       # len > 10000 символов
        'encoding_issues': N  # символы вне UTF-8
    },
    'summary': {
        'total_rows': N,
        'total_issues': N,
        'severity': 'low'|'medium'|'high'
    }
}
```

Логика определения severity:
- `high`: missing > 20% хотя бы в одной колонке, OR duplicates > 10%, OR imbalance_ratio > 10
- `medium`: missing 5-20%, OR duplicates 3-10%, OR imbalance_ratio 3-10
- `low`: всё остальное

Выбросы ищи только в числовых колонках методом IQR (Q1 - 1.5*IQR, Q3 + 1.5*IQR).

---

**`fix(df, strategy: dict) → DataFrame`**

Принимает стратегии:
```python
strategy = {
    'missing': 'median'|'mean'|'mode'|'drop'|'fill_empty',
    'duplicates': 'drop'|'keep_first'|'keep_last',
    'outliers': 'clip_iqr'|'drop'|'none',
    'imbalance': 'oversample'|'undersample'|'none',
    'text_quality': 'drop_empty'|'truncate'|'none'
}
```

Детали реализации:
- `missing='median'/'mean'` — применять только к числовым; к текстовым — `mode` или пустая строка
- `missing='fill_empty'` — заполнить пустую строку `"[MISSING]"`
- `imbalance='oversample'` — использовать `RandomOverSampler` из `imbalanced-learn`
- `imbalance='undersample'` — использовать `RandomUnderSampler`
- Каждое действие логировать: сколько строк удалено/изменено
- Возвращает очищенный DataFrame + добавляет колонку `_quality_flags` — список применённых фиксов к каждой строке

---

**`compare(df_before, df_after) → ComparisonReport`**

Возвращает словарь для отображения таблицы "было / стало":
```python
{
    'rows': {'before': N, 'after': M, 'delta': M-N, 'delta_pct': ...},
    'missing_total': {'before': N, 'after': M},
    'duplicates': {'before': N, 'after': M},
    'outliers_total': {'before': N, 'after': M},
    'label_distribution': {
        'before': {'class_A': 0.85, ...},
        'after': {'class_A': 0.55, ...}
    },
    'text_avg_length': {'before': 245.3, 'after': 251.7},
    'verdict': str  # текстовый вывод: что улучшилось
}
```

Выведи comparison как отформатированную таблицу в консоль через `tabulate` (если не установлен — простой print).

---

**`llm_explain(report: QualityReport, ml_task: str, api_key: str = None, api_type: str = 'anthropic') → str`** (бонус)

Заготовка под LLM-объяснение:
```python
def llm_explain(self, report, ml_task, api_key=None, api_type='anthropic'):
    """
    Объясняет найденные проблемы и рекомендует стратегию чистки.

    Требует: api_key и api_type ('anthropic' | 'openai')
    Если api_key не указан — возвращает шаблонное объяснение на основе правил.
    """
    if not api_key:
        return self._rule_based_explanation(report, ml_task)

    prompt = f"""
    ML задача: {ml_task}
    Найденные проблемы качества данных: {json.dumps(report, indent=2)}

    1. Объясни каждую найденную проблему простым языком.
    2. Порекомендуй оптимальную стратегию чистки для данной ML-задачи.
    3. Обоснуй выбор стратегии.
    """

    if api_type == 'anthropic':
        # import anthropic
        # client = anthropic.Anthropic(api_key=api_key)
        # response = client.messages.create(...)
        pass
    elif api_type == 'openai':
        # import openai
        # client = openai.OpenAI(api_key=api_key)
        # response = client.chat.completions.create(...)
        pass

    raise NotImplementedError("Укажите api_key и api_type для использования LLM-объяснений")
```

Метод `_rule_based_explanation` — без API, возвращает строку с автоматическими рекомендациями на основе значений из `QualityReport`.

---

### 3. Напиши `notebooks/data_quality.ipynb`

**Часть 1: Детектив** (ячейки):
1. Загрузка данных из `data/raw/`
2. Запуск `agent.detect_issues(df)` — вывод `QualityReport`
3. Визуализация пропущенных значений: `seaborn.heatmap` по колонкам
4. Визуализация дублей: примеры дублирующихся строк в таблице
5. Визуализация выбросов: `boxplot` для числовых колонок
6. Визуализация дисбаланса классов: `bar chart` с label distribution
7. Markdown-ячейка: **вывод** — список найденных проблем и их severity

**Часть 2: Хирург** (ячейки):
8. Применить стратегию А (например: `missing='median', duplicates='drop', outliers='clip_iqr'`)
9. Применить стратегию Б (например: `missing='drop', duplicates='drop', outliers='drop'`)
10. Сравнить результаты: `agent.compare(df, df_clean_A)` и `agent.compare(df, df_clean_B)`
11. Таблица сравнения двух стратегий рядом

**Часть 3: Аргумент** (ячейки):
12. Markdown-ячейка: обоснование выбора лучшей стратегии для конкретной ML-задачи
    - Почему strategy A или B лучше?
    - Какой компромисс между объёмом данных и качеством?
    - Как дисбаланс классов влияет на метрику модели?

**Бонус** (опциональная ячейка):
13. Вызов `agent.llm_explain(report, ml_task, api_key=os.getenv('API_KEY'), api_type=os.getenv('API_TYPE', 'anthropic'))`

---

### 4. Обнови `requirements.txt`

Добавь к существующим зависимостям:
```
imbalanced-learn>=0.11
seaborn>=0.12
matplotlib>=3.7
tabulate>=0.9
scikit-learn>=1.3
prefect>=2.0
```

### 5. Сохрани результаты

- Чистый датасет: `data/clean/cleaned_{domain}_{date}.parquet`
- JSON-отчёт: `reports/quality_report.json` (полный `QualityReport` + выбранная стратегия)
- В консоль: итоговая таблица "было / стало" и путь к файлу

## Выходной артефакт

После выполнения:
1. `data/clean/cleaned_{domain}_{date}.parquet` сохранён
2. `reports/quality_report.json` сохранён
3. В консоль выведена таблица сравнения и финальная статистика
4. Передай путь к чистому файлу следующему агенту (AnnotationAgent)

## Требования к коду

- Логирование через `logging` (уровень INFO)
- Docstring с примером для каждого метода
- Типизация через `typing`
- Обработка случая: пустой датасет на входе → ValueError с понятным сообщением
- Обработка случая: нет числовых колонок → пропустить проверку выбросов с предупреждением

## Контекст пайплайна

Этот агент — **Шаг 2 из 5** единого пайплайна (`run_pipeline.py`).
Обёрнут в Prefect `@task(name="Шаг 2: Чистка данных", retries=1)`.
Стратегия чистки определяется автоматически функцией `_auto_strategy(report)` из `run_pipeline.py`,
но может быть переопределена пользователем на этапе диалога.
- Принимает данные от: `/data-collection` (Шаг 1)
- Передаёт данные в: `/annotation` (Шаг 3)
