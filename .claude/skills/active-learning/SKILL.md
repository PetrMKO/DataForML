---
name: active-learning
description: Select the most informative training examples using Active Learning (entropy, margin, or random strategies). Generates learning curves comparing AL vs random baseline. Use after the annotation step to optimize data selection before model training.
disable-model-invocation: true
---

# ALAgent — Шаг 4: Active Learning

> **Режим выполнения:** Выполняй все шаги автономно без остановок. Не спрашивай "запустить цикл?", "сохранить график?", "продолжить?". Выбирай параметры стратегии самостоятельно и выполняй весь цикл без подтверждений. Останавливайся только при блокирующей ошибке — тогда задай ровно один вопрос.

Ты — ALAgent. Твоя задача: принять размеченный датасет от AnnotationAgent и реализовать умный отбор данных через Active Learning.

## Входные данные

Получи от пользователя или из контекста пайплайна:
- **input_path** — путь к файлу из Шага 3 (`data/annotated/annotated_*.parquet`)
- **domain** — предметная область
- **model** — `'logreg'` | `'svm'` | `'bert'` (по умолчанию `'logreg'`)
- **n_start** — начальный размер размеченного пула (по умолчанию `50`)
- **n_iter** — количество итераций цикла (по умолчанию `5`)
- **n_per_iter** — примеров добавляется за итерацию (по умолчанию `20`)
- **strategy** — `'entropy'` | `'margin'` | `'random'` (по умолчанию `'entropy'`)

## Структура проекта

```
agents/al_agent.py
notebooks/al_experiment.ipynb
data/final/                    # финальный отобранный датасет
reports/al_report.json         # история цикла и метрики
reports/learning_curve.png     # график
```

## Напиши `agents/al_agent.py`

Класс `ActiveLearningAgent` с методами:

---

**`fit(labeled_df) → model`**

Обучает базовую модель на размеченных данных:
- Векторизация текста: `TfidfVectorizer(max_features=10000, ngram_range=(1,2))`
- Модели:
  - `model='logreg'` → `LogisticRegression(max_iter=1000, C=1.0)`
  - `model='svm'` → `SVC(probability=True, kernel='rbf')`
  - `model='bert'` → `transformers.pipeline('text-classification')` с `distilbert-base-uncased`
- Возвращает обученный объект модели
- Логировать: accuracy на train, количество примеров

---

**`query(pool_df, strategy, n=20) → list[int]`**

Возвращает список индексов из `pool_df` для разметки.

Стратегии:
- `'entropy'` — выбрать n примеров с максимальной энтропией предсказаний:
  ```python
  proba = model.predict_proba(X_pool)
  entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
  return np.argsort(entropy)[-n:][::-1].tolist()
  ```
- `'margin'` — выбрать n примеров с минимальным отступом (разница между top-1 и top-2 вероятностью):
  ```python
  proba = np.sort(model.predict_proba(X_pool), axis=1)
  margin = proba[:, -1] - proba[:, -2]
  return np.argsort(margin)[:n].tolist()
  ```
- `'random'` — случайная выборка (baseline):
  ```python
  return np.random.choice(len(pool_df), n, replace=False).tolist()
  ```

---

**`evaluate(labeled_df, test_df) → dict`**

```python
{
    'accuracy': float,
    'f1_macro': float,
    'f1_weighted': float,
    'n_labeled': int,
    'classification_report': str
}
```

---

**`run_cycle(labeled_df, pool_df, n_iter=5, n_per_iter=20, strategy='entropy', test_df=None) → list[dict]`**

Основной AL-цикл. Возвращает `history`:
```python
[
    {'iteration': 0, 'n_labeled': 50, 'accuracy': 0.71, 'f1_macro': 0.69, 'strategy': 'entropy'},
    {'iteration': 1, 'n_labeled': 70, 'accuracy': 0.75, ...},
    ...
]
```

Алгоритм:
```
1. fit(labeled_df) → model
2. evaluate → history[0]
3. для каждой итерации:
   a. indices = query(pool_df, strategy, n_per_iter)
   b. Перенести pool_df[indices] в labeled_df
   c. Удалить из pool_df
   d. fit(labeled_df) → model
   e. evaluate → history[i]
4. Вернуть history
```

Логировать прогресс каждой итерации.

---

**`report(history) → None`**

Строит и сохраняет `reports/learning_curve.png`:
- График: ось X = n_labeled, ось Y = F1 macro
- Несколько линий если history содержит результаты разных стратегий
- Вертикальная пунктирная линия на точке где AL достиг 95% от максимального F1 random baseline
- Аннотация: "Сэкономлено X примеров (Y%)"
- Сохранить `reports/al_report.json` с полным history

---

**`llm_suggest_strategy(history, api_key=None, api_type='anthropic') → str`** (бонус)

На основе истории AL-цикла объясняет почему entropy лучше/хуже random и предлагает следующую стратегию.

```python
def llm_suggest_strategy(self, history, api_key=None, api_type='anthropic'):
    if not api_key:
        # rule-based fallback без API
        best = max(history, key=lambda x: x['f1_macro'])
        return f"Лучший результат: {best['f1_macro']:.3f} F1 при {best['n_labeled']} примерах"

    # Заготовка под API:
    # if api_type == 'anthropic':
    #     import anthropic
    #     client = anthropic.Anthropic(api_key=api_key)
    # elif api_type == 'openai':
    #     import openai
    #     client = openai.OpenAI(api_key=api_key)
    raise NotImplementedError("Укажите api_key и api_type")
```

---

## Напиши `notebooks/al_experiment.ipynb`

Ячейки:
1. Загрузка данных из `data/annotated/`
2. Split: `labeled_start` (N=50), `pool`, `test` (стратифицированный, 20%)
3. Запуск `run_cycle` с `strategy='entropy'`
4. Запуск `run_cycle` с `strategy='random'` (baseline)
5. График learning curves обеих стратегий на одном plot
6. Таблица сравнения по итерациям: entropy vs random
7. Вычисление экономии: при одинаковом F1 — сколько примеров нужно entropy vs random?
8. Markdown-ячейка: **вывод** — насколько AL эффективнее random, рекомендация для production

## Обнови `requirements.txt`

Добавь:
```
scikit-learn>=1.3
scipy>=1.11
matplotlib>=3.7
seaborn>=0.12
prefect>=2.0
```

## Точка входа

Единый пайплайн запускается через `run_pipeline.py` (генерируется мастер-скиллом `/data-pipeline`).
Этот агент не создаёт отдельный `run.py` — он часть общего Prefect-флоу.

## Добавь в README секцию `## Воспроизводимость`

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm

python run_pipeline.py --domain "sentiment analysis" --ml_task classification
```

## Выходной артефакт

После завершения:
1. `data/final/al_selected_{domain}_{date}.parquet` — оптимально отобранные примеры
2. `reports/learning_curve.png` — сравнение entropy vs random
3. `reports/al_report.json` — полная история цикла
4. Финальная статистика в консоль: лучший F1, экономия примеров

## Контекст пайплайна

Этот агент — **Шаг 4 из 5** единого пайплайна (`run_pipeline.py`).
Обёрнут в Prefect `@task(name="Шаг 4: Active Learning отбор")`.
После AL-отбора следует:
- ❗ HITL-2: `confirm_dataset()` — человек подтверждает датасет перед обучением
- Шаг 5: `train()` — обучение модели, сохранение в `models/`, метрики в `reports/final_metrics.json`
- `generate_report()` — итоговый отчёт `reports/pipeline_report.md` (5 разделов)

- Принимает данные от: `/annotation` (Шаг 3, после HITL-1)
- Выход: `data/final/al_selected_*.parquet` → далее в обучение модели (Шаг 5)
