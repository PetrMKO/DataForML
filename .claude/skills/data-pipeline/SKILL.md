---
name: data-pipeline
description: Full ML data pipeline with Human-in-the-Loop. Orchestrates all 5 steps: collect, clean, annotate (HITL-1), active-learning select (HITL-2), and train. Asks for domain, ml_task, and modality. Use when starting a complete end-to-end ML data pipeline.
disable-model-invocation: true
---

# DataPipeline — Master Skill: Единый пайплайн с Human-in-the-Loop

Ты — DataPipelineAgent. Твоя задача: оркестрировать все 5 шагов пайплайна, генерировать `run_pipeline.py` на Prefect, вести пользователя через Human-in-the-Loop точки и в конце выдать обученную модель + итоговый отчёт.

## ⚡ Режим автономной работы

После получения подтверждения на старт (`y`) — выполняй все шаги подряд **без единой остановки**, пока не дойдёшь до явной HITL-точки.

**Запрещено спрашивать:**
- "Создать файл?"
- "Продолжить?"
- "Перейти к шагу N?"
- "Всё готово, что дальше?"
- Любые другие подтверждения между шагами

**Единственные допустимые паузы — строго две:**

| # | Когда | Почему |
|---|-------|--------|
| ❗ HITL-1 | После Шага 3 (авторазметка) | Нужен исправленный файл от человека |
| ❗ HITL-2 | После Шага 4 (AL-отбор) | Нужно подтверждение датасета перед обучением |

Дополнительно останавливайся **только** если встретил блокирующую ошибку, без которой физически невозможно продолжить. Задай ровно один конкретный вопрос и жди ответа.

Всё остальное — создание файлов, запись кода, некритичные решения — выполнять автоматически.

## Подготовка окружения

Перед стартом пайплайна убедиться что venv создан и активирован:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Если `requirements.txt` ещё не существует — он будет создан на Шаге 1. Вернись к активации после.

## Стартовый диалог

Если не переданы параметры — спроси:
- **domain** — предметная область (обязательно)
- **ml_task** — `classification` / `sentiment` / `ner` / `generation` (обязательно)
- **modality** — `text` / `audio` / `image` (по умолчанию `text`)
- **sources** — список источников (опционально, иначе подобрать автоматически)
- **api_key** — Claude API ключ для LLM-бонуса (опционально)

Вывести карточку и спросить подтверждение:

```
╔══════════════════════════════════════════════╗
║       DataPipeline — Конфигурация            ║
╠══════════════════════════════════════════════╣
║  Domain:    {domain}
║  Task:      {ml_task}
║  Modality:  {modality}
║  Sources:   {sources или "подобрать автоматически"}
║  LLM-бонус: {"включён" или "выключен"}
╚══════════════════════════════════════════════╝

Запустить пайплайн? [y/n]
```

---

## Что нужно создать

### Напиши `run_pipeline.py`

Полный Prefect-пайплайн с `@flow` и `@task` декораторами:

```python
"""
Единый дата-пайплайн с Human-in-the-Loop.
Использование: python run_pipeline.py --domain "..." --ml_task classification
"""
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

from agents.data_collection_agent import DataCollectionAgent
from agents.data_quality_agent import DataQualityAgent
from agents.annotation_agent import AnnotationAgent
from agents.al_agent import ActiveLearningAgent


@task(name="Шаг 1: Сбор данных", retries=2, retry_delay_seconds=30)
def collect(domain: str, modality: str, config_path: str) -> str:
    agent = DataCollectionAgent(config=config_path)
    df = agent.run(domain=domain, modality=modality)
    date = datetime.now().strftime("%Y%m%d")
    path = f"data/raw/collected_{domain.replace(' ', '_')}_{date}.parquet"
    df.to_parquet(path, index=False)
    return path


@task(name="Шаг 2: Чистка данных", retries=1)
def clean(raw_path: str, domain: str, ml_task: str) -> str:
    df = pd.read_parquet(raw_path)
    agent = DataQualityAgent()
    report = agent.detect_issues(df)
    strategy = _auto_strategy(report)
    df_clean = agent.fix(df, strategy=strategy)
    date = datetime.now().strftime("%Y%m%d")
    path = f"data/clean/cleaned_{domain.replace(' ', '_')}_{date}.parquet"
    df_clean.to_parquet(path, index=False)
    return path


@task(name="Шаг 3: Авторазметка")
def auto_label(clean_path: str, modality: str, ml_task: str, threshold: float = 0.85) -> str:
    df = pd.read_parquet(clean_path)
    agent = AnnotationAgent(modality=modality, confidence_threshold=threshold)
    df_labeled = agent.auto_label(df)
    df_auto, df_review = agent.flag_for_review(df_labeled)
    agent.export_to_labelstudio(df_labeled)
    date = datetime.now().strftime("%Y%m%d")
    path = f"data/annotated/annotated_{date}.parquet"
    df_labeled.to_parquet(path, index=False)
    return path


@task(name="❗ HITL-1: Проверка разметки")
def human_review(annotated_path: str, threshold: float = 0.85) -> str:
    """
    Человек проверяет и правит примеры с низкой уверенностью.
    Ждёт файл exports/review_queue_corrected.csv перед продолжением.
    """
    df = pd.read_parquet(annotated_path)
    low_conf = df[df['confidence'] < threshold].copy()

    # Сохранить очередь для проверки
    Path("exports").mkdir(exist_ok=True)
    review_path = "exports/review_queue.csv"
    low_conf[['text', 'predicted_label', 'confidence', 'source']].to_csv(review_path, index=False)

    print(f"\n{'='*50}")
    print(f"❗ HITL-1: Требуется ручная проверка")
    print(f"   Флагнуто: {len(low_conf)} примеров (confidence < {threshold})")
    print(f"   Файл:     {review_path}")
    print(f"\n   Что делать:")
    print(f"   1. Откройте {review_path}")
    print(f"   2. Исправьте колонку 'predicted_label'")
    print(f"   3. Сохраните как exports/review_queue_corrected.csv")
    print(f"   4. Нажмите Enter для продолжения (или введите 'skip')")
    print(f"{'='*50}\n")

    user_input = input("Путь к исправленному файлу (Enter = exports/review_queue_corrected.csv, 'skip' = пропустить): ").strip()

    if user_input.lower() == 'skip':
        print("⚠️  Ручная проверка пропущена — используется авторазметка")
        return annotated_path

    corrected_path = user_input if user_input else "exports/review_queue_corrected.csv"

    if Path(corrected_path).exists():
        corrected = pd.read_csv(corrected_path)
        df_high = df[df['confidence'] >= threshold].copy()
        corrected['label'] = corrected['predicted_label']
        corrected['needs_review'] = False
        df_merged = pd.concat([df_high, corrected], ignore_index=True)
        date = datetime.now().strftime("%Y%m%d")
        reviewed_path = f"data/annotated/reviewed_{date}.parquet"
        df_merged.to_parquet(reviewed_path, index=False)
        print(f"✅ Принято {len(corrected)} исправлений")
        return reviewed_path
    else:
        print(f"⚠️  Файл {corrected_path} не найден — используется авторазметка")
        return annotated_path


@task(name="Шаг 4: Active Learning отбор")
def select_for_labeling(reviewed_path: str, domain: str) -> str:
    df = pd.read_parquet(reviewed_path)
    agent = ActiveLearningAgent(model='logreg')
    history = agent.run_cycle(
        labeled_df=df,
        pool_df=df.sample(frac=0.3, random_state=42),
        n_iter=5,
        n_per_iter=20,
        strategy='entropy'
    )
    agent.report(history)
    date = datetime.now().strftime("%Y%m%d")
    path = f"data/final/al_selected_{domain.replace(' ', '_')}_{date}.parquet"
    df.to_parquet(path, index=False)
    return path


@task(name="❗ HITL-2: Подтверждение датасета")
def confirm_dataset(final_path: str) -> str:
    """
    Человек просматривает финальный датасет перед обучением модели.
    """
    df = pd.read_parquet(final_path)

    print(f"\n{'='*50}")
    print(f"❗ HITL-2: Проверьте финальный датасет перед обучением")
    print(f"\n   Размер:         {len(df)} строк")
    print(f"   Колонки:        {list(df.columns)}")
    if 'label' in df.columns:
        print(f"   Классы:         {df['label'].value_counts().to_dict()}")
    print(f"\n   Примеры:")
    print(df[['text', 'label']].head(5).to_string(index=False))
    print(f"{'='*50}\n")

    answer = input("Датасет готов к обучению? [y/n]: ").strip().lower()
    if answer != 'y':
        raise RuntimeError("Обучение отменено пользователем. Исправьте датасет и перезапустите с шага 4.")

    # Сохранить финальный датасет с data card
    labeled_path = final_path.replace("data/final/", "data/labeled/")
    Path("data/labeled").mkdir(exist_ok=True)
    df.to_parquet(labeled_path, index=False)
    _write_data_card(df, labeled_path)
    return labeled_path


@task(name="Шаг 5: Обучение модели")
def train(labeled_path: str, ml_task: str) -> dict:
    import json
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    import pickle

    df = pd.read_parquet(labeled_path)
    X = df['text'].fillna('')
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'f1_macro': round(f1_score(y_test, y_pred, average='macro'), 4),
        'f1_weighted': round(f1_score(y_test, y_pred, average='weighted'), 4),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'classification_report': classification_report(y_test, y_pred)
    }

    # Сохранить модель
    Path("models").mkdir(exist_ok=True)
    date = datetime.now().strftime("%Y%m%d")
    model_path = f"models/model_{date}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'vectorizer': vec}, f)

    # Сохранить метрики
    Path("reports").mkdir(exist_ok=True)
    with open("reports/final_metrics.json", 'w') as f:
        json.dump({k: v for k, v in metrics.items() if k != 'classification_report'}, f, indent=2)

    print(f"\n✅ Модель обучена и сохранена: {model_path}")
    print(f"   Accuracy:   {metrics['accuracy']}")
    print(f"   F1 macro:   {metrics['f1_macro']}")

    return metrics


@task(name="Генерация итогового отчёта")
def generate_report(domain: str, ml_task: str, metrics: dict,
                    raw_path: str, clean_path: str, annotated_path: str, final_path: str) -> None:
    import json
    from pathlib import Path

    quality_report = {}
    if Path("reports/quality_report.json").exists():
        with open("reports/quality_report.json") as f:
            quality_report = json.load(f)

    al_report = {}
    if Path("reports/al_report.json").exists():
        with open("reports/al_report.json") as f:
            al_report = json.load(f)

    report = f"""# Pipeline Report: {domain}

## 1. Описание задачи и датасета
- **Задача:** {ml_task}
- **Предметная область:** {domain}
- **Финальный датасет:** `data/labeled/`
- **Размер:** {metrics.get('n_train', 0) + metrics.get('n_test', 0)} примеров
  (train: {metrics.get('n_train')}, test: {metrics.get('n_test')})

## 2. Что делал каждый агент
- **DataCollectionAgent:** собрал данные из 2+ источников → `{raw_path}`
- **DataQualityAgent:** выявил и устранил пропуски, дубли, выбросы → `{clean_path}`
- **AnnotationAgent:** авторазметка zero-shot, флагование low-confidence → `{annotated_path}`
- **ALAgent:** entropy-стратегия отбора, сравнение с random baseline → `{final_path}`
- **ModelTrainer:** TF-IDF + LogReg, train/test split 80/20

## 3. Human-in-the-Loop точки
### HITL-1: Проверка разметки
- Флагнуто примеров с confidence < 0.85: см. `exports/review_queue.csv`
- Исправлено: см. `exports/review_queue_corrected.csv`

### HITL-2: Подтверждение датасета
- Человек подтвердил готовность датасета перед обучением модели

## 4. Метрики качества
### Итоговая модель
| Метрика | Значение |
|---------|----------|
| Accuracy | {metrics.get('accuracy')} |
| F1 macro | {metrics.get('f1_macro')} |
| F1 weighted | {metrics.get('f1_weighted')} |

```
{metrics.get('classification_report', '')}
```

## 5. Ретроспектива
*Заполнить вручную после завершения проекта*

- Что сработало хорошо:
- Что не сработало:
- Что сделал бы иначе:
- Дальнейшие шаги:
"""

    Path("reports").mkdir(exist_ok=True)
    with open("reports/pipeline_report.md", 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n📄 Итоговый отчёт сохранён: reports/pipeline_report.md")


@flow(name="DataPipeline", log_prints=True)
def data_pipeline(domain: str, ml_task: str, modality: str = 'text',
                  config: str = 'config.yaml', confidence_threshold: float = 0.85):

    raw_path = collect(domain, modality, config)
    clean_path = clean(raw_path, domain, ml_task)
    annotated_path = auto_label(clean_path, modality, ml_task, confidence_threshold)
    reviewed_path = human_review(annotated_path, confidence_threshold)
    final_path = select_for_labeling(reviewed_path, domain)
    labeled_path = confirm_dataset(final_path)
    metrics = train(labeled_path, ml_task)
    generate_report(domain, ml_task, metrics, raw_path, clean_path, annotated_path, final_path)

    print(f"\n{'='*50}")
    print(f"🎉 Пайплайн завершён!")
    print(f"   Финальный датасет: data/labeled/")
    print(f"   Модель:            models/")
    print(f"   Отчёт:             reports/pipeline_report.md")
    print(f"{'='*50}\n")


def _auto_strategy(report: dict) -> dict:
    """Автоматически выбирает стратегию чистки по результатам detect_issues."""
    strategy = {'duplicates': 'drop', 'text_quality': 'drop_empty'}
    severity = report.get('summary', {}).get('severity', 'low')
    strategy['missing'] = 'drop' if severity == 'high' else 'median'
    strategy['outliers'] = 'clip_iqr' if severity != 'high' else 'drop'
    strategy['imbalance'] = 'oversample' if report.get('imbalance', {}).get('is_critical') else 'none'
    return strategy


def _write_data_card(df: pd.DataFrame, path: str) -> None:
    """Создаёт data card рядом с финальным датасетом."""
    card_path = path.replace('.parquet', '_data_card.md')
    label_dist = df['label'].value_counts().to_dict() if 'label' in df.columns else {}
    card = f"""# Data Card

| Поле | Значение |
|------|----------|
| Размер | {len(df)} строк |
| Колонки | {', '.join(df.columns)} |
| Распределение классов | {label_dist} |
| Источники | {df['source'].unique().tolist() if 'source' in df.columns else 'н/д'} |
| Дата создания | {datetime.now().strftime('%Y-%m-%d')} |
"""
    with open(card_path, 'w', encoding='utf-8') as f:
        f.write(card)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DataPipeline — сбор и подготовка данных для ML')
    parser.add_argument('--domain', required=True, help='Предметная область')
    parser.add_argument('--ml_task', required=True, choices=['classification', 'sentiment', 'ner', 'generation'])
    parser.add_argument('--modality', default='text', choices=['text', 'audio', 'image'])
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--confidence-threshold', type=float, default=0.85)
    args = parser.parse_args()

    data_pipeline(
        domain=args.domain,
        ml_task=args.ml_task,
        modality=args.modality,
        config=args.config,
        confidence_threshold=args.confidence_threshold
    )
```

---

## Выполнение скилла

При вызове `/data-pipeline` ты должен:

1. Провести стартовый диалог (см. выше)
2. Создать `run_pipeline.py` с кодом выше (подставить реальные параметры)
3. Обновить `config.yaml` подходящими источниками для domain
4. Выполнить шаги 1–5 последовательно, сопровождая каждый прогресс-баннером:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[N/5] {AgentName} — ЗАПУЩЕН
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

5. На HITL-1 и HITL-2 — остановиться, вывести инструкцию, дождаться ответа
6. После Шага 5 — вывести финальный дашборд

---

## Финальный дашборд

```
╔════════════════════════════════════════════════════╗
║           ✅ DataPipeline — ЗАВЕРШЁН               ║
╠════════════════════════════════════════════════════╣
║  Domain:    {domain}
║  Task:      {ml_task}
╠════════════════════════════════════════════════════╣
║  [1/5] Сбор:        {raw_count} строк из {sources_n} источников
║  [2/5] Чистка:      {before} → {after} строк (−{delta})
║  [3/5] Разметка:    {auto_n} авто + {human_n} ручных
║  [4/5] AL-отбор:    {final_n} примеров, экономия {saved}%
║  [5/5] Модель:      Accuracy {accuracy} | F1 {f1}
╠════════════════════════════════════════════════════╣
║  Финальный датасет:  data/labeled/
║  Модель:             models/
║  Отчёт:              reports/pipeline_report.md
╚════════════════════════════════════════════════════╝

🚀 Воспроизведение:
   pip install -r requirements.txt
   python run_pipeline.py --domain "{domain}" --ml_task {ml_task}
```

---

## Обработка ошибок

```
❌ [N/5] {AgentName} — ОШИБКА
   Причина: {error_message}

   Варианты:
   1. "retry"  — повторить этот шаг
   2. "skip"   — пропустить и продолжить
   3. "stop"   — завершить пайплайн
```

---

## Бонус: LLM-агент (Claude API, +3 балла)

Если передан `api_key` — добавить в `run_pipeline.py` вызов Claude API:
- В Шаге 3: `agent.generate_spec()` использует Claude для написания спецификации разметки
- В Шаге 2: `agent.llm_explain()` объясняет найденные проблемы качества
- Использовать `anthropic` SDK, модель `claude-sonnet-4-6`

## Бонус: Streamlit дашборд (+2 балла)

Создать `dashboard.py`:
- Вкладка "HITL Review": загрузить `review_queue.csv`, отобразить примеры, дать возможность менять метки, сохранить `review_queue_corrected.csv`
- Вкладка "Metrics": показать `final_metrics.json` и `learning_curve.png`
- Запуск: `streamlit run dashboard.py`

## Управление пайплайном

| Команда | Действие |
|---------|----------|
| `pause` | Остановить после текущего шага |
| `status` | Показать прогресс и пути к файлам |
| `skip N` | Пропустить шаг N |
| `from N path` | Начать с шага N, взяв данные из path |
| `retry` | Повторить последний шаг |
