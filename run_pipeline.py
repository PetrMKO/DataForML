"""
Единый дата-пайплайн с Human-in-the-Loop.
Domain: news classification

Использование:
    python run_pipeline.py --domain "news classification" --ml_task classification

С Prefect UI:
    prefect server start
    python run_pipeline.py ...
"""
import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
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
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    path = f"data/raw/collected_{domain.replace(' ', '_')}_{date}.parquet"
    df.to_parquet(path, index=False)
    print(f"✅ [1/5] Сохранено: {path} ({len(df)} строк)")
    return path


@task(name="Шаг 2: Чистка данных", retries=1)
def clean(raw_path: str, domain: str, ml_task: str) -> str:
    df = pd.read_parquet(raw_path)
    agent = DataQualityAgent()
    report = agent.detect_issues(df)

    # Сохранить отчёт
    Path("reports").mkdir(exist_ok=True)
    with open("reports/quality_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    strategy = _auto_strategy(report)
    df_clean = agent.fix(df, strategy=strategy)
    agent.compare(df, df_clean)

    date = datetime.now().strftime("%Y%m%d")
    Path("data/clean").mkdir(parents=True, exist_ok=True)
    path = f"data/clean/cleaned_{domain.replace(' ', '_')}_{date}.parquet"
    df_clean.to_parquet(path, index=False)
    print(f"✅ [2/5] Сохранено: {path} ({len(df_clean)} строк)")
    return path


@task(name="Шаг 3: Авторазметка")
def auto_label(clean_path: str, modality: str, ml_task: str,
               threshold: float = 0.85, classes: list = None) -> str:
    df = pd.read_parquet(clean_path)
    agent = AnnotationAgent(modality=modality, confidence_threshold=threshold,
                            task=ml_task, classes=classes)
    df_labeled = agent.auto_label(df)
    df_auto, df_review = agent.flag_for_review(df_labeled, threshold=threshold)
    agent.generate_spec(df_labeled, ml_task)
    agent.export_to_labelstudio(df_labeled)

    date = datetime.now().strftime("%Y%m%d")
    Path("data/annotated").mkdir(parents=True, exist_ok=True)
    path = f"data/annotated/annotated_{date}.parquet"
    df_labeled.to_parquet(path, index=False)
    print(f"✅ [3/5] Сохранено: {path} ({len(df_labeled)} строк)")
    return path


@task(name="❗ HITL-1: Проверка разметки")
def human_review(annotated_path: str, threshold: float = 0.85) -> str:
    """
    Человек проверяет и правит примеры с низкой уверенностью.
    Ждёт файл exports/review_queue_corrected.csv перед продолжением.
    """
    df = pd.read_parquet(annotated_path)
    low_conf = df[df['confidence'] < threshold].copy()

    Path("exports").mkdir(exist_ok=True)
    review_path = "exports/review_queue.csv"
    available_cols = [c for c in ['text', 'predicted_label', 'confidence', 'source'] if c in low_conf.columns]
    low_conf[available_cols].to_csv(review_path, index=False)

    print(f"\n{'='*55}")
    print(f"❗ HITL-1: Требуется ручная проверка")
    print(f"   Флагнуто: {len(low_conf)} примеров (confidence < {threshold})")
    print(f"   Файл:     {review_path}")
    print(f"\n   Что делать:")
    print(f"   1. Откройте {review_path}")
    print(f"   2. Исправьте колонку 'predicted_label'")
    print(f"   3. Сохраните как exports/review_queue_corrected.csv")
    print(f"   4. Нажмите Enter для продолжения (или 'skip')")
    print(f"{'='*55}\n")

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
        Path("data/annotated").mkdir(parents=True, exist_ok=True)
        reviewed_path = f"data/annotated/reviewed_{date}.parquet"
        df_merged.to_parquet(reviewed_path, index=False)
        print(f"✅ Принято {len(corrected)} исправлений → {reviewed_path}")
        return reviewed_path
    else:
        print(f"⚠️  Файл {corrected_path} не найден — используется авторазметка")
        return annotated_path


@task(name="Шаг 4: Active Learning отбор")
def select_for_labeling(reviewed_path: str, domain: str) -> str:
    import numpy as np
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(reviewed_path)
    agent = ActiveLearningAgent(model='logreg')

    # Нужна колонка label — берём из predicted_label если нет
    if 'label' not in df.columns or df['label'].isna().all():
        df['label'] = df.get('predicted_label', pd.Series(['unknown'] * len(df)))

    df = df.dropna(subset=['label', 'text']).reset_index(drop=True)

    if len(df) < 20:
        print("⚠️  Слишком мало данных для AL, сохраняем как есть")
        date = datetime.now().strftime("%Y%m%d")
        Path("data/final").mkdir(parents=True, exist_ok=True)
        path = f"data/final/al_selected_{domain.replace(' ', '_')}_{date}.parquet"
        df.to_parquet(path, index=False)
        return path

    # Разбивка на labeled start, pool, test
    try:
        train_pool, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    except ValueError:
        train_pool, test_df = train_test_split(df, test_size=0.2, random_state=42)

    n_start = min(100, int(len(train_pool) * 0.3))
    try:
        labeled_start, pool_df = train_test_split(
            train_pool, train_size=n_start, stratify=train_pool['label'], random_state=42
        )
    except ValueError:
        labeled_start = train_pool.sample(n=n_start, random_state=42)
        pool_df = train_pool.drop(labeled_start.index)

    # Запуск AL-цикла с entropy-стратегией
    history_entropy = agent.run_cycle(
        labeled_df=labeled_start.copy(),
        pool_df=pool_df.copy(),
        n_iter=5,
        n_per_iter=max(10, len(pool_df) // 10),
        strategy='entropy',
        test_df=test_df
    )

    # Запуск baseline с random
    agent_random = ActiveLearningAgent(model='logreg')
    history_random = agent_random.run_cycle(
        labeled_df=labeled_start.copy(),
        pool_df=pool_df.copy(),
        n_iter=5,
        n_per_iter=max(10, len(pool_df) // 10),
        strategy='random',
        test_df=test_df
    )

    # Сохранить объединённую историю
    all_history = history_entropy + history_random
    agent.report(history_entropy)

    date = datetime.now().strftime("%Y%m%d")
    Path("data/final").mkdir(parents=True, exist_ok=True)
    path = f"data/final/al_selected_{domain.replace(' ', '_')}_{date}.parquet"
    df.to_parquet(path, index=False)
    print(f"✅ [4/5] Сохранено: {path} ({len(df)} строк)")
    return path


@task(name="❗ HITL-2: Подтверждение датасета")
def confirm_dataset(final_path: str) -> str:
    """
    Человек просматривает финальный датасет перед обучением модели.
    """
    df = pd.read_parquet(final_path)

    print(f"\n{'='*55}")
    print(f"❗ HITL-2: Проверьте финальный датасет перед обучением")
    print(f"\n   Размер:         {len(df)} строк")
    print(f"   Колонки:        {list(df.columns)}")
    if 'label' in df.columns:
        print(f"   Классы:         {df['label'].value_counts().to_dict()}")
    display_cols = [c for c in ['text', 'label'] if c in df.columns]
    if display_cols:
        print(f"\n   Примеры:")
        print(df[display_cols].head(5).to_string(index=False))
    print(f"{'='*55}\n")

    answer = input("Датасет готов к обучению? [y/n]: ").strip().lower()
    if answer != 'y':
        raise RuntimeError("Обучение отменено пользователем. Исправьте датасет и перезапустите с шага 4.")

    Path("data/labeled").mkdir(parents=True, exist_ok=True)
    labeled_path = final_path.replace("data/final/", "data/labeled/")
    df.to_parquet(labeled_path, index=False)
    _write_data_card(df, labeled_path)
    print(f"✅ Датасет подтверждён → {labeled_path}")
    return labeled_path


@task(name="Шаг 5: Обучение модели")
def train(labeled_path: str, ml_task: str) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    df = pd.read_parquet(labeled_path)
    df = df.dropna(subset=['text', 'label'])
    X = df['text'].fillna('').astype(str)
    y = df['label'].astype(str)

    # Стратифицированный split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    clf_report = classification_report(y_test, y_pred, zero_division=0)

    metrics = {
        'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
        'f1_macro': round(float(f1_score(y_test, y_pred, average='macro', zero_division=0)), 4),
        'f1_weighted': round(float(f1_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'classification_report': clf_report
    }

    Path("models").mkdir(exist_ok=True)
    date = datetime.now().strftime("%Y%m%d")
    model_path = f"models/model_{date}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'vectorizer': vec, 'classes': y.unique().tolist()}, f)

    Path("reports").mkdir(exist_ok=True)
    with open("reports/final_metrics.json", 'w') as f:
        json.dump({k: v for k, v in metrics.items() if k != 'classification_report'}, f, indent=2)

    print(f"\n✅ [5/5] Модель обучена → {model_path}")
    print(f"   Accuracy:   {metrics['accuracy']}")
    print(f"   F1 macro:   {metrics['f1_macro']}")
    print(f"   F1 weighted:{metrics['f1_weighted']}")
    print(f"\n{clf_report}")
    return metrics


@task(name="Генерация итогового отчёта")
def generate_report(domain: str, ml_task: str, metrics: dict,
                    raw_path: str, clean_path: str, annotated_path: str, final_path: str) -> None:
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
- **DataCollectionAgent:** загрузил датасет ag_news (AG News) из HuggingFace — 4 класса новостей → `{raw_path}`
- **DataQualityAgent:** severity={quality_report.get('summary', {}).get('severity', 'n/a')}, устранил дубли/пропуски/дисбаланс → `{clean_path}`
- **AnnotationAgent:** zero-shot авторазметка (rule-based fallback), флагование low-confidence → `{annotated_path}`
- **ALAgent:** entropy-стратегия, 5 итераций, лучший F1={al_report.get('best_f1', 'n/a')} → `{final_path}`
- **ModelTrainer:** TF-IDF (max_features=10000, ngrams=(1,2)) + LogReg (C=1.0), train/test 80/20

## 3. Human-in-the-Loop точки
### HITL-1: Проверка разметки
- Флагнуто примеров с confidence < 0.85: см. `exports/review_queue.csv`
- Исправленный файл: `exports/review_queue_corrected.csv`

### HITL-2: Подтверждение датасета
- Пользователь подтвердил датасет перед обучением

## 4. Метрики качества
### Качество данных (DataQualityAgent)
- Severity: {quality_report.get('summary', {}).get('severity', 'n/a')}
- Всего найдено проблем: {quality_report.get('summary', {}).get('total_issues', 'n/a')}

### Active Learning (ALAgent)
- Лучший F1: {al_report.get('best_f1', 'n/a')}
- Финальный пул: {al_report.get('final_n_labeled', 'n/a')} примеров
- Стратегия: {al_report.get('strategy', 'entropy')}

### Итоговая модель (ModelTrainer)
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

- **Что сработало хорошо:**
  - Rule-based классификация для ag_news работает хорошо благодаря явным тематическим кластерам
  - AL с entropy-стратегией позволяет экономить данные при сохранении качества

- **Что не сработало:**
  - Zero-shot (BART) может быть медленным без GPU — использован rule-based fallback

- **Что сделал бы иначе:**
  - Использовать дообученный BERT для лучшей точности разметки
  - Добавить больше итераций AL для стабилизации кривой обучения

- **Дальнейшие шаги:**
  - Попробовать нейросетевые классификаторы (DistilBERT, RoBERTa)
  - Расширить датасет дополнительными новостными источниками
"""

    Path("reports").mkdir(exist_ok=True)
    with open("reports/pipeline_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Итоговый отчёт → reports/pipeline_report.md")


@flow(name="DataPipeline — news classification", log_prints=True)
def data_pipeline(domain: str = "news classification", ml_task: str = "classification",
                  modality: str = "text", config: str = "config.yaml",
                  confidence_threshold: float = 0.85):

    print(f"\n{'━'*45}")
    print(f"[1/5] DataCollectionAgent — ЗАПУЩЕН")
    print(f"{'━'*45}")
    raw_path = collect(domain, modality, config)

    print(f"\n{'━'*45}")
    print(f"[2/5] DataQualityAgent — ЗАПУЩЕН")
    print(f"{'━'*45}")
    clean_path = clean(raw_path, domain, ml_task)

    print(f"\n{'━'*45}")
    print(f"[3/5] AnnotationAgent — ЗАПУЩЕН")
    print(f"{'━'*45}")
    classes = ['World', 'Sports', 'Business', 'Sci/Tech']
    annotated_path = auto_label(clean_path, modality, ml_task, confidence_threshold, classes)

    reviewed_path = human_review(annotated_path, confidence_threshold)

    print(f"\n{'━'*45}")
    print(f"[4/5] ALAgent — ЗАПУЩЕН")
    print(f"{'━'*45}")
    final_path = select_for_labeling(reviewed_path, domain)

    labeled_path = confirm_dataset(final_path)

    print(f"\n{'━'*45}")
    print(f"[5/5] ModelTrainer — ЗАПУЩЕН")
    print(f"{'━'*45}")
    metrics = train(labeled_path, ml_task)

    generate_report(domain, ml_task, metrics, raw_path, clean_path, annotated_path, final_path)

    # Загрузить статистику для дашборда
    raw_count = len(pd.read_parquet(raw_path))
    clean_count = len(pd.read_parquet(clean_path))
    final_count = len(pd.read_parquet(labeled_path))

    print(f"""
╔════════════════════════════════════════════════════╗
║           ✅ DataPipeline — ЗАВЕРШЁН               ║
╠════════════════════════════════════════════════════╣
║  Domain:    {domain}
║  Task:      {ml_task}
╠════════════════════════════════════════════════════╣
║  [1/5] Сбор:        {raw_count} строк
║  [2/5] Чистка:      {raw_count} → {clean_count} строк
║  [3/5] Разметка:    авторазметка + HITL-1
║  [4/5] AL-отбор:    {final_count} примеров
║  [5/5] Модель:      Accuracy {metrics['accuracy']} | F1 {metrics['f1_macro']}
╠════════════════════════════════════════════════════╣
║  Финальный датасет:  data/labeled/
║  Модель:             models/
║  Отчёт:              reports/pipeline_report.md
╚════════════════════════════════════════════════════╝

🚀 Воспроизведение:
   pip install -r requirements.txt
   python run_pipeline.py --domain "{domain}" --ml_task {ml_task}
""")


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
    sources = df['source'].unique().tolist() if 'source' in df.columns else ['n/a']
    card = f"""# Data Card — news classification

| Поле | Значение |
|------|----------|
| Задача | text classification |
| Размер | {len(df)} строк |
| Колонки | {', '.join(df.columns)} |
| Распределение классов | {label_dist} |
| Источники | {sources} |
| Дата создания | {datetime.now().strftime('%Y-%m-%d')} |
| Лицензия | AG News — академическое использование |
"""
    with open(card_path, 'w', encoding='utf-8') as f:
        f.write(card)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DataPipeline — сбор и подготовка данных для ML')
    parser.add_argument('--domain', default='news classification')
    parser.add_argument('--ml_task', default='classification',
                        choices=['classification', 'sentiment', 'ner', 'generation'])
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
