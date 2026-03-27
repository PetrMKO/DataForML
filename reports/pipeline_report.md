# Pipeline Report: news classification

## 1. Описание задачи и датасета
- **Задача:** classification
- **Предметная область:** news classification
- **Финальный датасет:** `data/labeled/`
- **Размер:** 4000 примеров
  (train: 3200, test: 800)

## 2. Что делал каждый агент
- **DataCollectionAgent:** загрузил датасет ag_news (AG News) из HuggingFace — 4 класса новостей → `data/raw/collected_news_classification_20260327.parquet`
- **DataQualityAgent:** severity=low, устранил дубли/пропуски/дисбаланс → `data/clean/cleaned_news_classification_20260327.parquet`
- **AnnotationAgent:** zero-shot авторазметка (rule-based fallback), флагование low-confidence → `data/annotated/annotated_20260327.parquet`
- **ALAgent:** entropy-стратегия, 5 итераций, лучший F1=0.8624 → `data/final/al_selected_news_classification_20260327.parquet`
- **ModelTrainer:** TF-IDF (max_features=10000, ngrams=(1,2)) + LogReg (C=1.0), train/test 80/20

## 3. Human-in-the-Loop точки
### HITL-1: Проверка разметки
- Флагнуто примеров с confidence < 0.85: см. `exports/review_queue.csv`
- Исправленный файл: `exports/review_queue_corrected.csv`

### HITL-2: Подтверждение датасета
- Пользователь подтвердил датасет перед обучением

## 4. Метрики качества
### Качество данных (DataQualityAgent)
- Severity: low
- Всего найдено проблем: 0

### Active Learning (ALAgent)
- Лучший F1: 0.8624
- Финальный пул: 1650 примеров
- Стратегия: entropy

### Итоговая модель (ModelTrainer)
| Метрика | Значение |
|---------|----------|
| Accuracy | 0.88 |
| F1 macro | 0.8772 |
| F1 weighted | 0.8785 |

```
              precision    recall  f1-score   support

    Business       0.86      0.76      0.81       192
    Sci/Tech       0.84      0.91      0.88       207
      Sports       0.92      0.97      0.95       207
       World       0.89      0.87      0.88       194

    accuracy                           0.88       800
   macro avg       0.88      0.88      0.88       800
weighted avg       0.88      0.88      0.88       800

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
