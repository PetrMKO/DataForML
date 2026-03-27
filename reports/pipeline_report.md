# Pipeline Report: news categorization

## 1. Описание задачи и датасета
- **Задача:** classification
- **Предметная область:** news categorization
- **Финальный датасет:** `data/labeled/`
- **Размер:** 4000 примеров (train: 3200, test: 800)
- **Классы:** World, Sports, Business, Sci/Tech (4-class AG News)

## 2. Что делал каждый агент
- **DataCollectionAgent:** загрузил ag_news из HuggingFace (train 3000 + test 1000) → `data/raw/collected_news_categorization_20260327.parquet`
- **DataQualityAgent:** severity=low, 0 проблем, 0 дублей → `data/clean/cleaned_news_categorization_20260327.parquet`
- **AnnotationAgent:** rule-based разметка (zero-shot недоступен), флагнуто 1984 (49.6%) low-confidence → `data/annotated/annotated_20260327.parquet`
- **ALAgent:** entropy-стратегия, 5 итераций, лучший F1=0.8651 (vs random baseline 0.8521) → `data/final/al_selected_news_categorization_20260327.parquet`
- **ModelTrainer:** TF-IDF (max_features=10000, ngrams=(1,2)) + LogReg (C=1.0), train/test 80/20

## 3. Human-in-the-Loop точки
### HITL-1: Проверка разметки
- Флагнуто 1984 примеров с confidence < 0.85
- Файл: `exports/review_queue.csv`
- Статус: **пропущено пользователем** (используется авторазметка)

### HITL-2: Подтверждение датасета
- Пользователь подтвердил датасет (4000 строк, 4 класса) перед обучением

## 4. Метрики качества
### Качество данных (DataQualityAgent)
- Severity: low
- Всего найдено проблем: 0

### Active Learning (ALAgent)
- Лучший F1 (entropy): 0.8651
- Лучший F1 (random baseline): 0.8521
- Финальный пул: 1700 примеров
- Стратегия: entropy > random на +1.3% F1

### Итоговая модель (ModelTrainer)
| Метрика | Значение |
|---------|----------|
| Accuracy | 0.88 |
| F1 macro | 0.8772 |
| F1 weighted | 0.8785 |



## 5. Ретроспектива

- **Что сработало хорошо:**
  - AG News — хорошо структурированный датасет, Sports легко отделяется (F1=0.95)
  - AL с entropy-стратегией превзошёл random baseline на 1.3% F1

- **Что не сработало:**
  - Rule-based fallback флагнул 49.6% данных как low-confidence (нет GPU для zero-shot BART)
  - Business имеет самый низкий F1=0.81 — пересекается с World и Sci/Tech

- **Что сделал бы иначе:**
  - Установить transformers/torch для zero-shot авторазметки вместо rule-based
  - Использовать дообученный DistilBERT для лучшей точности (ожидаемый F1 ~0.93+)

- **Дальнейшие шаги:**
  - Провести HITL-1 проверку для 1984 примеров с низкой уверенностью
  - Попробовать нейросетевые классификаторы (DistilBERT, RoBERTa)
  - Добавить cross-validation для более надёжной оценки метрик
