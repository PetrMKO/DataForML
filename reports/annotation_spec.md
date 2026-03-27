# Annotation Specification: news classification

## 1. Описание задачи
Задача: мультиклассовая классификация новостных статей.
Цель: обучить ML-модель автоматически категоризировать новости по 4 классам.

## 2. Классы и определения

| Класс | Определение | Ключевые признаки |
|-------|-------------|-------------------|
| World | Новость категории World | Ключевые слова категории |
| Business | Новость категории Business | Ключевые слова категории |
| Sports | Новость категории Sports | Ключевые слова категории |
| Sci/Tech | Новость категории Sci/Tech | Ключевые слова категории |

## 3. Примеры на каждый класс (минимум 3)

### World
- "BBC set for major shake-up, claims newspaper London - The British Broadcasting Corporation, the worl..." → World
- "Flying the Sun to Safety When the Genesis capsule comes back to Earth with its samples of the sun, h..." → World
- "US genocide charge is Bush election ploy - Sudan FM (AFP) AFP - Sudan's foreign minister rejected US..." → World

### Business
- "Marsh averts cash crunch Embattled insurance broker #39;s banks agree to waive clause that may have ..." → Business
- "Construction Spending Hits All-Time High WASHINGTON - Construction spending surged in August to the ..." → Business
- "MLB Notebook: Johnson trade still in the works The New York Yankees, Arizona and Los Angeles spent y..." → Business

### Sports
- "Inter Milan seeks redemption win against Juventus It is early in the season for a decisive match, ye..." → Sports
- "HP Revises Cluster Plans HP (Quote, Chart) is dropping its efforts to port some Tru64 Unix products ..." → Sports
- "O'Brien Sues OSU for  #36;3.4 Million (AP) AP - Former Ohio State basketball coach Jim O'Brien sued ..." → Sports

### Sci/Tech
- "Medical Experts Fear Charley's Aftermath PUNTA GORDA, Fla. - Until the electricity hums again and th..." → Sci/Tech
- "Wireless to Drive Internet Growth, Tech Leaders Say Wireless services will lead the next growth phas..." → Sci/Tech
- "Why The Open-Source Model Can Work In India An Indian Institute of Technology professor--and open-so..." → Sci/Tech


## 4. Граничные случаи
- Случай 1: Статья о экономических последствиях войны — World или Business?
  Правило: если акцент на геополитике → World, если на рынках → Business
- Случай 2: Спортивный контракт на крупную сумму — Sports или Business?
  Правило: если акцент на спортивных результатах → Sports

## 5. Что НЕ размечать
- Рекламные материалы
- Технические заголовки без контентной ценности
- Дубликаты и варианты одной новости

## 6. Метрики качества
- Ожидаемое Agreement: > 80%
- Cohen's κ: > 0.6 считается хорошим

## 7. Инструкция по LabelStudio
1. Откройте LabelStudio (http://localhost:8080)
2. Создайте новый проект: Text Classification
3. Импортируйте `exports/labelstudio_import.json`
4. Начните разметку с примеров из `exports/labelstudio_review.json` (флагнутые)
5. После разметки экспортируйте JSON и сохраните как `exports/review_queue_corrected.csv`
