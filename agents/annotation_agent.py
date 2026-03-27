"""
AnnotationAgent — Шаг 3: Авторазметка данных.

Использование:
    agent = AnnotationAgent(modality='text', confidence_threshold=0.85)
    df_labeled = agent.auto_label(df)
    df_auto, df_review = agent.flag_for_review(df_labeled)
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class AnnotationAgent:
    """Авторазметчик данных с поддержкой text/audio/image и HITL.

    Example:
        agent = AnnotationAgent(modality='text', confidence_threshold=0.85)
        df_labeled = agent.auto_label(df)
    """

    def __init__(self, modality: str = 'text', confidence_threshold: float = 0.85,
                 task: str = 'classification', classes: Optional[list] = None):
        """
        Args:
            modality: 'text' | 'audio' | 'image'
            confidence_threshold: порог уверенности для HITL флага
            task: тип задачи ('classification', 'sentiment', 'ner')
            classes: список классов (автоопределение если None)
        """
        self.modality = modality
        self.threshold = confidence_threshold
        self.task = task
        self.classes = classes
        self._classifier = None
        self._nlp = None
        logger.info(f"AnnotationAgent: modality={modality}, threshold={confidence_threshold}, task={task}")

    def _get_classifier(self, classes: list):
        """Ленивая загрузка zero-shot классификатора."""
        if self._classifier is None:
            try:
                from transformers import pipeline
                logger.info("Загрузка zero-shot классификатора facebook/bart-large-mnli...")
                self._classifier = pipeline(
                    'zero-shot-classification',
                    model='facebook/bart-large-mnli',
                    device=-1  # CPU
                )
                logger.info("Классификатор загружен")
            except ImportError:
                raise ImportError("Установите transformers и torch: pip install transformers torch")
        return self._classifier

    def _get_nlp(self):
        """Ленивая загрузка spaCy модели."""
        if self._nlp is None:
            try:
                import spacy
                try:
                    self._nlp = spacy.load('en_core_web_sm')
                except OSError:
                    logger.warning("Модель en_core_web_sm не найдена, использую rule-based fallback")
                    self._nlp = None
            except ImportError:
                raise ImportError("Установите spacy: pip install spacy && python -m spacy download en_core_web_sm")
        return self._nlp

    def _rule_based_classify(self, text: str, classes: list) -> tuple:
        """Простая rule-based классификация новостей без ML."""
        text_lower = text.lower()
        scores = {}

        keywords = {
            'World': ['world', 'international', 'global', 'country', 'government', 'war', 'peace', 'military',
                      'president', 'minister', 'nation', 'foreign', 'diplomat', 'un ', 'nato', 'crisis'],
            'Sports': ['sport', 'game', 'team', 'player', 'match', 'tournament', 'championship', 'league',
                       'football', 'soccer', 'basketball', 'tennis', 'baseball', 'olympic', 'coach', 'win', 'score'],
            'Business': ['business', 'market', 'stock', 'economy', 'company', 'trade', 'financial', 'bank',
                         'investment', 'profit', 'revenue', 'ceo', 'corporate', 'merger', 'acquisition', 'shares'],
            'Sci/Tech': ['technology', 'science', 'research', 'computer', 'software', 'internet', 'data',
                         'artificial intelligence', 'robot', 'space', 'nasa', 'medical', 'drug', 'climate', 'ai ']
        }

        for cls in classes:
            kws = keywords.get(cls, [])
            score = sum(1 for kw in kws if kw in text_lower)
            scores[cls] = score

        total = sum(scores.values())
        if total == 0:
            # равномерное распределение
            probs = {cls: 1.0 / len(classes) for cls in classes}
        else:
            probs = {cls: scores[cls] / total for cls in classes}

        best_class = max(probs, key=probs.get)
        confidence = probs[best_class]
        # Нормализуем confidence чтобы было в диапазоне [0.5, 0.95]
        confidence = 0.5 + confidence * 0.45
        return best_class, round(confidence, 4)

    def auto_label(self, df: pd.DataFrame, modality: str = None) -> pd.DataFrame:
        """Автоматически размечает DataFrame.

        Args:
            df: входной DataFrame с колонкой 'text'
            modality: переопределить модальность из конструктора

        Returns:
            DataFrame с добавленными колонками: predicted_label, confidence, model_used, needs_review

        Example:
            df_labeled = agent.auto_label(df)
        """
        from tqdm import tqdm

        modality = modality or self.modality
        result = df.copy()

        # Определить классы
        classes = self.classes
        if classes is None and 'label' in df.columns:
            classes = df['label'].dropna().unique().tolist()
        if not classes:
            classes = ['World', 'Sports', 'Business', 'Sci/Tech']
        logger.info(f"Классы для разметки: {classes}")

        predicted_labels = []
        confidences = []
        models_used = []

        if modality == 'text':
            task_lower = self.task.lower()

            if 'ner' in task_lower:
                # NER через spaCy
                nlp = self._get_nlp()
                for _, row in tqdm(df.iterrows(), total=len(df), desc="NER разметка"):
                    text = str(row.get('text', ''))
                    if nlp:
                        doc = nlp(text)
                        entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
                        predicted_labels.append(json.dumps(entities, ensure_ascii=False))
                        confidences.append(0.9)
                        models_used.append('spacy/en_core_web_sm')
                    else:
                        predicted_labels.append('[]')
                        confidences.append(0.5)
                        models_used.append('rule_based_ner')
            else:
                # Classification / Sentiment через zero-shot или rule-based
                use_zero_shot = True
                try:
                    classifier = self._get_classifier(classes)
                except (ImportError, Exception) as e:
                    logger.warning(f"Zero-shot классификатор недоступен: {e}. Использую rule-based fallback")
                    use_zero_shot = False

                for _, row in tqdm(df.iterrows(), total=len(df), desc="Авторазметка"):
                    text = str(row.get('text', ''))[:512]  # BART ограничение
                    if not text.strip():
                        predicted_labels.append(classes[0])
                        confidences.append(0.5)
                        models_used.append('fallback')
                        continue

                    if use_zero_shot:
                        try:
                            out = classifier(text, candidate_labels=classes)
                            predicted_labels.append(out['labels'][0])
                            confidences.append(round(float(out['scores'][0]), 4))
                            models_used.append('facebook/bart-large-mnli')
                        except Exception as e:
                            label, conf = self._rule_based_classify(text, classes)
                            predicted_labels.append(label)
                            confidences.append(conf)
                            models_used.append('rule_based')
                    else:
                        label, conf = self._rule_based_classify(text, classes)
                        predicted_labels.append(label)
                        confidences.append(conf)
                        models_used.append('rule_based')

        elif modality == 'audio':
            try:
                import whisper
                model = whisper.load_model('base')
                for _, row in tqdm(df.iterrows(), total=len(df), desc="Whisper транскрипция"):
                    audio_path = row.get('audio_path') or row.get('audio', '')
                    try:
                        result_w = model.transcribe(str(audio_path))
                        text = result_w['text']
                        no_speech_prob = result_w.get('segments', [{}])[0].get('no_speech_prob', 0.1)
                        conf = 1.0 - no_speech_prob
                        predicted_labels.append(text[:100])
                        confidences.append(round(conf, 4))
                        models_used.append('openai/whisper-base')
                    except Exception as e:
                        predicted_labels.append('')
                        confidences.append(0.0)
                        models_used.append('whisper_error')
            except ImportError:
                raise ImportError("Установите openai-whisper: pip install openai-whisper")

        elif modality == 'image':
            try:
                from ultralytics import YOLO
                model = YOLO('yolov8n.pt')
                for _, row in tqdm(df.iterrows(), total=len(df), desc="YOLO детекция"):
                    img_path = row.get('image_path') or row.get('image', '')
                    try:
                        results = model(str(img_path))
                        detections = results[0]
                        if len(detections.boxes) > 0:
                            cls_names = [detections.names[int(c)] for c in detections.boxes.cls]
                            confs = detections.boxes.conf.tolist()
                            predicted_labels.append(', '.join(cls_names))
                            confidences.append(round(max(confs), 4))
                        else:
                            predicted_labels.append('no_detection')
                            confidences.append(0.1)
                        models_used.append('yolov8n')
                    except Exception:
                        predicted_labels.append('')
                        confidences.append(0.0)
                        models_used.append('yolo_error')
            except ImportError:
                raise ImportError("Установите ultralytics: pip install ultralytics")

        result['predicted_label'] = predicted_labels
        result['confidence'] = confidences
        result['model_used'] = models_used
        result['needs_review'] = [c < self.threshold for c in confidences]

        # Если нет финальной метки label — используем predicted_label
        if 'label' not in result.columns:
            result['label'] = result['predicted_label']
        else:
            # Для строк где label уже есть — оставляем, иначе берём predicted
            result['label'] = result['label'].fillna(result['predicted_label'])

        n_review = result['needs_review'].sum()
        logger.info(f"auto_label(): {len(result)} примеров, флагнуто {n_review} ({n_review/len(result)*100:.1f}%) для HITL")
        return result

    def flag_for_review(self, df: pd.DataFrame, threshold: float = None) -> tuple:
        """Разделяет датасет на автоматически размеченные и требующие проверки.

        Args:
            df: DataFrame после auto_label()
            threshold: порог уверенности (по умолчанию из конструктора)

        Returns:
            (df_auto, df_review) — кортеж DataFrame

        Example:
            df_auto, df_review = agent.flag_for_review(df_labeled)
        """
        threshold = threshold or self.threshold
        df_auto = df[df['confidence'] >= threshold].copy()
        df_review = df[df['confidence'] < threshold].copy()

        Path('exports').mkdir(exist_ok=True)
        review_cols = ['text', 'predicted_label', 'confidence', 'source']
        available_cols = [c for c in review_cols if c in df_review.columns]
        df_review[available_cols].to_csv('exports/review_queue.csv', index=False)

        logger.info(f"flag_for_review(): {len(df_auto)} авто + {len(df_review)} на проверку")
        print(f"\nФлагнуто {len(df_review)} примеров ({len(df_review)/len(df)*100:.1f}%) для ручной проверки")

        if len(df_review) > 0:
            print("\nТоп-10 примеров с наименьшей уверенностью:")
            top10 = df_review.nsmallest(min(10, len(df_review)), 'confidence')[available_cols]
            print(top10.to_string(index=False))

        return df_auto, df_review

    def generate_spec(self, df: pd.DataFrame, task: str) -> str:
        """Генерирует спецификацию разметки в Markdown.

        Args:
            df: размеченный DataFrame
            task: тип задачи

        Returns:
            Строка с Markdown-содержимым спецификации

        Example:
            spec = agent.generate_spec(df, 'news classification')
        """
        classes = df['predicted_label'].unique().tolist() if 'predicted_label' in df.columns else []

        class_rows = []
        for cls in classes:
            class_rows.append(f"| {cls} | Новость категории {cls} | Ключевые слова категории |")

        examples_section = ""
        for cls in classes:
            cls_df = df[df.get('predicted_label', df.get('label', pd.Series())) == cls] if 'predicted_label' in df.columns else pd.DataFrame()
            if len(cls_df) >= 3:
                examples_section += f"\n### {cls}\n"
                for _, row in cls_df.nlargest(3, 'confidence').iterrows():
                    text_preview = str(row.get('text', ''))[:100]
                    examples_section += f"- \"{text_preview}...\" → {cls}\n"

        spec = f"""# Annotation Specification: {task}

## 1. Описание задачи
Задача: мультиклассовая классификация новостных статей.
Цель: обучить ML-модель автоматически категоризировать новости по 4 классам.

## 2. Классы и определения

| Класс | Определение | Ключевые признаки |
|-------|-------------|-------------------|
{chr(10).join(class_rows)}

## 3. Примеры на каждый класс (минимум 3)
{examples_section}

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
"""

        Path('reports').mkdir(exist_ok=True)
        with open('reports/annotation_spec.md', 'w', encoding='utf-8') as f:
            f.write(spec)

        logger.info("generate_spec(): сохранено reports/annotation_spec.md")
        return spec

    def check_quality(self, df_labeled: pd.DataFrame) -> dict:
        """Оценивает качество авторазметки.

        Args:
            df_labeled: DataFrame с predicted_label и опционально human_label

        Returns:
            QualityMetrics словарь

        Example:
            metrics = agent.check_quality(df_labeled)
        """
        has_human = 'human_label' in df_labeled.columns and df_labeled['human_label'].notna().any()
        confidences = df_labeled['confidence'].tolist() if 'confidence' in df_labeled.columns else []
        below = sum(1 for c in confidences if c < self.threshold)

        metrics = {
            'cohen_kappa': None,
            'percent_agreement': None,
            'label_distribution': {},
            'confidence_stats': {
                'mean': round(float(sum(confidences) / len(confidences)), 4) if confidences else 0,
                'median': round(float(sorted(confidences)[len(confidences) // 2]), 4) if confidences else 0,
                'below_threshold': below,
                'below_threshold_pct': round(below / len(confidences) * 100, 2) if confidences else 0
            },
            'needs_review_count': int(df_labeled.get('needs_review', pd.Series()).sum()) if 'needs_review' in df_labeled.columns else below,
            'has_human_labels': has_human,
            'verdict': ''
        }

        if 'predicted_label' in df_labeled.columns:
            metrics['label_distribution'] = {
                str(k): round(float(v), 4)
                for k, v in df_labeled['predicted_label'].value_counts(normalize=True).items()
            }

        if has_human:
            try:
                from sklearn.metrics import cohen_kappa_score
                common = df_labeled.dropna(subset=['human_label', 'predicted_label'])
                kappa = cohen_kappa_score(common['human_label'], common['predicted_label'])
                agreement = (common['human_label'] == common['predicted_label']).mean()
                metrics['cohen_kappa'] = round(float(kappa), 4)
                metrics['percent_agreement'] = round(float(agreement * 100), 2)

                if kappa > 0.6 and metrics['confidence_stats']['below_threshold_pct'] < 15:
                    metrics['verdict'] = 'good — высокое согласие и уверенность'
                elif kappa >= 0.4:
                    metrics['verdict'] = 'acceptable — умеренное согласие'
                else:
                    metrics['verdict'] = 'poor — низкое согласие, рекомендуется пересмотр стратегии'
            except ImportError:
                pass
        else:
            pct = metrics['confidence_stats']['below_threshold_pct']
            if pct < 10:
                metrics['verdict'] = 'good — низкий процент флагнутых примеров'
            elif pct < 25:
                metrics['verdict'] = 'acceptable — умеренное количество на проверку'
            else:
                metrics['verdict'] = 'poor — много примеров с низкой уверенностью'

        logger.info(f"check_quality(): verdict={metrics['verdict']}")
        return metrics

    def export_to_labelstudio(self, df: pd.DataFrame) -> dict:
        """Экспортирует данные в формат LabelStudio JSON.

        Args:
            df: размеченный DataFrame

        Returns:
            словарь с путями к экспортированным файлам

        Example:
            paths = agent.export_to_labelstudio(df_labeled)
        """
        Path('exports').mkdir(exist_ok=True)

        def make_item(idx, row, needs_review=False):
            item = {
                "id": int(idx) + 1,
                "data": {
                    "text": str(row.get('text', '')),
                    "meta": {
                        "source": str(row.get('source', '')),
                        "collected_at": str(row.get('collected_at', '')),
                        "predicted_label": str(row.get('predicted_label', '')),
                        "confidence": float(row.get('confidence', 0))
                    }
                },
                "annotations": [],
                "predictions": [
                    {
                        "model_version": "auto_label_v1",
                        "result": [
                            {
                                "type": "choices",
                                "value": {"choices": [str(row.get('predicted_label', ''))]},
                                "score": float(row.get('confidence', 0))
                            }
                        ]
                    }
                ]
            }
            if needs_review:
                item["data"]["meta"]["flag"] = "needs_review"
            return item

        # Флагнутые идут первыми
        review_df = df[df['needs_review'] == True] if 'needs_review' in df.columns else pd.DataFrame()
        auto_df = df[df['needs_review'] == False] if 'needs_review' in df.columns else df

        all_items = []
        for idx, row in review_df.iterrows():
            all_items.append(make_item(idx, row, needs_review=True))
        for idx, row in auto_df.iterrows():
            all_items.append(make_item(idx, row, needs_review=False))

        with open('exports/labelstudio_import.json', 'w', encoding='utf-8') as f:
            json.dump(all_items, f, ensure_ascii=False, indent=2)

        # Только флагнутые
        review_items = [item for item in all_items if item['data']['meta'].get('flag') == 'needs_review']
        with open('exports/labelstudio_review.json', 'w', encoding='utf-8') as f:
            json.dump(review_items, f, ensure_ascii=False, indent=2)

        logger.info(f"export_to_labelstudio(): {len(all_items)} записей → exports/labelstudio_import.json")
        logger.info(f"  Review (HITL): {len(review_items)} записей → exports/labelstudio_review.json")

        return {
            'import_path': 'exports/labelstudio_import.json',
            'review_path': 'exports/labelstudio_review.json',
            'total': len(all_items),
            'needs_review': len(review_items)
        }
