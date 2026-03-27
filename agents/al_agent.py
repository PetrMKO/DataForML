"""
ActiveLearningAgent — Шаг 4: Active Learning отбор данных.

Использование:
    agent = ActiveLearningAgent(model='logreg')
    history = agent.run_cycle(labeled_df, pool_df, n_iter=5, n_per_iter=20, strategy='entropy')
    agent.report(history)
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class ActiveLearningAgent:
    """Отбирает наиболее информативные примеры через Active Learning.

    Example:
        agent = ActiveLearningAgent(model='logreg')
        history = agent.run_cycle(labeled_df, pool_df, n_iter=5)
    """

    def __init__(self, model: str = 'logreg', random_state: int = 42):
        """
        Args:
            model: 'logreg' | 'svm' | 'bert'
            random_state: для воспроизводимости
        """
        self.model_type = model
        self.random_state = random_state
        self._model = None
        self._vectorizer = None
        logger.info(f"ActiveLearningAgent: model={model}")

    def _build_model(self):
        """Создаёт новый экземпляр модели."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

        if self.model_type == 'logreg':
            self._model = LogisticRegression(max_iter=1000, C=1.0, random_state=self.random_state)
        elif self.model_type == 'svm':
            self._model = SVC(probability=True, kernel='rbf', random_state=self.random_state)
        elif self.model_type == 'bert':
            try:
                from transformers import pipeline
                self._bert_pipeline = pipeline('text-classification', model='distilbert-base-uncased')
                self._model = None
                return
            except ImportError:
                logger.warning("transformers не установлен, использую logreg")
                self._model = LogisticRegression(max_iter=1000, C=1.0, random_state=self.random_state)
        else:
            raise ValueError(f"Неизвестная модель: {self.model_type}")

    def fit(self, labeled_df: pd.DataFrame):
        """Обучает модель на размеченных данных.

        Args:
            labeled_df: DataFrame с колонками text и label

        Returns:
            обученный объект модели

        Example:
            model = agent.fit(labeled_df)
        """
        self._build_model()
        X = labeled_df['text'].fillna('').astype(str).tolist()
        y = labeled_df['label'].astype(str).tolist()

        X_vec = self._vectorizer.fit_transform(X)
        self._model.fit(X_vec, y)

        train_acc = self._model.score(X_vec, y)
        logger.info(f"fit(): обучено на {len(labeled_df)} примерах, train_acc={train_acc:.4f}")
        return self._model

    def query(self, pool_df: pd.DataFrame, strategy: str, n: int = 20) -> list:
        """Выбирает n наиболее информативных примеров из пула.

        Args:
            pool_df: пул неразмеченных данных
            strategy: 'entropy' | 'margin' | 'random'
            n: количество примеров для запроса

        Returns:
            список индексов из pool_df

        Example:
            indices = agent.query(pool_df, strategy='entropy', n=20)
        """
        n = min(n, len(pool_df))
        if n == 0:
            return []

        if strategy == 'random':
            rng = np.random.default_rng(self.random_state)
            return rng.choice(len(pool_df), n, replace=False).tolist()

        X_pool = pool_df['text'].fillna('').astype(str).tolist()
        X_vec = self._vectorizer.transform(X_pool)
        proba = self._model.predict_proba(X_vec)

        if strategy == 'entropy':
            entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
            return np.argsort(entropy)[-n:][::-1].tolist()
        elif strategy == 'margin':
            sorted_proba = np.sort(proba, axis=1)
            if sorted_proba.shape[1] >= 2:
                margin = sorted_proba[:, -1] - sorted_proba[:, -2]
            else:
                margin = sorted_proba[:, -1]
            return np.argsort(margin)[:n].tolist()
        else:
            raise ValueError(f"Неизвестная стратегия: {strategy}. Используй 'entropy', 'margin' или 'random'")

    def evaluate(self, labeled_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        """Оценивает модель на тестовой выборке.

        Args:
            labeled_df: тренировочные данные (для обучения если нужно)
            test_df: тестовые данные

        Returns:
            словарь с метриками качества

        Example:
            metrics = agent.evaluate(labeled_df, test_df)
        """
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        if len(test_df) == 0:
            return {'accuracy': 0, 'f1_macro': 0, 'f1_weighted': 0, 'n_labeled': len(labeled_df)}

        X_test = test_df['text'].fillna('').astype(str).tolist()
        y_test = test_df['label'].astype(str).tolist()

        X_vec = self._vectorizer.transform(X_test)
        y_pred = self._model.predict(X_vec)

        return {
            'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
            'f1_macro': round(float(f1_score(y_test, y_pred, average='macro', zero_division=0)), 4),
            'f1_weighted': round(float(f1_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
            'n_labeled': len(labeled_df),
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
        }

    def run_cycle(self, labeled_df: pd.DataFrame, pool_df: pd.DataFrame,
                  n_iter: int = 5, n_per_iter: int = 20, strategy: str = 'entropy',
                  test_df: Optional[pd.DataFrame] = None) -> list:
        """Запускает AL-цикл.

        Args:
            labeled_df: начальная размеченная выборка
            pool_df: пул данных для активного запроса
            n_iter: количество итераций
            n_per_iter: примеров добавляется за итерацию
            strategy: стратегия запроса
            test_df: тестовая выборка (если None — берём 20% из labeled_df)

        Returns:
            history — список словарей с метриками по итерациям

        Example:
            history = agent.run_cycle(labeled_df, pool_df, n_iter=5, strategy='entropy')
        """
        from sklearn.model_selection import train_test_split

        # Создать тестовую выборку если не передана
        if test_df is None and len(labeled_df) >= 10:
            try:
                labeled_train, test_df = train_test_split(
                    labeled_df, test_size=0.2, stratify=labeled_df['label'], random_state=self.random_state
                )
                labeled_df = labeled_train
            except Exception:
                labeled_train, test_df = train_test_split(labeled_df, test_size=0.2, random_state=self.random_state)
                labeled_df = labeled_train

        current_labeled = labeled_df.copy()
        current_pool = pool_df.copy()
        history = []

        logger.info(f"run_cycle(): strategy={strategy}, n_iter={n_iter}, n_per_iter={n_per_iter}")
        logger.info(f"  Start: labeled={len(current_labeled)}, pool={len(current_pool)}, test={len(test_df) if test_df is not None else 0}")

        # Итерация 0 — базовая
        self.fit(current_labeled)
        metrics = self.evaluate(current_labeled, test_df) if test_df is not None else {'accuracy': 0, 'f1_macro': 0, 'f1_weighted': 0, 'n_labeled': len(current_labeled)}
        metrics['iteration'] = 0
        metrics['strategy'] = strategy
        history.append(metrics)
        logger.info(f"  Iter 0: n_labeled={len(current_labeled)}, acc={metrics['accuracy']}, f1={metrics['f1_macro']}")

        for i in range(1, n_iter + 1):
            if len(current_pool) == 0:
                logger.warning(f"Пул пуст на итерации {i}, останавливаем цикл")
                break

            indices = self.query(current_pool, strategy=strategy, n=n_per_iter)
            new_samples = current_pool.iloc[indices].copy()

            current_labeled = pd.concat([current_labeled, new_samples], ignore_index=True)
            current_pool = current_pool.drop(current_pool.index[indices]).reset_index(drop=True)

            self.fit(current_labeled)
            metrics = self.evaluate(current_labeled, test_df) if test_df is not None else {'accuracy': 0, 'f1_macro': 0, 'f1_weighted': 0, 'n_labeled': len(current_labeled)}
            metrics['iteration'] = i
            metrics['strategy'] = strategy
            history.append(metrics)
            logger.info(f"  Iter {i}: n_labeled={len(current_labeled)}, acc={metrics['accuracy']}, f1={metrics['f1_macro']}")

        return history

    def report(self, history: list) -> None:
        """Строит learning curve и сохраняет отчёт.

        Args:
            history: список из run_cycle()

        Example:
            agent.report(history)
        """
        import matplotlib.pyplot as plt

        Path('reports').mkdir(exist_ok=True)

        # Сохранить JSON-отчёт
        report_data = {
            'history': [{k: v for k, v in h.items() if k != 'classification_report'} for h in history],
            'best_f1': max(h['f1_macro'] for h in history),
            'final_n_labeled': history[-1]['n_labeled'] if history else 0,
            'strategy': history[0]['strategy'] if history else 'unknown',
            'generated_at': datetime.now().isoformat()
        }
        with open('reports/al_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)

        # Learning curve
        n_labeled = [h['n_labeled'] for h in history]
        f1_scores = [h['f1_macro'] for h in history]
        strategy = history[0]['strategy'] if history else 'entropy'

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(n_labeled, f1_scores, marker='o', linewidth=2, label=f'AL ({strategy})', color='steelblue')

        # Линия на 95% от максимального F1
        max_f1 = max(f1_scores)
        target_f1 = max_f1 * 0.95
        ax.axhline(y=target_f1, linestyle='--', color='orange', alpha=0.7, label=f'95% max F1 ({target_f1:.3f})')

        # Найти точку достижения 95%
        for i, (n, f) in enumerate(zip(n_labeled, f1_scores)):
            if f >= target_f1:
                ax.axvline(x=n, linestyle=':', color='green', alpha=0.7)
                if i > 0:
                    savings = n_labeled[-1] - n
                    savings_pct = round(savings / n_labeled[-1] * 100, 1)
                    ax.annotate(f'Сэкономлено {savings} примеров ({savings_pct}%)',
                                xy=(n, target_f1), xytext=(n + 5, target_f1 - 0.05),
                                fontsize=9, color='green')
                break

        ax.set_xlabel('Количество размеченных примеров')
        ax.set_ylabel('F1 macro')
        ax.set_title(f'Learning Curve — Active Learning vs Random\nDomain: news classification')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('reports/learning_curve.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"report(): learning curve → reports/learning_curve.png")
        logger.info(f"report(): al_report.json сохранён")
        print(f"\n✅ AL-отчёт сохранён: reports/al_report.json")
        print(f"   Learning curve:  reports/learning_curve.png")
        print(f"   Лучший F1:       {report_data['best_f1']:.4f}")
        print(f"   Финальный пул:   {report_data['final_n_labeled']} примеров")

    def llm_suggest_strategy(self, history: list, api_key: str = None,
                             api_type: str = 'anthropic') -> str:
        """Объясняет результаты AL и предлагает следующую стратегию.

        Args:
            history: список из run_cycle()
            api_key: API ключ (если None — rule-based fallback)
            api_type: 'anthropic' | 'openai'

        Returns:
            строка с объяснением и рекомендацией
        """
        if not api_key:
            best = max(history, key=lambda x: x['f1_macro'])
            worst = min(history, key=lambda x: x['f1_macro'])
            gain = best['f1_macro'] - worst['f1_macro']
            return (f"Лучший результат: {best['f1_macro']:.3f} F1 при {best['n_labeled']} примерах. "
                    f"Прирост от AL: +{gain:.3f} F1 по сравнению со стартом.")

        prompt = f"""
        Результаты Active Learning цикла: {json.dumps(history, indent=2)}

        1. Объясни почему выбранная стратегия ({history[0].get('strategy')}) сработала или нет.
        2. Сравни с random baseline если доступно.
        3. Предложи улучшения для следующего цикла.
        """

        if api_type == 'anthropic':
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model='claude-sonnet-4-6',
                max_tokens=512,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response.content[0].text
        elif api_type == 'openai':
            import openai
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response.choices[0].message.content

        raise NotImplementedError("Укажите api_key и api_type")
