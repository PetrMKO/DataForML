"""
DataQualityAgent — Шаг 2: Чистка и валидация данных.

Использование:
    agent = DataQualityAgent()
    report = agent.detect_issues(df)
    df_clean = agent.fix(df, strategy={'duplicates': 'drop', 'missing': 'median'})
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


class DataQualityAgent:
    """Выявляет и устраняет проблемы качества данных.

    Example:
        agent = DataQualityAgent()
        report = agent.detect_issues(df)
        df_clean = agent.fix(df, strategy)
    """

    def detect_issues(self, df: pd.DataFrame) -> dict:
        """Анализирует DataFrame и возвращает QualityReport со всеми найденными проблемами.

        Args:
            df: входной DataFrame

        Returns:
            QualityReport словарь с секциями: missing, duplicates, outliers, imbalance, text_quality, summary

        Example:
            report = agent.detect_issues(df)
            print(report['summary']['severity'])
        """
        if len(df) == 0:
            raise ValueError("Пустой DataFrame — невозможно провести анализ качества")

        report = {}

        # Missing values
        missing = {}
        for col in df.columns:
            n_missing = df[col].isna().sum() + (df[col] == '').sum() if df[col].dtype == object else df[col].isna().sum()
            if n_missing > 0:
                missing[col] = {
                    'count': int(n_missing),
                    'pct': round(float(n_missing / len(df) * 100), 2)
                }
        report['missing'] = missing

        # Duplicates
        dup_mask = df.duplicated(keep='first')
        n_dups = int(dup_mask.sum())
        report['duplicates'] = {
            'count': n_dups,
            'pct': round(float(n_dups / len(df) * 100), 2),
            'examples': df[dup_mask].head(3).to_dict(orient='records') if n_dups > 0 else []
        }

        # Outliers (только для числовых колонок)
        outliers = {}
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            logger.warning("Нет числовых колонок — проверка выбросов пропущена")
        for col in num_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            out_mask = (df[col] < lower) | (df[col] > upper)
            n_out = int(out_mask.sum())
            if n_out > 0:
                outliers[col] = {
                    'method': 'IQR',
                    'count': n_out,
                    'bounds': {'lower': float(lower), 'upper': float(upper)},
                    'examples': df[out_mask][col].head(5).tolist()
                }
        report['outliers'] = outliers

        # Class imbalance
        report['imbalance'] = {}
        if 'label' in df.columns and df['label'].notna().any():
            vc = df['label'].value_counts(normalize=True)
            ratio = float(vc.iloc[0] / vc.iloc[-1]) if len(vc) > 1 else 1.0
            report['imbalance'] = {
                'label_col': 'label',
                'distribution': {str(k): round(float(v), 4) for k, v in vc.items()},
                'imbalance_ratio': round(ratio, 2),
                'is_critical': ratio > 3
            }

        # Text quality
        report['text_quality'] = {}
        if 'text' in df.columns:
            texts = df['text'].astype(str)
            report['text_quality'] = {
                'empty_strings': int((texts.str.strip() == '').sum()),
                'very_short': int((texts.str.len() < 10).sum()),
                'very_long': int((texts.str.len() > 10000).sum()),
                'encoding_issues': int(texts.apply(lambda x: any(ord(c) > 65535 for c in x)).sum())
            }

        # Summary
        max_missing_pct = max((v['pct'] for v in missing.values()), default=0)
        dup_pct = report['duplicates']['pct']
        imb_ratio = report['imbalance'].get('imbalance_ratio', 1)

        if max_missing_pct > 20 or dup_pct > 10 or imb_ratio > 10:
            severity = 'high'
        elif max_missing_pct > 5 or dup_pct > 3 or imb_ratio > 3:
            severity = 'medium'
        else:
            severity = 'low'

        total_issues = (
            sum(v['count'] for v in missing.values()) +
            n_dups +
            sum(v['count'] for v in outliers.values()) +
            report['text_quality'].get('empty_strings', 0)
        )

        report['summary'] = {
            'total_rows': len(df),
            'total_issues': int(total_issues),
            'severity': severity
        }

        logger.info(f"detect_issues(): severity={severity}, total_issues={total_issues}, rows={len(df)}")
        return report

    def fix(self, df: pd.DataFrame, strategy: dict) -> pd.DataFrame:
        """Применяет стратегию чистки к DataFrame.

        Args:
            df: входной DataFrame
            strategy: словарь стратегий по типу проблемы:
                - missing: 'median'|'mean'|'mode'|'drop'|'fill_empty'
                - duplicates: 'drop'|'keep_first'|'keep_last'
                - outliers: 'clip_iqr'|'drop'|'none'
                - imbalance: 'oversample'|'undersample'|'none'
                - text_quality: 'drop_empty'|'truncate'|'none'

        Returns:
            Очищенный DataFrame с колонкой _quality_flags

        Example:
            df_clean = agent.fix(df, {'duplicates': 'drop', 'missing': 'median', 'outliers': 'clip_iqr'})
        """
        result = df.copy()
        result['_quality_flags'] = [[] for _ in range(len(result))]

        def add_flag(mask, flag):
            for idx in result[mask].index:
                result.at[idx, '_quality_flags'] = result.at[idx, '_quality_flags'] + [flag]

        # Text quality
        tq = strategy.get('text_quality', 'drop_empty')
        if tq == 'drop_empty' and 'text' in result.columns:
            mask = result['text'].astype(str).str.strip() == ''
            n = mask.sum()
            result = result[~mask].reset_index(drop=True)
            logger.info(f"fix(text_quality=drop_empty): удалено {n} пустых строк")
        elif tq == 'truncate' and 'text' in result.columns:
            mask = result['text'].astype(str).str.len() > 10000
            n = mask.sum()
            result.loc[mask, 'text'] = result.loc[mask, 'text'].astype(str).str[:10000]
            logger.info(f"fix(text_quality=truncate): обрезано {n} длинных текстов")

        # Duplicates
        dup = strategy.get('duplicates', 'drop')
        if dup in ('drop', 'keep_first'):
            before = len(result)
            result = result.drop_duplicates(subset='text', keep='first').reset_index(drop=True)
            logger.info(f"fix(duplicates=drop): удалено {before - len(result)} дублей")
        elif dup == 'keep_last':
            before = len(result)
            result = result.drop_duplicates(subset='text', keep='last').reset_index(drop=True)
            logger.info(f"fix(duplicates=keep_last): удалено {before - len(result)} дублей")

        # Missing values
        miss = strategy.get('missing', 'median')
        for col in result.columns:
            if col in ('_quality_flags',):
                continue
            n_missing = result[col].isna().sum()
            if n_missing == 0:
                continue
            if miss == 'drop':
                result = result.dropna(subset=[col]).reset_index(drop=True)
                logger.info(f"fix(missing=drop): удалено {n_missing} строк с NaN в {col}")
            elif miss in ('median', 'mean') and pd.api.types.is_numeric_dtype(result[col]):
                val = result[col].median() if miss == 'median' else result[col].mean()
                result[col] = result[col].fillna(val)
                logger.info(f"fix(missing={miss}): заполнено {n_missing} NaN в {col} → {val:.3f}")
            elif miss == 'mode':
                val = result[col].mode().iloc[0] if len(result[col].mode()) > 0 else ''
                result[col] = result[col].fillna(val)
                logger.info(f"fix(missing=mode): заполнено {n_missing} NaN в {col}")
            elif miss == 'fill_empty':
                result[col] = result[col].fillna('[MISSING]')
                logger.info(f"fix(missing=fill_empty): заполнено {n_missing} NaN в {col}")

        # Outliers
        out = strategy.get('outliers', 'clip_iqr')
        num_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            if col in ('_quality_flags',):
                continue
            q1 = result[col].quantile(0.25)
            q3 = result[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            out_mask = (result[col] < lower) | (result[col] > upper)
            n_out = out_mask.sum()
            if n_out == 0:
                continue
            if out == 'clip_iqr':
                result[col] = result[col].clip(lower, upper)
                logger.info(f"fix(outliers=clip_iqr): обрезано {n_out} выбросов в {col}")
            elif out == 'drop':
                result = result[~out_mask].reset_index(drop=True)
                logger.info(f"fix(outliers=drop): удалено {n_out} выбросов в {col}")

        # Imbalance
        imb = strategy.get('imbalance', 'none')
        if imb != 'none' and 'label' in result.columns and result['label'].notna().any():
            try:
                X = result.drop(columns=['label', '_quality_flags'])
                y = result['label']
                if imb == 'oversample':
                    from imblearn.over_sampling import RandomOverSampler
                    ros = RandomOverSampler(random_state=42)
                    idx = result.index.to_numpy().reshape(-1, 1)
                    idx_res, _ = ros.fit_resample(idx, y)
                    result = result.iloc[idx_res.flatten()].reset_index(drop=True)
                    logger.info(f"fix(imbalance=oversample): размер датасета → {len(result)}")
                elif imb == 'undersample':
                    from imblearn.under_sampling import RandomUnderSampler
                    rus = RandomUnderSampler(random_state=42)
                    idx = result.index.to_numpy().reshape(-1, 1)
                    idx_res, _ = rus.fit_resample(idx, y)
                    result = result.iloc[idx_res.flatten()].reset_index(drop=True)
                    logger.info(f"fix(imbalance=undersample): размер датасета → {len(result)}")
            except ImportError:
                logger.warning("imbalanced-learn не установлен, пропускаем балансировку")

        result['_quality_flags'] = result['_quality_flags'].apply(lambda x: x if isinstance(x, list) else [])
        logger.info(f"fix(): итого {len(result)} строк после чистки")
        return result

    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
        """Сравнивает DataFrame до и после чистки.

        Args:
            df_before: исходный DataFrame
            df_after: очищенный DataFrame

        Returns:
            ComparisonReport со статистикой "было / стало"

        Example:
            report = agent.compare(df_raw, df_clean)
        """
        def null_count(df):
            return int(df.isna().sum().sum())

        def dup_count(df):
            return int(df.duplicated(subset='text' if 'text' in df.columns else None).sum())

        before_dist = {}
        after_dist = {}
        if 'label' in df_before.columns:
            before_dist = {str(k): round(float(v), 4)
                           for k, v in df_before['label'].value_counts(normalize=True).items()}
        if 'label' in df_after.columns:
            after_dist = {str(k): round(float(v), 4)
                          for k, v in df_after['label'].value_counts(normalize=True).items()}

        delta = len(df_after) - len(df_before)
        report = {
            'rows': {
                'before': len(df_before),
                'after': len(df_after),
                'delta': delta,
                'delta_pct': round(delta / len(df_before) * 100, 2) if len(df_before) > 0 else 0
            },
            'missing_total': {
                'before': null_count(df_before),
                'after': null_count(df_after)
            },
            'duplicates': {
                'before': dup_count(df_before),
                'after': dup_count(df_after)
            },
            'label_distribution': {
                'before': before_dist,
                'after': after_dist
            },
            'text_avg_length': {
                'before': round(float(df_before['text'].astype(str).str.len().mean()), 1) if 'text' in df_before.columns else 0,
                'after': round(float(df_after['text'].astype(str).str.len().mean()), 1) if 'text' in df_after.columns else 0,
            },
            'verdict': ''
        }

        improvements = []
        if report['missing_total']['after'] < report['missing_total']['before']:
            improvements.append(f"пропуски: {report['missing_total']['before']} → {report['missing_total']['after']}")
        if report['duplicates']['after'] < report['duplicates']['before']:
            improvements.append(f"дубли: {report['duplicates']['before']} → {report['duplicates']['after']}")
        report['verdict'] = "Улучшено: " + ", ".join(improvements) if improvements else "Существенных изменений нет"

        try:
            from tabulate import tabulate
            rows = [
                ["Строк", report['rows']['before'], report['rows']['after'], f"{delta:+d}"],
                ["Пропусков", report['missing_total']['before'], report['missing_total']['after'], ''],
                ["Дублей", report['duplicates']['before'], report['duplicates']['after'], ''],
                ["Ср. длина текста", report['text_avg_length']['before'], report['text_avg_length']['after'], ''],
            ]
            print("\n" + tabulate(rows, headers=["Метрика", "До", "После", "Δ"], tablefmt="rounded_outline"))
        except ImportError:
            print(f"\nДо: {len(df_before)} строк | После: {len(df_after)} строк | Δ={delta}")

        print(f"Вывод: {report['verdict']}\n")
        return report

    def _rule_based_explanation(self, report: dict, ml_task: str) -> str:
        """Автоматические рекомендации по результатам QualityReport без API."""
        lines = [f"=== Анализ качества данных для задачи: {ml_task} ===\n"]
        severity = report.get('summary', {}).get('severity', 'low')
        lines.append(f"Severity: {severity.upper()}\n")

        if report.get('missing'):
            lines.append("Пропущенные значения:")
            for col, info in report['missing'].items():
                rec = "drop" if info['pct'] > 20 else "median/mode"
                lines.append(f"  - {col}: {info['pct']}% пропусков → рекомендация: {rec}")

        if report.get('duplicates', {}).get('count', 0) > 0:
            lines.append(f"\nДубли: {report['duplicates']['count']} ({report['duplicates']['pct']}%) → удалить")

        if report.get('imbalance', {}).get('is_critical'):
            ratio = report['imbalance']['imbalance_ratio']
            lines.append(f"\nДисбаланс классов (ratio={ratio}) → oversample миноритарный класс")

        return "\n".join(lines)

    def llm_explain(self, report: dict, ml_task: str, api_key: str = None,
                    api_type: str = 'anthropic') -> str:
        """Объясняет найденные проблемы через LLM или rule-based логику.

        Args:
            report: QualityReport из detect_issues()
            ml_task: тип ML-задачи
            api_key: API ключ (если None — используется rule-based объяснение)
            api_type: 'anthropic' | 'openai'

        Returns:
            Строка с объяснением и рекомендациями
        """
        if not api_key:
            return self._rule_based_explanation(report, ml_task)

        prompt = f"""
        ML задача: {ml_task}
        Найденные проблемы качества данных: {json.dumps(report, indent=2, ensure_ascii=False)}

        1. Объясни каждую найденную проблему простым языком.
        2. Порекомендуй оптимальную стратегию чистки для данной ML-задачи.
        3. Обоснуй выбор стратегии.
        """

        if api_type == 'anthropic':
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model='claude-sonnet-4-6',
                max_tokens=1024,
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

        raise NotImplementedError("Укажите api_key и api_type для использования LLM-объяснений")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Путь к raw parquet')
    parser.add_argument('--output', default='data/clean/')
    parser.add_argument('--domain', default='news classification')
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    agent = DataQualityAgent()
    report = agent.detect_issues(df)

    Path('reports').mkdir(exist_ok=True)
    with open('reports/quality_report.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    strategy = {
        'duplicates': 'drop',
        'text_quality': 'drop_empty',
        'missing': 'drop' if report['summary']['severity'] == 'high' else 'median',
        'outliers': 'clip_iqr',
        'imbalance': 'oversample' if report.get('imbalance', {}).get('is_critical') else 'none'
    }
    df_clean = agent.fix(df, strategy=strategy)
    agent.compare(df, df_clean)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y%m%d")
    out_path = Path(args.output) / f"cleaned_{args.domain.replace(' ', '_')}_{date}.parquet"
    df_clean.to_parquet(out_path, index=False)
    print(f"✅ Сохранено: {out_path} ({len(df_clean)} строк)")
