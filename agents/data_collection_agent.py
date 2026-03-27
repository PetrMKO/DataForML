"""
DataCollectionAgent — Шаг 1: Сбор данных для задачи news classification.

Использование:
    agent = DataCollectionAgent(config='config.yaml')
    df = agent.run(domain='news classification', modality='text')
    df.to_parquet('data/raw/collected.parquet', index=False)
"""
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class DataCollectionAgent:
    """Собирает данные из нескольких источников и возвращает унифицированный DataFrame.

    Example:
        agent = DataCollectionAgent(config='config.yaml')
        df = agent.run(domain='news classification', modality='text')
    """

    def __init__(self, config: str = 'config.yaml'):
        """
        Args:
            config: путь к config.yaml
        """
        config_path = Path(config)
        if config_path.exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        logger.info(f"DataCollectionAgent инициализирован. Config: {config}")

    def scrape(self, url: str, selector: str, max_pages: int = 3) -> pd.DataFrame:
        """Парсит HTML-страницы и возвращает DataFrame с текстами.

        Args:
            url: URL страницы или шаблон с {page}
            selector: CSS-селектор для извлечения текстовых элементов
            max_pages: максимальное число страниц

        Returns:
            DataFrame с колонками: text, source, collected_at

        Example:
            df = agent.scrape('https://news.ycombinator.com/', 'a.titlelink')
        """
        records = []
        headers = {'User-Agent': 'Mozilla/5.0 (research bot)'}

        for page in range(1, max_pages + 1):
            page_url = url.format(page=page) if '{page}' in url else url
            for attempt in range(3):
                try:
                    resp = requests.get(page_url, headers=headers, timeout=10)
                    resp.raise_for_status()
                    break
                except requests.RequestException as e:
                    logger.warning(f"Попытка {attempt+1}/3 для {page_url}: {e}")
                    if attempt == 2:
                        logger.error(f"Не удалось получить {page_url}")
                        continue
                    time.sleep(2)

            soup = BeautifulSoup(resp.text, 'html.parser')
            elements = soup.select(selector)
            for el in elements:
                text = el.get_text(strip=True)
                if text:
                    records.append({
                        'text': text,
                        'source': url,
                        'collected_at': datetime.now().isoformat()
                    })

            if page_url == url:
                break

        df = pd.DataFrame(records)
        logger.info(f"scrape(): получено {len(df)} строк из {url}")
        return df

    def fetch_api(self, endpoint: str, params: Optional[dict] = None,
                  max_pages: int = 5, headers: Optional[dict] = None) -> pd.DataFrame:
        """Получает данные из REST API с поддержкой пагинации.

        Args:
            endpoint: URL эндпоинта
            params: query-параметры
            max_pages: максимальное число страниц пагинации
            headers: заголовки авторизации

        Returns:
            DataFrame с колонками: text, source, collected_at

        Example:
            df = agent.fetch_api('https://api.example.com/news', {'category': 'tech'})
        """
        records = []
        params = params or {}
        headers = headers or {}

        for page in range(1, max_pages + 1):
            params['page'] = page
            try:
                resp = requests.get(endpoint, params=params, headers=headers, timeout=10)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error(f"fetch_api() ошибка на странице {page}: {e}")
                break

            items = data if isinstance(data, list) else data.get('results', data.get('items', []))
            if not items:
                break

            for item in items:
                text = item.get('text') or item.get('content') or item.get('title') or str(item)
                records.append({
                    'text': text,
                    'source': endpoint,
                    'collected_at': datetime.now().isoformat()
                })

        df = pd.DataFrame(records)
        logger.info(f"fetch_api(): получено {len(df)} строк из {endpoint}")
        return df

    def load_dataset(self, name: str, source: str = 'hf',
                     split: str = 'train', max_samples: int = 5000,
                     text_col: Optional[str] = None,
                     label_col: Optional[str] = None) -> pd.DataFrame:
        """Загружает датасет из HuggingFace или Kaggle.

        Args:
            name: название датасета
            source: 'hf' для HuggingFace, 'kaggle' для Kaggle
            split: split датасета (train/test/validation)
            max_samples: максимальное число примеров
            text_col: колонка с текстом (автоопределение если None)
            label_col: колонка с метками (автоопределение если None)

        Returns:
            DataFrame со стандартными колонками: text, label, source, collected_at

        Example:
            df = agent.load_dataset('ag_news', source='hf', max_samples=3000)
        """
        if source == 'hf':
            try:
                from datasets import load_dataset as hf_load
            except ImportError:
                raise ImportError("Установите datasets: pip install datasets")

            logger.info(f"Загрузка HuggingFace датасета: {name} (split={split})")
            try:
                raw = hf_load(name, split=split, trust_remote_code=True)
            except Exception:
                try:
                    raw = hf_load(name, split=split)
                except Exception as e:
                    logger.error(f"Не удалось загрузить {name}: {e}")
                    return pd.DataFrame(columns=['text', 'label', 'source', 'collected_at'])

            df = raw.to_pandas()
            if max_samples and len(df) > max_samples:
                df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        elif source == 'kaggle':
            try:
                import kaggle
                kaggle.api.authenticate()
                Path('/tmp/kaggle_data').mkdir(exist_ok=True)
                kaggle.api.dataset_download_files(name, path='/tmp/kaggle_data', unzip=True)
                csvs = list(Path('/tmp/kaggle_data').glob('*.csv'))
                if not csvs:
                    raise FileNotFoundError("CSV файлы не найдены после скачивания")
                df = pd.read_csv(csvs[0])
                if max_samples and len(df) > max_samples:
                    df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
            except Exception as e:
                logger.error(f"Ошибка Kaggle: {e}")
                return pd.DataFrame(columns=['text', 'label', 'source', 'collected_at'])
        else:
            raise ValueError(f"Неизвестный источник: {source}. Используй 'hf' или 'kaggle'")

        # Автоопределение колонок
        if text_col is None:
            for col in df.columns:
                if any(kw in col.lower() for kw in ('text', 'sentence', 'review', 'content', 'title', 'description')):
                    text_col = col
                    break
            if text_col is None:
                text_col = df.columns[0]
                logger.warning(f"Текстовая колонка не найдена, используем: {text_col}")

        if label_col is None:
            for col in df.columns:
                if any(kw in col.lower() for kw in ('label', 'target', 'sentiment', 'category', 'class')):
                    label_col = col
                    break

        result = pd.DataFrame()
        result['text'] = df[text_col].astype(str)
        result['label'] = df[label_col] if label_col and label_col in df.columns else None

        # Для ag_news маппинг числовых меток в текстовые
        if name in ('ag_news', 'fancyzhx/ag_news') and result['label'] is not None:
            label_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
            result['label'] = result['label'].map(label_map).fillna(result['label'])

        result['source'] = name
        result['collected_at'] = datetime.now().isoformat()

        logger.info(f"load_dataset(): загружено {len(result)} строк из {name}")
        if label_col:
            logger.info(f"  Распределение меток: {result['label'].value_counts().to_dict()}")
        return result

    def merge(self, sources: list) -> pd.DataFrame:
        """Объединяет список DataFrame, дедуплицирует по тексту.

        Args:
            sources: список DataFrame для объединения

        Returns:
            Объединённый DataFrame без дублей

        Example:
            df = agent.merge([df1, df2, df3])
        """
        non_empty = [df for df in sources if len(df) > 0]
        if not non_empty:
            raise ValueError("Все источники пусты — нет данных для объединения")

        for i, df in enumerate(non_empty):
            logger.info(f"  Источник {i+1}: {len(df)} строк")

        merged = pd.concat(non_empty, ignore_index=True)
        before = len(merged)
        merged = merged.drop_duplicates(subset='text', keep='first').reset_index(drop=True)
        after = len(merged)
        logger.info(f"merge(): итого {after} строк (дублей удалено: {before - after})")
        return merged

    def run(self, domain: str = None, modality: str = 'text',
            sources: Optional[list] = None) -> pd.DataFrame:
        """Основной метод — оркестрирует сбор из всех источников.

        Args:
            domain: предметная область
            modality: тип данных ('text', 'audio', 'image')
            sources: список источников (если None — берёт из config.yaml)

        Returns:
            Унифицированный DataFrame со стандартными колонками

        Example:
            df = agent.run(domain='news classification', modality='text')
        """
        logger.info(f"DataCollectionAgent.run() domain={domain}, modality={modality}")

        cfg_sources = sources or self.config.get('sources', [])

        if not cfg_sources:
            logger.warning("Источники не указаны, использую ag_news по умолчанию")
            cfg_sources = [
                {'type': 'hf_dataset', 'name': 'ag_news', 'split': 'train', 'max_samples': 3000}
            ]

        dataframes = []
        for src in cfg_sources:
            src_type = src.get('type')
            try:
                if src_type == 'hf_dataset':
                    df = self.load_dataset(
                        name=src['name'],
                        source='hf',
                        split=src.get('split', 'train'),
                        max_samples=src.get('max_samples', 5000),
                        text_col=src.get('text_col'),
                        label_col=src.get('label_col')
                    )
                    dataframes.append(df)
                elif src_type == 'scrape':
                    df = self.scrape(
                        url=src['url'],
                        selector=src['selector'],
                        max_pages=src.get('max_pages', 3)
                    )
                    dataframes.append(df)
                elif src_type == 'api':
                    df = self.fetch_api(
                        endpoint=src['endpoint'],
                        params=src.get('params', {}),
                        max_pages=src.get('max_pages', 5)
                    )
                    dataframes.append(df)
                elif src_type == 'kaggle':
                    df = self.load_dataset(
                        name=src['name'],
                        source='kaggle',
                        max_samples=src.get('max_samples', 5000)
                    )
                    dataframes.append(df)
                else:
                    logger.warning(f"Неизвестный тип источника: {src_type}")
            except Exception as e:
                logger.error(f"Ошибка при сборе из {src}: {e}")

        if not dataframes:
            raise RuntimeError("Не удалось собрать данные ни из одного источника")

        result = self.merge(dataframes)
        logger.info(f"\n{'='*50}")
        logger.info(f"Сбор данных завершён:")
        logger.info(f"  Всего строк: {len(result)}")
        logger.info(f"  Колонки: {list(result.columns)}")
        if 'label' in result.columns:
            logger.info(f"  Классы: {result['label'].value_counts().to_dict()}")
        logger.info(f"{'='*50}")
        return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DataCollectionAgent — сбор данных')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--domain', default='news classification')
    parser.add_argument('--output', default='data/raw/')
    args = parser.parse_args()

    agent = DataCollectionAgent(config=args.config)
    df = agent.run(domain=args.domain, modality='text')

    Path(args.output).mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y%m%d")
    output_path = Path(args.output) / f"collected_{args.domain.replace(' ', '_')}_{date}.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\n✅ Сохранено: {output_path} ({len(df)} строк)")
