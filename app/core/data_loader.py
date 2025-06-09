# app/core/data_loader.py

import pandas as pd
import requests
from datetime import datetime
import os
import logging
from app.utils import config

logging.basicConfig(level=logging.DEBUG if config.DEBUG_MODE else logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_moex_data(ticker: str, board: str, start_date_str: str, end_date_str: str) -> pd.DataFrame | None:
    """
    Получает исторические данные котировок (только CLOSE) с MOEX ISS API.
    """
    base_url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/{board}/securities/{ticker}.json"
    all_data = []
    start_index = 0

    logging.info(f"Загрузка данных MOEX ISS для {ticker} ({board}) с {start_date_str} по {end_date_str}...")

    while True:
        params = {
            'from': start_date_str,
            'till': end_date_str,
            'start': start_index,
            'iss.meta': 'off',
            # Запрашиваем все нужные колонки OHLCV сразу
            'history.columns': 'TRADEDATE,OPEN,HIGH,LOW,CLOSE,VOLUME'
        }
        try:
            response = requests.get(base_url, params=params, timeout=30) # Добавлен таймаут
            response.raise_for_status() # Проверка на HTTP ошибки (4xx, 5xx)
            data = response.json()

            # Проверяем корректность ответа API
            if 'history' in data and 'data' in data['history']:
                chunk = data['history']['data']
                if chunk: # Если список не пустой
                    all_data.extend(chunk)
                    # API обычно возвращает не более 100 записей за раз
                    if len(chunk) < 100:
                        break # Больше данных нет
                    else:
                        start_index += 100 # Переходим к следующей странице
                else:
                    # Если первая же страница пуста
                    if start_index == 0:
                         logging.warning(f"MOEX ISS: Данные для {ticker} ({board}) не найдены за период {start_date_str} - {end_date_str}.")
                    break # Больше данных нет
            else:
                logging.error(f"MOEX ISS: Неожиданный формат ответа для {ticker}. 'history' или 'data' отсутствуют. Ответ: {str(data)[:500]}...")
                return None

        except requests.exceptions.Timeout:
             logging.error(f"MOEX ISS: Таймаут при запросе данных для {ticker}.")
             return None
        except requests.exceptions.RequestException as e:
            logging.error(f"MOEX ISS: Ошибка сети или HTTP при запросе для {ticker}: {e}")
            return None
        except ValueError as e: # Ошибка декодирования JSON
            logging.error(f"MOEX ISS: Ошибка декодирования JSON ответа для {ticker}: {e}. Ответ: {response.text[:500]}...")
            return None
        except Exception as e:
            logging.error(f"MOEX ISS: Неожиданная ошибка при обработке данных для {ticker}: {e}", exc_info=config.DEBUG_MODE)
            return None

    if not all_data:
        logging.warning(f"MOEX ISS: Не удалось загрузить данные для {ticker} ({board}). Список пуст.")
        return None

    # Указываем корректные колонки, которые мы запросили
    columns = ['TRADEDATE', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = pd.DataFrame(all_data, columns=columns)

    # Преобразование типов и обработка ошибок
    try:
        df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
        df = df.sort_values('TRADEDATE')
        df.set_index('TRADEDATE', inplace=True)
        # Преобразуем все числовые колонки, заменяя ошибки на NaN
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Удаляем строки с NaN, которые могли появиться при преобразовании
        initial_len = len(df)
        df.dropna(inplace=True)
        if len(df) < initial_len:
            logging.warning(f"MOEX ISS: Удалено {initial_len - len(df)} строк с некорректными числовыми данными для {ticker}.")

        if df.empty:
            logging.error(f"MOEX ISS: DataFrame для {ticker} пуст после обработки данных.")
            return None

    except Exception as e:
        logging.error(f"MOEX ISS: Ошибка при обработке DataFrame для {ticker}: {e}", exc_info=config.DEBUG_MODE)
        return None

    logging.info(f"MOEX ISS: Загружено {len(df)} валидных торговых дней для {ticker} ({board}).")
    return df



def load_stock_data_moex(ticker: str, start_date_str: str, end_date_str: str) -> pd.DataFrame | None:
    """
    Обертка для вызова fetch_moex_data с параметрами из конфига.
    """
    board = 'TQBR'
    return fetch_moex_data(ticker, board, start_date_str, end_date_str)


def load_data_from_csv(csv_path: str) -> pd.DataFrame | None:
    """Загружает данные из локального CSV файла."""
    logging.info(f"Загрузка данных из CSV: {csv_path}")
    if not os.path.exists(csv_path):
        logging.error(f"Файл CSV не найден: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path, index_col='TRADEDATE', parse_dates=True)
        df.index.name = 'Date'
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
             logging.error(f"CSV файл {csv_path} не содержит все необходимые колонки ({required_cols}).")
             return None
        logging.info(f"Данные из CSV {csv_path} успешно загружены. Строк: {len(df)}")
        return df
    except Exception as e:
        logging.error(f"Ошибка при чтении CSV файла {csv_path}: {e}", exc_info=config.DEBUG_MODE)
        return None

def save_data_to_csv(df: pd.DataFrame, ticker: str, start_date_str: str, end_date_str: str) -> str | None:
    """Сохраняет DataFrame в CSV файл в папке assets/data."""
    filename = f"{ticker}_{start_date_str.replace('-', '')}_{end_date_str.replace('-', '')}.csv"
    filepath = os.path.join(config.DATA_DIR, filename)
    logging.info(f"Попытка сохранения данных в CSV: {filepath}")
    try:
        cols_to_save = ['Open', 'High', 'Low', 'Close', 'Volume']
        df_to_save = df[cols_to_save]
        df_to_save.to_csv(filepath, date_format='%Y-%m-%d')
        logging.info(f"Данные успешно сохранены в {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"Ошибка при сохранении данных в CSV {filepath}: {e}", exc_info=config.DEBUG_MODE)
        return None