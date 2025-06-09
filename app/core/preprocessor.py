# app/core/preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import logging
import json
from app.utils import config

logger = logging.getLogger(__name__)

# --- НОВАЯ ФУНКЦИЯ: Дифференцирование ---
def add_differencing(df: pd.DataFrame, column: str, order: int) -> pd.DataFrame:
    """Добавляет дифференцированную колонку."""
    if column not in df.columns:
        logger.error(f"Колонка '{column}' отсутствует в DataFrame для дифференцирования.")
        return df
    
    diff_col_name = f"{column}_diff{order}"
    df[diff_col_name] = df[column].diff(periods=order)
    # Первые 'order' строк будут NaN, их нужно будет удалить позже, после всех расчетов
    logger.debug(f"Рассчитана дифференцированная колонка: '{diff_col_name}' (порядок {order}).")
    return df

# --- Функции для расчета индикаторов (небольшие правки для явного указания колонки) ---
def add_sma(df: pd.DataFrame, window: int, column: str = 'Close') -> pd.DataFrame:
    if column not in df.columns:
        logger.error(f"Колонка '{column}' отсутствует в DataFrame для расчета SMA_{window}.")
        return df
    df[f'SMA_{window}'] = df[column].rolling(window=window, min_periods=1).mean()
    return df

def add_bollinger_bands(df: pd.DataFrame, window: int, std_dev: int, column: str = 'Close') -> pd.DataFrame:
    if column not in df.columns:
        logger.error(f"Колонка '{column}' отсутствует в DataFrame для расчета Bollinger Bands (окно {window}).")
        return df
    sma = df[column].rolling(window=window, min_periods=1).mean()
    rolling_std = df[column].rolling(window=window, min_periods=1).std()
    df[f'BB_Middle_{window}'] = sma
    df[f'BB_Upper_{window}'] = sma + (rolling_std * std_dev)
    df[f'BB_Lower_{window}'] = sma - (rolling_std * std_dev)
    return df

def add_rsi(df: pd.DataFrame, window: int, column: str = 'Close') -> pd.DataFrame:
    if column not in df.columns:
        logger.error(f"Колонка '{column}' отсутствует в DataFrame для расчета RSI_{window}.")
        return df
    delta = df[column].diff(1)
    gain = delta.where(delta > 0, 0.0).rolling(window=window, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(window=window, min_periods=1).mean()
    rs = gain / loss
    rs = rs.replace([np.inf, -np.inf], np.nan)

    if rs.isnull().any():
        rs = rs.bfill()
        rs = rs.ffill()
    rs.fillna(1, inplace=True)

    df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
    df[f'RSI_{window}'] = np.clip(df[f'RSI_{window}'], 0, 100)
    return df

def calculate_indicators_and_diff(df: pd.DataFrame, 
                                  use_differencing_runtime: bool
                                 ) -> pd.DataFrame:
    """
    Рассчитывает технические индикаторы и, если включено в конфиге,
    дифференцирует целевой столбец.
    NaN строки, возникшие из-за rolling window и дифференцирования, удаляются в конце.
    """
    df_copy = df.copy()
    logger.info(f"Начало расчета признаков для DataFrame размером {df_copy.shape}.")

    # 1. Расчет технических индикаторов
    base_col_for_indicators = config.TARGET_FEATURE_ORIGINAL

    if base_col_for_indicators not in df_copy.columns:
        logger.error(f"Ключевая колонка '{base_col_for_indicators}' для расчета индикаторов отсутствует. Расчет индикаторов пропущен.")
    else:
        for name, params in config.INDICATORS_CONFIG.items():
            try:
                indicator_base_col = params.get('column', base_col_for_indicators)
                if indicator_base_col not in df_copy.columns:
                    logger.warning(f"Базовая колонка '{indicator_base_col}' для индикатора {name} отсутствует. Пропускаем.")
                    continue
                
                logger.debug(f"Расчет индикатора: {name} на колонке '{indicator_base_col}' с параметрами {params}")
                if name.upper().startswith('SMA_'):
                    window = params.get('window')
                    if window:
                        df_copy = add_sma(df_copy, window, column=indicator_base_col)
                        logger.debug(f"Рассчитан {name}.")
                elif name.upper() == 'BOLLINGERBANDS':
                    window = params.get('window')
                    std = params.get('std')
                    if window and std:
                        df_copy = add_bollinger_bands(df_copy, window, std, column=indicator_base_col)
                        logger.debug(f"Рассчитаны Bollinger Bands (окно={window}, std={std}).")
                elif name.upper().startswith('RSI_'):
                    window = params.get('window')
                    if window:
                        df_copy = add_rsi(df_copy, window, column=indicator_base_col)
                        logger.debug(f"Рассчитан {name}.")
                else:
                    logger.warning(f"Неизвестный тип индикатора '{name}' или логика не реализована.")
            except Exception as e:
                logger.error(f"Ошибка при расчете индикатора {name}: {e}", exc_info=config.DEBUG_MODE)

    # 2. Дифференцирование (если включено)
    if use_differencing_runtime:
        logger.info(f"Применение дифференцирования порядка {config.DIFFERENCING_ORDER} для колонки '{config.DIFFERENCING_COLUMN}'.")
        df_copy = add_differencing(df_copy, config.DIFFERENCING_COLUMN, config.DIFFERENCING_ORDER)
    
    # 3. Удаление всех NaN, которые могли появиться
    initial_len = len(df_copy)
    df_copy.dropna(inplace=True)
    dropped_rows = initial_len - len(df_copy)
    if dropped_rows > 0:
         logger.info(f"Удалено {dropped_rows} строк с NaN после расчета всех признаков и дифференцирования.")
    
    if df_copy.empty:
        logger.warning("DataFrame стал пустым после расчета признаков, дифференцирования и удаления NaN.")
        return pd.DataFrame()
    return df_copy


def preprocess_for_training(df: pd.DataFrame, model_save_dir: str,
                            test_split_ratio: float, prediction_horizon: int,
                            use_differencing_runtime: bool,
                            current_target_feature_runtime: str
                           ) -> tuple | None:
    """
    Предобрабатывает данные для обучения: расчет индикаторов (и дифференцирование, если включено),
    разделение на train/test, масштабирование, создание последовательностей X и Y,
    сохранение скейлеров.
    """
    try:
        logger.info(f"Начало предобработки данных для обучения. Горизонт прогноза: {prediction_horizon} дней, Доля теста: {test_split_ratio:.2f}")
        logger.info(f"Используется дифференцирование (runtime): {use_differencing_runtime}. "
                    f"Целевой признак (runtime): '{current_target_feature_runtime}'. "
                    f"Исходный целевой признак (для справки): '{config.TARGET_FEATURE_ORIGINAL}'.")


        df_processed = calculate_indicators_and_diff(df, use_differencing_runtime=use_differencing_runtime)
        if df_processed.empty:
            logger.error("DataFrame пуст после расчета признаков/дифференцирования. Предобработка для обучения прервана.")
            return None

        # Проверка, что все ФИЧИ из config.FEATURES и ЦЕЛЕВОЙ ПРИЗНАК config.TARGET_FEATURE присутствуют
        missing_features = [f for f in config.FEATURES if f not in df_processed.columns]
        if missing_features:
            logger.error(f"Следующие признаки из config.FEATURES отсутствуют в DataFrame: {missing_features}. Предобработка прервана.")
            return None
        if current_target_feature_runtime not in df_processed.columns:
            logger.error(f"Целевой признак '{current_target_feature_runtime}' (ожидаемый для обучения) отсутствует в DataFrame. Предобработка прервана.")
            return None

        logger.debug(f"DataFrame с признаками (перед разделением): {df_processed.shape[0]} строк.")
        logger.debug(f"Признаки для модели (config.FEATURES): {config.FEATURES}")

        logger.debug(f"Цель для модели (runtime): {current_target_feature_runtime}")

        # 1. Разделение на обучающую и тестовую выборки ДО масштабирования
        split_index = int(len(df_processed) * (1 - test_split_ratio))
        
        min_data_for_sequence = config.TIME_STEPS + prediction_horizon -1 
        
        if split_index < min_data_for_sequence :
            logger.error(f"Недостаточно данных для обучающей выборки после разделения. "
                         f"Требуется {min_data_for_sequence}, получено {split_index}. Всего данных: {len(df_processed)}.")
            return None
        
        train_df_full = df_processed.iloc[:split_index]
        test_df_full = df_processed.iloc[split_index:] # Может быть пустым, если test_split_ratio слишком мал
        
        last_original_target_value_in_train = None
        if use_differencing_runtime and not train_df_full.empty:
            if config.TARGET_FEATURE_ORIGINAL in train_df_full.columns:
                 last_original_target_value_in_train = train_df_full[config.TARGET_FEATURE_ORIGINAL].iloc[-1]
                 logger.debug(f"Сохранено последнее значение '{config.TARGET_FEATURE_ORIGINAL}' из train_df_full: {last_original_target_value_in_train}")
            else:
                 logger.warning(f"Колонка '{config.TARGET_FEATURE_ORIGINAL}' не найдена в train_df_full, не удастся сохранить последнее значение.")
        
        # Сохраняем также исходные недифференцированные значения для теста, если они есть
        y_test_original_absolute = pd.Series(dtype=float)
        if not test_df_full.empty and config.TARGET_FEATURE_ORIGINAL in test_df_full.columns:
            y_test_original_absolute = test_df_full[config.TARGET_FEATURE_ORIGINAL].copy()
            logger.debug(f"Сохранены исходные абсолютные значения '{config.TARGET_FEATURE_ORIGINAL}' для тестовой выборки ({len(y_test_original_absolute)} значений).")


        logger.info(f"DataFrame разделен на train ({train_df_full.shape[0]} строк) и test ({test_df_full.shape[0]} строк) ДО масштабирования.")

        train_features_df = train_df_full[config.FEATURES].copy()
        train_target_series = train_df_full[current_target_feature_runtime].copy()
        
        test_features_df = test_df_full[config.FEATURES].copy() if not test_df_full.empty else pd.DataFrame()
        test_target_series = test_df_full[current_target_feature_runtime].copy() if not test_df_full.empty else pd.Series(dtype=float)


        # 2. Масштабирование
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1)) # Скейлер для TARGET_FEATURE (возможно, разностей)

        scaled_train_features = feature_scaler.fit_transform(train_features_df)
        scaled_train_target = target_scaler.fit_transform(train_target_series.values.reshape(-1, 1))
        logger.info("Скейлеры обучены на ТРЕНИРОВОЧНЫХ данных (признаки и текущий TARGET_FEATURE).")
        
        logger.debug(f"Статистика scaled_train_target ({current_target_feature_runtime}): "
                     f"Min={np.min(scaled_train_target):.6f}, "
                     f"Max={np.max(scaled_train_target):.6f}, "
                     f"Mean={np.mean(scaled_train_target):.6f}, "
                     f"StdDev={np.std(scaled_train_target):.6f}")
        near_zero_threshold = 1e-5 
        num_near_zero = np.sum(np.abs(scaled_train_target) < near_zero_threshold)
        logger.debug(f"Количество значений в scaled_train_target ({current_target_feature_runtime}) близких к нулю (< {near_zero_threshold}): {num_near_zero} из {len(scaled_train_target)}")

        scaled_test_features = np.array([])
        scaled_test_target = np.array([])
        if not test_features_df.empty and not test_target_series.empty:
            scaled_test_features = feature_scaler.transform(test_features_df)
            scaled_test_target = target_scaler.transform(test_target_series.values.reshape(-1, 1))
            logger.info("Масштабирование применено к ТЕСТОВЫМ данным.")
        else:
            logger.warning("Тестовая выборка (признаки или цель) пуста, масштабирование для нее не выполняется.")

        # 3. Сохранение скейлеров
        if not os.path.exists(model_save_dir):
            try:
                os.makedirs(model_save_dir)
                logger.info(f"Создана директория для сохранения скейлеров: {model_save_dir}")
            except OSError as e:
                logger.error(f"Не удалось создать директорию {model_save_dir} для скейлеров: {e}")
                return None
        joblib.dump(feature_scaler, os.path.join(model_save_dir, 'feature_scaler.pkl'))
        joblib.dump(target_scaler, os.path.join(model_save_dir, 'target_scaler.pkl')) # Скейлер для TARGET_FEATURE
        logger.info(f"Скейлеры сохранены в: {model_save_dir}")

        # 4. Создание последовательностей X_train, y_train
        X_train, y_train = [], []
        if len(scaled_train_features) >= config.TIME_STEPS + prediction_horizon -1 :
            for i in range(config.TIME_STEPS, len(scaled_train_features) - prediction_horizon + 1):
                X_train.append(scaled_train_features[i - config.TIME_STEPS:i, :])
                y_train.append(scaled_train_target[i : i + prediction_horizon, 0])
        
        if not X_train or not y_train:
            logger.error(f"Не удалось создать ТРЕНИРОВОЧНЫЕ последовательности X_train, y_train. "
                         f"Длина scaled_train_features: {len(scaled_train_features)}.")
            return None
        X_train, y_train = np.array(X_train), np.array(y_train)
        logger.info(f"Созданы ТРЕНИРОВОЧНЫЕ последовательности: X_train shape={X_train.shape}, y_train shape={y_train.shape} (цель: {current_target_feature_runtime})")
        # 5. Создание последовательностей X_test, y_test
        X_test, y_test = [], []
        # y_test_original_for_eval будет содержать абсолютные значения для оценки, если USE_DIFFERENCING
        # а y_test будет содержать то, на чем модель непосредственно оценивается (возможно, разности)
        y_test_original_for_eval = np.array([]) 
        
        if len(scaled_test_features) >= config.TIME_STEPS + prediction_horizon -1:
            temp_y_test_original_list = []
            
            for i in range(config.TIME_STEPS, len(scaled_test_features) - prediction_horizon + 1):
                X_test.append(scaled_test_features[i - config.TIME_STEPS:i, :])
                y_test.append(scaled_test_target[i : i + prediction_horizon, 0])
                
                # Если используем дифференцирование, нам нужны соответствующие ОРИГИНАЛЬНЫЕ АБСОЛЮТНЫЕ значения для оценки
                if config.USE_DIFFERENCING and not y_test_original_absolute.empty:
                    original_target_slice = y_test_original_absolute.iloc[i : i + prediction_horizon].values
                    temp_y_test_original_list.append(original_target_slice)

            if X_test and y_test:
                X_test, y_test = np.array(X_test), np.array(y_test)
                logger.info(f"Созданы ТЕСТОВЫЕ последовательности: X_test shape={X_test.shape}, y_test shape={y_test.shape} (цель: {current_target_feature_runtime})")
                if use_differencing_runtime and temp_y_test_original_list:
                    y_test_original_for_eval = np.array(temp_y_test_original_list)
                    logger.info(f"Создан массив y_test_original_for_eval (абсолютные значения) shape={y_test_original_for_eval.shape} для оценки.")
            else:
                X_test, y_test = np.array([]), np.array([])
                y_test_original_for_eval = np.array([])
        else:
            logger.warning(f"Недостаточно данных в ТЕСТОВОЙ выборке ({len(scaled_test_features)} строк) для создания "
                           f"полноценных последовательностей. X_test и y_test будут пустыми.")
            X_test, y_test = np.array([]), np.array([])
            y_test_original_for_eval = np.array([])
        
        if X_train.shape[0] == 0:
            logger.error("Обучающая выборка X_train пуста.")
            return None

        logger.info(f"Данные успешно разделены и предобработаны.")
        
        # Возвращаем также last_original_target_value_in_train и y_test_original_for_eval
        # Эти значения понадобятся в trainer.py для восстановления абсолютных цен и корректной оценки
        return (X_train, y_train, X_test, y_test, 
                feature_scaler, target_scaler, 
                last_original_target_value_in_train, y_test_original_for_eval, df_processed)

    except Exception as e:
        logger.error(f"Критическая ошибка в preprocess_for_training: {e}", exc_info=config.DEBUG_MODE)
        return None


def preprocess_for_prediction(df_latest: pd.DataFrame, model_load_dir: str) -> tuple | None:
    """
    Готовит данные для одного предсказания: загружает скейлеры, рассчитывает признаки
    (и дифференцирование, если модель так обучалась), формирует последнюю входную последовательность X.
    Возвращает: input_sequence, last_actual_price_original (исходная цена), df_with_features (для графика),
                last_known_original_target_for_diff_restore (если используется дифференцирование).
    """
    try:
        logger.info("Начало предобработки данных для предсказания...")
        
        # Загрузка метаданных, чтобы узнать, использовалось ли дифференцирование при обучении этой модели
        metadata_path = os.path.join(model_load_dir, 'metadata.json')
        model_used_differencing = False
        model_diff_column = None
        model_diff_order = None
        model_target_feature_original = config.TARGET_FEATURE_ORIGINAL
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            model_used_differencing = metadata.get('used_differencing', False)
            if model_used_differencing:
                model_diff_column = metadata.get('differencing_column', config.DIFFERENCING_COLUMN)
                model_diff_order = metadata.get('differencing_order', config.DIFFERENCING_ORDER)
                model_target_feature_original = metadata.get('target_feature_original', config.TARGET_FEATURE_ORIGINAL)
                logger.info(f"Модель '{os.path.basename(model_load_dir)}' обучалась с дифференцированием (колонка: {model_diff_column}, порядок: {model_diff_order}).")
            else:
                 logger.info(f"Модель '{os.path.basename(model_load_dir)}' обучалась на абсолютных значениях (согласно метаданным или их отсутствию).")
        else:
            logger.warning(f"Файл metadata.json не найден в {model_load_dir}. Невозможно точно определить, "
                           f"использовалось ли дифференцирование при обучении. "
                           f"Будет использована текущая настройка config.USE_DIFFERENCING={config.USE_DIFFERENCING}.")
            # Если метаданных нет, полагаемся на текущий конфиг
            model_used_differencing = config.USE_DIFFERENCING
            if model_used_differencing:
                model_diff_column = config.DIFFERENCING_COLUMN
                model_diff_order = config.DIFFERENCING_ORDER
        
        feature_scaler_path = os.path.join(model_load_dir, 'feature_scaler.pkl')
        # target_scaler (для предсказываемой величины - разностей или абсолютных цен)
        
        if not os.path.exists(feature_scaler_path):
            logger.error(f"Не найден файл скейлера признаков ('feature_scaler.pkl') в {model_load_dir}")
            return None
        
        feature_scaler: MinMaxScaler = joblib.load(feature_scaler_path)
        logger.debug(f"Скейлер признаков загружен из {model_load_dir}")

        df_copy = df_latest.copy()
        
        # 1. Расчет индикаторов
        base_col_for_indicators_pred = model_target_feature_original

        if base_col_for_indicators_pred not in df_copy.columns:
            logger.error(f"Ключевая колонка '{base_col_for_indicators_pred}' для расчета индикаторов отсутствует. Расчет индикаторов пропущен.")
        else:
            for name, params in config.INDICATORS_CONFIG.items():
                try:
                    indicator_base_col = params.get('column', base_col_for_indicators_pred)
                    if indicator_base_col not in df_copy.columns:
                        logger.warning(f"Базовая колонка '{indicator_base_col}' для индикатора {name} (предсказание) отсутствует. Пропускаем.")
                        continue
                    
                    if name.upper().startswith('SMA_'):
                        df_copy = add_sma(df_copy, params['window'], column=indicator_base_col)
                    elif name.upper() == 'BOLLINGERBANDS':
                        df_copy = add_bollinger_bands(df_copy, params['window'], params['std'], column=indicator_base_col)
                    elif name.upper().startswith('RSI_'):
                        df_copy = add_rsi(df_copy, params['window'], column=indicator_base_col)
                except Exception as e_ind:
                    logger.error(f"Ошибка расчета индикатора {name} для предсказания: {e_ind}")
        
        # 2. Дифференцирование, ЕСЛИ модель обучалась с ним
        if model_used_differencing:
            logger.info(f"Применение дифференцирования (порядок {model_diff_order}) для колонки '{model_diff_column}' для предсказания.")
            df_copy = add_differencing(df_copy, model_diff_column, model_diff_order)
        
        # 3. Удаление NaN
        initial_len_pred = len(df_copy)
        df_copy.dropna(inplace=True) # Удаляем все NaN
        if len(df_copy) < initial_len_pred:
            logger.info(f"Удалено {initial_len_pred - len(df_copy)} строк с NaN после расчета признаков/дифференцирования (для предсказания).")

        df_with_features = df_copy # Это DataFrame, который пойдет на график и для извлечения признаков

        if df_with_features.empty:
            logger.error("DataFrame (для предсказания) пуст после расчета признаков/дифференцирования. Предобработка прервана.")
            return None

        # Проверка наличия всех необходимых признаков (config.FEATURES) ПОСЛЕ всех преобразований
        missing_features_pred = [f for f in config.FEATURES if f not in df_with_features.columns]
        if missing_features_pred:
            logger.error(f"Признаки {missing_features_pred} из config.FEATURES отсутствуют в DataFrame (для предсказания).")
            return None
        
        # Последняя актуальная ИСХОДНАЯ цена Close для информации и для восстановления, если было дифференцирование
        if model_target_feature_original not in df_with_features.columns:
            logger.error(f"Исходная целевая колонка '{model_target_feature_original}' отсутствует в df_with_features (для предсказания).")

            if model_target_feature_original in df_latest.columns and not df_latest.empty:
                 last_actual_price_original = df_latest[model_target_feature_original].iloc[-1]
                 logger.warning(f"Взято значение '{model_target_feature_original}' из исходного df_latest, т.к. в df_with_features его нет.")
            else:
                 last_actual_price_original = np.nan
                 logger.error(f"Критично: Не удалось получить последнее значение '{model_target_feature_original}'.")
                 return None
        else:
            last_actual_price_original = df_with_features[model_target_feature_original].iloc[-1]

        last_known_original_target_for_diff_restore = None
        if model_used_differencing:
            # Это значение понадобится в PredictionWorker для начала восстановления абсолютных цен
            last_known_original_target_for_diff_restore = last_actual_price_original
            logger.debug(f"Сохранено последнее известное значение '{model_target_feature_original}' ({last_known_original_target_for_diff_restore:.4f}) для восстановления из разностей.")

        # Проверка на достаточность данных для формирования ОДНОЙ последовательности
        if len(df_with_features) < config.TIME_STEPS:
             logger.error(f"Недостаточно данных ({len(df_with_features)} строк) после всех преобразований для "
                          f"входной последовательности (требуется {config.TIME_STEPS} строк).")
             return None

        features_df_pred = df_with_features[config.FEATURES].copy()
        last_sequence_features = features_df_pred.tail(config.TIME_STEPS)
        
        if len(last_sequence_features) < config.TIME_STEPS:
            logger.error(f"Не удалось получить {config.TIME_STEPS} строк для входной последовательности. Получено: {len(last_sequence_features)}")
            return None

        logger.debug(f"Взяты последние {config.TIME_STEPS} записей признаков для предсказания. "
                     f"Последняя актуальная цена ({model_target_feature_original}): {last_actual_price_original:.2f}")

        scaled_last_sequence_features = feature_scaler.transform(last_sequence_features)
        input_sequence = np.reshape(scaled_last_sequence_features, (1, config.TIME_STEPS, len(config.FEATURES)))
        logger.info(f"Финальная форма входной последовательности для предсказания: {input_sequence.shape}")

        # df_with_features возвращается для построения графика истории
        # last_actual_price_original - для отображения "цена до прогноза"
        # last_known_original_target_for_diff_restore - для PredictionWorker, если он восстанавливает разности
        return (input_sequence, last_actual_price_original, df_with_features, 
                last_known_original_target_for_diff_restore)

    except Exception as e:
        logger.error(f"Критическая ошибка в preprocess_for_prediction: {e}", exc_info=config.DEBUG_MODE)
        return None
