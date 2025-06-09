# app/utils/config.py

import os
import logging # Используется для LOG_LEVEL и DEBUG_MODE

# --- Базовые Настройки ---
APP_NAME = "Prognostr"
# Уровень логирования: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL" (строкой)
LOG_LEVEL = "INFO"
DEBUG_MODE = (str(LOG_LEVEL).upper() == "DEBUG")


# --- Пути ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
MODELS_DIR = os.path.join(ASSETS_DIR, 'models')
DATA_DIR = os.path.join(ASSETS_DIR, 'data')

# Путь к UI ресурсам
STYLES_PATH = os.path.join(BASE_DIR, 'app', 'ui', 'styles.qss') # Явное указание

# LOG_FILE_PATH = os.path.join(BASE_DIR, 'app.log') # Опционально, если логирование в файл

# --- Настройки Модели и Данных ---

# --- Настройка для дифференцирования ---
USE_DIFFERENCING = True # Поставить False, чтобы использовать старую логику с абсолютными ценами
DIFFERENCING_COLUMN = 'Close' # Колонка, которую будем дифференцировать
DIFFERENCING_ORDER = 1       # Порядок дифференцирования (1 для Close_t - Close_t-1)
# ---------------------------------------------

# Признаки, которые будут использоваться моделью.
# Если USE_DIFFERENCING = True, TARGET_FEATURE будет изменен на имя колонки с разностями.
FEATURES = [
    'Close', 'Volume',
    'SMA_20', 'SMA_50',
    'BB_Middle_20', 'BB_Upper_20', 'BB_Lower_20',
    'RSI_14'
]
TARGET_FEATURE_ORIGINAL = 'Close' # Исходный целевой признак (абсолютное значение)

if USE_DIFFERENCING:
    TARGET_FEATURE = f'{DIFFERENCING_COLUMN}_diff{DIFFERENCING_ORDER}'
else:
    TARGET_FEATURE = TARGET_FEATURE_ORIGINAL

# Конфигурация технических индикаторов для расчета (в preprocessor.py)
INDICATORS_CONFIG = {
    'SMA_20': {'window': 20, 'column': 'Close'},
    'SMA_50': {'window': 50, 'column': 'Close'},
    'BollingerBands': {'window': 20, 'std': 2, 'column': 'Close'},
    'RSI_14': {'window': 14, 'column': 'Close'}
}

# --- Параметры Обучения по Умолчанию ---
TIME_STEPS = 30 # Длина последовательности
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_TEST_SPLIT = 0.2
DEFAULT_PREDICTION_HORIZON = 7
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MONITOR = 'val_loss'

# --- Параметры для кастомной функции потерь и модели ---
LOSS_WEIGHT_BASE = 1.15  # База для экспоненциального роста весов в функции потерь
DROPOUT_RATE = 0.1       # Единая ставка Dropout для всех слоев

# --- Настройки Интерфейса ---
WINDOW_TITLE = f"{APP_NAME} (BiLSTM Stock Predictor)"
DEFAULT_TICKER = "SBER"
DEFAULT_START_DATE = "2018-01-01"
HISTORY_DAYS_TO_PLOT = 180 # Дней истории на графике прогноза


# --- настройки для ARIMA ---
ARIMA_DEFAULT_ORDER = (5, 1, 0)  # Стандартные параметры p, d, q
ARIMA_MIN_TRAIN_SIZE = 60        # Минимальный размер начального окна обучения для Walk-Forward
ARIMA_MIN_TRAIN_SIZE_ITER = 30   # Минимальный размер окна на каждой итерации (чтобы избежать ошибок)


# Минимальное количество ТОРГОВЫХ дней данных, необходимое для:
# 1. Расчета самого "длинного" индикатора из INDICATORS_CONFIG (max_window).
# 2. Формирования ОДНОЙ входной последовательности для LSTM (TIME_STEPS).
# 3. Небольшого запаса на случай пропусков данных или неторговых дней внутри периода.
# 4. Учета дифференцирования (теряется `DIFFERENCING_ORDER` строк)
_max_indicator_window = 0
if INDICATORS_CONFIG:
    _max_indicator_window = max(params.get('window', 0) for params in INDICATORS_CONFIG.values() if isinstance(params, dict) and params)

MIN_DAYS_FOR_PREDICTION = _max_indicator_window + TIME_STEPS + (DIFFERENCING_ORDER if USE_DIFFERENCING else 0) + 30 # Запас

# --- Проверка и Создание Папок ---
def check_and_create_dirs():
    """Проверяет наличие и при необходимости создает директории.
    Вызывает SystemExit при критической ошибке создания.
    """
    _local_logger = logging.getLogger(__name__ + ".check_dirs")
    dirs_to_check = [ASSETS_DIR, MODELS_DIR, DATA_DIR]
    _local_logger.info(f"Проверка существования директорий: {dirs_to_check}")
    for d_path in dirs_to_check:
        if not os.path.exists(d_path):
            try:
                os.makedirs(d_path)
                _local_logger.info(f"Создана директория: {d_path}")
            except OSError as e:
                err_msg = f"Критическая ошибка: Не удалось создать директорию {d_path}. Ошибка: {e}"
                _local_logger.critical(err_msg)
                raise SystemExit(err_msg) from e
        elif not os.path.isdir(d_path):
            err_msg = f"Критическая ошибка: Путь {d_path} существует, но не является директорией."
            _local_logger.critical(err_msg)
            raise SystemExit(err_msg)
    _local_logger.info("Все необходимые директории существуют или были успешно созданы.")


# --- Настройки Логирования (константы) ---
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s:%(thread)d] - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

if DEBUG_MODE:
    _debug_logger = logging.getLogger(__name__ + ".config_debug")
    _debug_logger.debug("--- Configuration Loaded (DEBUG_MODE=True) ---")
    _debug_logger.debug(f"BASE_DIR: {BASE_DIR}")
    _debug_logger.debug(f"MODELS_DIR: {MODELS_DIR}")
    _debug_logger.debug(f"DATA_DIR: {DATA_DIR}")
    _debug_logger.debug(f"STYLES_PATH: {STYLES_PATH}")
    _debug_logger.debug(f"LOG_LEVEL (from config string): {LOG_LEVEL}")
    _debug_logger.debug(f"TIME_STEPS: {TIME_STEPS}")
    _debug_logger.debug(f"USE_DIFFERENCING: {USE_DIFFERENCING}")
    _debug_logger.debug(f"DIFFERENCING_COLUMN: {DIFFERENCING_COLUMN}")
    _debug_logger.debug(f"DIFFERENCING_ORDER: {DIFFERENCING_ORDER}")
    _debug_logger.debug(f"FEATURES (input to model): {FEATURES}")
    _debug_logger.debug(f"TARGET_FEATURE (model predicts this): {TARGET_FEATURE}")
    _debug_logger.debug(f"TARGET_FEATURE_ORIGINAL (for plotting/reference): {TARGET_FEATURE_ORIGINAL}")
    _debug_logger.debug(f"MIN_DAYS_FOR_PREDICTION: {MIN_DAYS_FOR_PREDICTION}")
    _debug_logger.debug("--- End Configuration Debug Output ---")