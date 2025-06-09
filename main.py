# main.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1: filter INFO, 2: filter WARNING, 3: filter ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import logging

os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--disable-gpu'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

initial_logger = logging.getLogger(__name__)
initial_logger.info("Попытка установки переменных окружения для отключения GPU (предварительная настройка логгера).")

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt

from app.ui.main_window import MainWindow
from app.utils import config

# --- Перенастраиваем логгер с уровнем и форматом из config ---
log_level_config = getattr(logging, str(config.LOG_LEVEL).upper(), logging.INFO)
log_format_config = config.LOG_FORMAT
log_date_format_config = config.LOG_DATE_FORMAT

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=log_level_config, format=log_format_config, datefmt=log_date_format_config, stream=sys.stdout)
logger = logging.getLogger(__name__)


# --- Вызов функции проверки и создания директорий ---
try:
    config.check_and_create_dirs()
    logger.info("Проверка и создание директорий успешно завершены.")
except SystemExit as dir_err:
    logger.critical(f"Критическая ошибка при проверке/создании директорий: {dir_err}")
    temp_app_for_msgbox = None
    try:
        temp_app_for_msgbox = QApplication.instance()
        if temp_app_for_msgbox is None:
            temp_args = sys.argv if len(sys.argv) > 0 else [""]
            temp_app_for_msgbox = QApplication(temp_args)
        QMessageBox.critical(None, "Критическая ошибка", f"Не удалось создать необходимые директории.\n{dir_err}\nПриложение будет закрыто.")
    except Exception as msg_ex:
        logger.error(f"Не удалось показать QMessageBox: {msg_ex}")
    finally:
        if temp_app_for_msgbox and not QApplication.instance():
             pass
    sys.exit(1)

if config.DEBUG_MODE:
    logger.debug("--- Режим отладки активен. Вывод конфигурации: ---")
    logger.debug(f"BASE_DIR: {config.BASE_DIR}")
    logger.debug(f"MODELS_DIR: {config.MODELS_DIR}")
    logger.debug(f"DATA_DIR: {config.DATA_DIR}")
    logger.debug(f"LOG_LEVEL: {config.LOG_LEVEL}")

if __name__ == "__main__":
    logger.info(f"Запуск приложения {config.APP_NAME}...")

    app = None
    try:
        app = QApplication.instance()
        if app is None:
            app_args = sys.argv if len(sys.argv) > 0 else [""]
            app = QApplication(app_args)
    except Exception as e_app:
        logger.critical(f"Критическая ошибка при создании QApplication: {e_app}", exc_info=True)
        sys.exit(1)

    try:
        # Установка атрибутов отрисовки для High DPI
        try:
            if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
                app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
            if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
                app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        except AttributeError:
            logger.warning("Атрибуты AA_EnableHighDpiScaling/AA_UseHighDpiPixmaps не найдены (возможно, старая версия Qt?).")
        except RuntimeError as e_attr:
             logger.warning(f"Не удалось установить атрибуты отрисовки High DPI: {e_attr}. Возможно, они уже установлены или не поддерживаются.")

        window = MainWindow()
        window.show()

        logger.info(f"Приложение {config.APP_NAME} успешно запущено. Ожидание действий пользователя.")
        sys.exit(app.exec())

    except Exception as e:
        logger.critical(f"Необработанное исключение на верхнем уровне при запуске основного цикла приложения: {e}", exc_info=True)
        if app:
            QMessageBox.critical(None, "Критическая ошибка", f"Произошла непредвиденная ошибка во время работы приложения:\n{e}\nПриложение будет закрыто.")
        sys.exit(1)