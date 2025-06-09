#main_window.py

import sys
import logging
logger = logging.getLogger(__name__)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QStatusBar
)
from PySide6.QtCore import Qt, Slot, QThreadPool
from PySide6.QtGui import QIcon

from app.ui.training_tab import TrainingTab
from app.ui.prediction_tab import PredictionTab
from app.utils import config


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.WINDOW_TITLE)
        self.setGeometry(100, 100, 1100, 850)

        self._active_model_name = None
        self._active_model_dir = None

        self._load_styles()
        self._init_ui()
        self._connect_signals()

        logging.info("Главное окно инициализировано.")
        self.statusBar().showMessage("Готово к работе.", 5000) # Начальное сообщение в статус-баре

    def _load_styles(self):
        """Загружает стили из QSS файла."""
        try:
            with open(config.STYLES_PATH, "r", encoding="utf-8") as f:
                style_sheet = f.read()
            self.setStyleSheet(style_sheet)
            logging.debug("Стили QSS успешно загружены.")
        except FileNotFoundError:
            logging.warning(f"Файл стилей не найден: {config.STYLES_PATH}.")
        except Exception as e:
             logging.error(f"Ошибка загрузки стилей QSS: {e}", exc_info=config.DEBUG_MODE)

    def _init_ui(self):
        """Инициализирует основной интерфейс с вкладками."""
        # --- Центральный виджет с вкладками ---
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # --- Создание вкладок ---
        self.training_tab = TrainingTab()
        self.prediction_tab = PredictionTab()

        # --- Добавление вкладок ---
        self.tab_widget.addTab(self.training_tab, "Данные и Обучение")
        self.tab_widget.addTab(self.prediction_tab, "Прогнозирование и Анализ")

        # --- Статус-бар ---
        self.setStatusBar(QStatusBar(self))

        logging.debug("UI главного окна инициализирован.")

    def _connect_signals(self):
        """Соединяет сигналы между вкладками и главным окном."""
        # Сигнал из TrainingTab об успешном обучении
        self.training_tab.model_trained.connect(self._on_model_trained_activated)

        # Сигналы обновления статуса из обеих вкладок
        self.training_tab.status_update.connect(self._update_status_bar)
        self.prediction_tab.status_update.connect(self._update_status_bar)

    @Slot(str, str)
    def _on_model_trained_activated(self, model_name, model_dir):
        """Обрабатывает сигнал об обучении новой модели."""
        logging.info(f"Получен сигнал: Новая модель обучена - {model_name}")
        self._active_model_name = model_name
        self._active_model_dir = model_dir
        # Передаем информацию во вкладку прогнозирования
        self.prediction_tab.set_active_model(model_name, model_dir)
        self.statusBar().showMessage(f"Модель '{model_name}' обучена и активна для предсказания.", 5000)

    @Slot(str)
    def _update_status_bar(self, message):
        """Обновляет сообщение в статус-баре."""
        self.statusBar().showMessage(message)

    def closeEvent(self, event):
        """Обработка закрытия окна."""
        logging.info("Запрос на закрытие приложения...")
        QThreadPool.globalInstance().waitForDone(1000) # Ждем до 1 секунды
        logging.info("Приложение закрывается.")
        event.accept()