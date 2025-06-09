# app/ui/prediction_tab.py

import os
import json
import logging
import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QComboBox, QCheckBox, QSizePolicy, QSpacerItem, QTextEdit
)
from PySide6.QtCore import Slot, Qt, QThreadPool, Signal
from PySide6.QtWebEngineWidgets import QWebEngineView

from app.core.workers import PredictionWorker
from app.utils import config
from app.plotting import plot_generator

logger = logging.getLogger(__name__)

class PredictionTab(QWidget):
    status_update = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active_model_name = None
        self._active_model_display_name = "Нет"
        self._active_model_dir = None
        self._active_model_prediction_horizon = 0 

        self._last_predicted_values_list = None
        self._last_df_processed_history = None
        self._last_ticker_predicted = None

        self._init_ui()
        self._connect_signals()
        self._populate_model_list() 
        self._update_ui_state()
        logger.debug("PredictionTab инициализирована.")

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # --- Секция 1: Выбор Модели ---
        model_group = QGroupBox("1. Выбор Модели")
        model_layout = QVBoxLayout(model_group)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(QLabel("Активная модель:"))
        self.active_model_label = QLabel(self._active_model_display_name)
        self.active_model_label.setStyleSheet("font-weight: bold; color: #AAAAAA;")
        self.active_model_label.setToolTip("Имя активной модели для прогнозирования.")
        hbox1.addWidget(self.active_model_label, 1)

        self.active_model_horizon_label = QLabel("") 
        self.active_model_horizon_label.setStyleSheet("color: #AAAAAA; font-style: italic;")
        self.active_model_horizon_label.setToolTip("Горизонт прогнозирования активной модели.")
        hbox1.addWidget(self.active_model_horizon_label)
        model_layout.addLayout(hbox1)

        hbox2 = QHBoxLayout()
        self.model_combobox = QComboBox()
        self.model_combobox.setToolTip("Выберите сохраненную модель из списка.")
        self.model_combobox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.load_model_button = QPushButton("Загрузить Выбранную")
        self.load_model_button.setToolTip("Загрузить модель, выбранную в списке, и сделать ее активной.")
        hbox2.addWidget(QLabel("Сохраненные модели:"))
        hbox2.addWidget(self.model_combobox, 1)
        hbox2.addWidget(self.load_model_button)
        model_layout.addLayout(hbox2)
        main_layout.addWidget(model_group)

        # --- Секция 2: Выполнение Прогноза ---
        predict_group = QGroupBox("2. Прогноз")
        predict_layout = QVBoxLayout(predict_group)

        hbox3 = QHBoxLayout()
        self.predict_button = QPushButton("Сделать Прогноз")
        self.predict_button.setToolTip("Выполнить прогноз с использованием активной модели.")
        self.predict_status_label = QLabel("Модель не активна")
        self.predict_status_label.setObjectName("StatusLabel")
        hbox3.addWidget(self.predict_button)
        hbox3.addWidget(self.predict_status_label, 1)
        predict_layout.addLayout(hbox3)

        self.prediction_output_display = QTextEdit() 
        self.prediction_output_display.setReadOnly(True)
        self.prediction_output_display.setObjectName("PredictionTextEdit") 
        self.prediction_output_display.setVisible(False)
        self.prediction_output_display.setMinimumHeight(80)
        self.prediction_output_display.setMaximumHeight(150)
        predict_layout.addWidget(self.prediction_output_display)
        main_layout.addWidget(predict_group)

        # --- Секция 3: График и Анализ ---
        analysis_group = QGroupBox("3. Анализ")
        analysis_layout = QVBoxLayout(analysis_group)

        indicators_layout = QHBoxLayout()
        indicators_layout.addWidget(QLabel("Показать на графике:"))
        self.cb_sma_20 = QCheckBox("SMA(20)")
        self.cb_sma_50 = QCheckBox("SMA(50)")
        self.cb_bb = QCheckBox("Bollinger Bands(20)")
        self.cb_rsi = QCheckBox("RSI(14)")
        indicators_layout.addWidget(self.cb_sma_20)
        indicators_layout.addWidget(self.cb_sma_50)
        indicators_layout.addWidget(self.cb_bb)
        indicators_layout.addWidget(self.cb_rsi)
        indicators_layout.addStretch(1)
        analysis_layout.addLayout(indicators_layout)

        self.prediction_plot_webview = QWebEngineView()
        self.prediction_plot_webview.setMinimumHeight(350)
        self.prediction_plot_webview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        analysis_layout.addWidget(self.prediction_plot_webview)
        main_layout.addWidget(analysis_group)

        main_layout.addStretch(1)
        logger.debug("UI PredictionTab создан.")

    def _connect_signals(self):
        self.load_model_button.clicked.connect(self._on_load_selected_model_clicked)
        self.predict_button.clicked.connect(self._on_predict_clicked)
        for cb in [self.cb_sma_20, self.cb_sma_50, self.cb_bb, self.cb_rsi]:
            cb.stateChanged.connect(self._on_indicator_toggled)
        self.model_combobox.currentIndexChanged.connect(self._on_model_selection_changed_in_combobox)
        logger.debug("Сигналы PredictionTab подключены.")

    def _update_ui_state(self, is_predicting=False):
        model_is_truly_active = self._active_model_dir is not None and self._active_model_prediction_horizon > 0
        data_for_redraw_exists = (self._last_df_processed_history is not None and
                                  self._last_predicted_values_list is not None)

        self.model_combobox.setEnabled(not is_predicting)
        
        # Логика для кнопки "Загрузить выбранную"
        can_load_selected = False
        if self.model_combobox.count() > 0 and not is_predicting:
            selected_model_data = self.model_combobox.currentData()
            if selected_model_data:
                # Активна, если выбранная модель не является уже активной
                # или если вообще нет активной модели.
                if not model_is_truly_active or selected_model_data.get('dir') != self._active_model_dir:
                    can_load_selected = True
        self.load_model_button.setEnabled(can_load_selected)

        self.predict_button.setEnabled(model_is_truly_active and not is_predicting)

        if not model_is_truly_active:
            self._set_predict_status("Модель не активна или неверна.", error=False, is_neutral=True)
            self.prediction_output_display.setVisible(False)
            self.active_model_label.setText("Нет")
            self.active_model_label.setStyleSheet("font-weight: bold; color: #AAAAAA;")
            self.active_model_horizon_label.setText("")
        elif is_predicting:
            self._set_predict_status("Выполнение прогноза...", error=False)

        checkboxes_enabled = (data_for_redraw_exists or model_is_truly_active) and not is_predicting
        for cb in [self.cb_sma_20, self.cb_sma_50, self.cb_bb, self.cb_rsi]:
            cb.setEnabled(checkboxes_enabled)
        
        if model_is_truly_active: # Обновляем лейблы активной модели
            self.active_model_label.setText(self._active_model_display_name)
            self.active_model_label.setStyleSheet("font-weight: bold; color: #00AEEF;")
            self.active_model_horizon_label.setText(f"(Прогноз на {self._active_model_prediction_horizon} дн.)")
            self.active_model_horizon_label.setStyleSheet("color: #00AEEF; font-style: italic;")
        
        logger.debug(f"UI state updated: is_predicting={is_predicting}, model_truly_active={model_is_truly_active}, data_for_redraw={data_for_redraw_exists}")

    def _populate_model_list(self):
        current_active_model_dir = self._active_model_dir
        self.model_combobox.clear()
        
        try:
            if not os.path.exists(config.MODELS_DIR):
                logger.warning(f"Директория моделей {config.MODELS_DIR} не найдена.")
                self._update_ui_state()
                return

            model_dirs = [d for d in os.listdir(config.MODELS_DIR) if os.path.isdir(os.path.join(config.MODELS_DIR, d))]
            valid_models_data = []

            for model_name_dir in sorted(model_dirs, reverse=True): 
                 model_full_dir_path = os.path.join(config.MODELS_DIR, model_name_dir)
                 model_file_path = os.path.join(model_full_dir_path, 'bilstm_predictor.keras')
                 scaler_path = os.path.join(model_full_dir_path, 'target_scaler.pkl')
                 metadata_path = os.path.join(model_full_dir_path, 'metadata.json')

                 if os.path.exists(model_file_path) and os.path.exists(scaler_path) and os.path.exists(metadata_path):
                     try:
                         with open(metadata_path, 'r') as f:
                             metadata = json.load(f)
                         horizon = metadata.get('prediction_horizon')
                         stored_model_name = metadata.get('model_name', model_name_dir)

                         if isinstance(horizon, int) and horizon > 0:
                             valid_models_data.append({
                                 'name_in_dir': model_name_dir,
                                 'display_name': f"{stored_model_name} (H: {horizon})", # Для отображения в комбобоксе
                                 'dir': model_full_dir_path,
                                 'horizon': horizon
                             })
                         else:
                             logger.warning(f"Модель '{model_name_dir}': некорректный 'prediction_horizon' ({horizon}) в metadata.json. Пропускаем.")
                     except Exception as e:
                         logger.warning(f"Ошибка чтения metadata.json для '{model_name_dir}': {e}. Пропускаем.")
                 else:
                     logger.debug(f"Директория '{model_name_dir}' не содержит полный набор файлов. Пропускаем.")
            
            if valid_models_data:
                for model_data in valid_models_data:
                    self.model_combobox.addItem(model_data['display_name'], userData=model_data) 
                logger.info(f"Загружен список моделей: {len(valid_models_data)} шт.")
                
                # Пытаемся восстановить выбор, если активная модель все еще в списке
                if current_active_model_dir:
                    for i in range(self.model_combobox.count()):
                        if self.model_combobox.itemData(i).get('dir') == current_active_model_dir:
                            self.model_combobox.setCurrentIndex(i)
                            break
            else:
                logger.info("Сохраненные валидные модели не найдены.")
        except Exception as e:
            logger.error(f"Ошибка при сканировании директории моделей: {e}", exc_info=config.DEBUG_MODE)
        
        self._update_ui_state()


    def _extract_ticker_from_model_name(self, model_name_in_dir: str) -> str:
        try:
            # Ищем первую часть до "_h" или, если нет, до первого "_"
            if "_h" in model_name_in_dir:
                return model_name_in_dir.split('_h')[0].upper()
            return model_name_in_dir.split('_')[0].upper()
        except (IndexError, AttributeError):
            logger.warning(f"Не удалось извлечь тикер из имени директории модели: '{model_name_in_dir}'. Возвращено 'UNKNOWN'.")
            return "UNKNOWN"


    @Slot(int)
    def _on_model_selection_changed_in_combobox(self, index: int):
        logger.debug(f"Выбор в ComboBox изменен, индекс: {index}. Обновление UI.")
        self._update_ui_state(is_predicting=False)

    @Slot(str, str) # model_name_from_signal (имя директории), model_dir (полный путь)
    def set_active_model(self, model_name_from_signal: str, model_dir: str):
        logger.info(f"Установка активной модели (из сигнала): '{model_name_from_signal}' из '{model_dir}'")
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        horizon = 0
        actual_display_name = model_name_from_signal 

        if not os.path.exists(metadata_path):
            err_msg = f"metadata.json не найден для '{model_name_from_signal}'"
            logger.error(f"Критично: Файл {err_msg} в '{model_dir}'. Модель не может быть активирована.")
            self.status_update.emit(f"Ошибка: {err_msg}.")
            self._active_model_name = None
            self._active_model_display_name = "Ошибка метаданных"
            self._active_model_dir = None
            self._active_model_prediction_horizon = 0
            self._update_ui_state()
            return
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            h = metadata.get('prediction_horizon')
            if not (isinstance(h, int) and h > 0):
                err_msg = f"модель '{model_name_from_signal}' имеет некорректный 'prediction_horizon' ({h}) в metadata.json"
                logger.error(f"Ошибка активации: {err_msg}.")
                self.status_update.emit(f"Ошибка: {err_msg}.")
                self._active_model_name = None
                self._active_model_display_name = "Ошибка метаданных"
                self._active_model_dir = None
                self._active_model_prediction_horizon = 0
                self._update_ui_state()
                return
            horizon = h
            
            model_found_in_list = False
            for i in range(self.model_combobox.count()):
                item_data = self.model_combobox.itemData(i)
                if item_data and item_data.get('dir') == model_dir:
                    actual_display_name = self.model_combobox.itemText(i)
                    self.model_combobox.setCurrentIndex(i)
                    model_found_in_list = True
                    break
            
            if not model_found_in_list: 
                self._populate_model_list() 
                for i in range(self.model_combobox.count()): 
                    item_data = self.model_combobox.itemData(i)
                    if item_data and item_data.get('dir') == model_dir:
                        actual_display_name = self.model_combobox.itemText(i)
                        self.model_combobox.setCurrentIndex(i)
                        break
        
        except Exception as e:
            err_msg = f"Ошибка чтения metadata.json для '{model_name_from_signal}': {e}"
            logger.error(err_msg, exc_info=True)
            self.status_update.emit(f"Ошибка: не удалось прочитать метаданные модели.")
            self._active_model_name = None
            self._active_model_display_name = "Ошибка метаданных"
            self._active_model_dir = None
            self._active_model_prediction_horizon = 0
            self._update_ui_state()
            return

        self._active_model_name = model_name_from_signal 
        self._active_model_display_name = actual_display_name
        self._active_model_dir = model_dir
        self._active_model_prediction_horizon = horizon
        
        self.prediction_output_display.setVisible(False)
        self.prediction_output_display.clear()
        self.prediction_plot_webview.setHtml("")
        self._last_predicted_values_list = None
        self._last_df_processed_history = None
        self._last_ticker_predicted = None

        self._update_ui_state() 
        self.status_update.emit(f"Активная модель: {self._active_model_display_name}")

    @Slot()
    def _on_load_selected_model_clicked(self):
        selected_index = self.model_combobox.currentIndex()
        if selected_index == -1:
            self.status_update.emit("Модель для загрузки не выбрана.")
            return
        
        model_data = self.model_combobox.itemData(selected_index)
        if model_data and isinstance(model_data, dict):
            model_dir_name = model_data.get('name_in_dir')
            model_full_dir = model_data.get('dir')
            
            if model_dir_name and model_full_dir:
                logger.info(f"Загрузка выбранной модели: '{model_dir_name}' из '{model_full_dir}'")
                self.set_active_model(model_dir_name, model_full_dir)
            else:
                logger.error("Ошибка: неполные данные (name_in_dir или dir) для выбранной модели в ComboBox.")
                self.status_update.emit("Ошибка данных выбранной модели.")
        else:
            logger.error(f"Ошибка: некорректные userData для выбранной модели в ComboBox (index {selected_index}).")
            self.status_update.emit("Ошибка получения данных модели.")

    def _get_current_indicators_state(self) -> dict:
        return {
            'show_sma_20': self.cb_sma_20.isChecked(),
            'show_sma_50': self.cb_sma_50.isChecked(),
            'show_bb': self.cb_bb.isChecked(),
            'show_rsi': self.cb_rsi.isChecked()
        }

    @Slot()
    def _on_model_selection_changed_in_combobox(self, index: int):
        """Обновляет состояние кнопки 'Загрузить Выбранную' при изменении выбора в ComboBox."""
        self._update_ui_state()


    @Slot()
    def _on_predict_clicked(self):
        if not self._active_model_dir or self._active_model_prediction_horizon == 0:
            self._set_predict_status("Ошибка: Модель не активна или неверна.", error=True)
            return

        ticker_from_dir = self._extract_ticker_from_model_name(os.path.basename(self._active_model_dir))
        
        if not ticker_from_dir or ticker_from_dir == "UNKNOWN":
             self._set_predict_status("Ошибка: Не удалось определить тикер из имени активной модели.", error=True)
             return

        logger.info(f"Запуск прогнозирования для тикера '{ticker_from_dir}' с моделью '{self._active_model_display_name}' "
                    f"(горизонт {self._active_model_prediction_horizon} дн.).")
        
        self._last_predicted_values_list = None
        self._last_df_processed_history = None
        self._last_ticker_predicted = None
        self.prediction_plot_webview.setHtml("")

        current_indicators_state = self._get_current_indicators_state()

        self._update_ui_state(is_predicting=True)
        self._set_predict_status(f"Запрос прогноза для {ticker_from_dir}...", error=False)
        self.status_update.emit(f"Запрос прогноза для {ticker_from_dir} ({self._active_model_display_name})...")

        worker = PredictionWorker(ticker_from_dir, self._active_model_dir, current_indicators_state)
        worker.signals.prediction_ready.connect(self._handle_prediction_ready)
        worker.signals.error.connect(self._handle_worker_error)
        worker.signals.finished.connect(lambda: self._update_ui_state(is_predicting=False))
        QThreadPool.globalInstance().start(worker)

    @Slot()
    def _on_indicator_toggled(self):
        logger.debug(f"Изменение состояния индикатора. Sender: {self.sender().text() if self.sender() else 'N/A'}")
        # Условие для перерисовки: есть активная модель И есть данные последнего прогноза
        if self._active_model_dir and self._active_model_prediction_horizon > 0 and \
           self._last_predicted_values_list is not None and \
           self._last_df_processed_history is not None and \
           self._last_ticker_predicted:

            if self.predict_button.text() == "Отменить Прогноз" or not self.predict_button.isEnabled():
                 logger.debug("Процесс предсказания или загрузки модели активен, перерисовка графика отложена.")
                 return

            logger.info("Перерисовка графика с обновленными индикаторами (данные из кеша)...")
            self.status_update.emit("Обновление отображения индикаторов на графике...")
            current_indicators_state = self._get_current_indicators_state()

            plot_html = plot_generator.create_prediction_plot(
                self._last_df_processed_history,
                self._last_predicted_values_list, 
                self._last_ticker_predicted,
                **current_indicators_state
            )
            if plot_html:
                self.prediction_plot_webview.setHtml(plot_html)
                self.status_update.emit("График индикаторов обновлен.")
                logger.debug("График успешно перерисован с новыми индикаторами.")
            else:
                self.status_update.emit("Ошибка обновления графика индикаторов.")
                logger.warning("Не удалось перерисовать график с новыми индикаторами (plot_generator вернул None).")
        else:
            logger.debug("Нет сохраненных данных для перерисовки графика. Полный прогноз не был выполнен или модель не активна.")


    @Slot(list, str, float, pd.DataFrame, str) 
    def _handle_prediction_ready(self, list_of_predictions: list, plot_html: str, 
                                 last_actual_price: float, df_history: pd.DataFrame, ticker: str):
        
        if not list_of_predictions: 
            logger.error(f"Получен пустой список предсказаний для {ticker} в _handle_prediction_ready.")
            self._handle_worker_error(f"Внутренняя ошибка: получен пустой список предсказаний для {ticker}.")
            return

        logger.info(f"Получен готовый прогноз для {ticker} на {len(list_of_predictions)} дней. "
                    f"Первый день: {list_of_predictions[0]:.2f}, Последний день: {list_of_predictions[-1]:.2f}")
        
        self._last_predicted_values_list = list_of_predictions
        self._last_df_processed_history = df_history 
        self._last_ticker_predicted = ticker

        prediction_text_lines = [f"<b>Прогноз на {len(list_of_predictions)} дней для {ticker.upper()}:</b>"]
        
        last_history_date = df_history.index[-1] if not df_history.empty else pd.Timestamp.today().normalize()
        prediction_dates = pd.bdate_range(start=last_history_date + pd.tseries.offsets.BDay(1), 
                                          periods=len(list_of_predictions))

        for i, pred_price in enumerate(list_of_predictions):
            date_str = prediction_dates[i].strftime('%Y-%m-%d')
            prediction_text_lines.append(f"{date_str}: {pred_price:.2f} RUB")
        
        prediction_text_lines.append(f"<br><i>Последняя актуальная цена ({last_history_date.strftime('%Y-%m-%d')}): {last_actual_price:.2f} RUB</i>")

        self.prediction_output_display.setHtml("<br>".join(prediction_text_lines))
        self.prediction_output_display.setVisible(True)

        status_msg_short = f"Прогноз для {ticker} на {len(list_of_predictions)} дн. (до {prediction_dates[-1].strftime('%y-%m-%d')}) " \
                           f"посл. {list_of_predictions[-1]:.2f} RUB"
        self._set_predict_status("Готово.", error=False)
        self.status_update.emit(status_msg_short)

        if plot_html:
            self.prediction_plot_webview.setHtml(plot_html)
        else:
            logger.warning("Получен пустой HTML для графика прогноза.")
            self.prediction_plot_webview.setHtml(
                "<body style='background-color:#1E1E1E; color:#E74C3C; font-family: sans-serif; "
                "text-align: center; padding-top: 50px;'>"
                "<h2>Ошибка генерации графика прогноза</h2>"
                "<p>Предсказания были получены, но график не удалось построить. "
                "Проверьте логи для получения дополнительной информации.</p></body>"
            )

        self._update_ui_state()

    @Slot(str)
    def _handle_worker_error(self, error_message: str):
        logger.error(f"Ошибка от PredictionWorker: {error_message}")
        self._set_predict_status(f"Ошибка: {error_message}", error=True)
        self.status_update.emit(f"Ошибка прогноза: {error_message}")
        
        self.prediction_output_display.setVisible(False)
        self.prediction_output_display.clear()

        self._last_predicted_values_list = None
        self._last_df_processed_history = None
        self._last_ticker_predicted = None
        self.prediction_plot_webview.setHtml("<body style='background-color:#1E1E1E; color:#AAAAAA; font-family: sans-serif; text-align: center; padding-top: 50px;'><h2>Ошибка получения прогноза</h2><p>Подробности смотрите в логах.</p></body>")
        self._update_ui_state(is_predicting=False)

    def _set_predict_status(self, message: str, error: bool = False, is_neutral: bool = False):
        self.predict_status_label.setText(message)
        if error:
            self.predict_status_label.setObjectName("ErrorLabel")
            self.predict_status_label.setStyleSheet("QLabel#ErrorLabel { color: #E74C3C; font-weight: bold; font-size: 9pt; }")
        elif is_neutral:
            self.predict_status_label.setObjectName("StatusLabel")
            self.predict_status_label.setStyleSheet("QLabel#StatusLabel { color: #AAAAAA; font-size: 9pt; }")
        else: 
            self.predict_status_label.setObjectName("StatusLabel")
            self.predict_status_label.setStyleSheet("QLabel#StatusLabel { color: #00AEEF; font-size: 9pt; }")
        logger.debug(f"Статус прогноза обновлен: '{message}', ошибка={error}, нейтральный={is_neutral}")
