import os
import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
    QRadioButton, QFileDialog, QDateEdit, QSpinBox, QDoubleSpinBox, QProgressBar,
    QTextEdit, QSizePolicy, QSpacerItem, QCheckBox
)
from PySide6.QtCore import QDate, Slot, Signal, QThreadPool, Qt
from PySide6.QtWebEngineWidgets import QWebEngineView

from app.ui.components.mpl_canvas import MplCanvas
from app.core.workers import DataWorker, TrainingWorker
from app.plotting import plot_generator
from app.utils import config


class TrainingTab(QWidget):
    model_trained = Signal(str, str) 
    status_update = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data_df = None
        self._data_path = None
        self._training_history = {}
        self._init_ui()
        self._connect_signals()
        self._update_ui_state()
        logger.debug("TrainingTab инициализирована.")

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)

        # --- Секция 1: Загрузка Данных ---
        data_group = QGroupBox("1. Загрузка Данных")
        data_layout = QVBoxLayout(data_group)

        source_layout = QHBoxLayout()
        self.rb_moex = QRadioButton("Загрузить с MOEX")
        self.rb_csv = QRadioButton("Загрузить из CSV")
        self.rb_moex.setChecked(True)
        source_layout.addWidget(self.rb_moex)
        source_layout.addWidget(self.rb_csv)
        data_layout.addLayout(source_layout)

        self.moex_widget = QWidget()
        moex_layout = QHBoxLayout(self.moex_widget)
        moex_layout.setContentsMargins(0,0,0,0)
        self.ticker_input = QLineEdit(config.DEFAULT_TICKER)
        self.ticker_input.setPlaceholderText("Тикер (напр. SBER)")
        self.start_date_edit = QDateEdit(QDate.fromString(config.DEFAULT_START_DATE, "yyyy-MM-dd"))
        self.start_date_edit.setCalendarPopup(True)
        self.end_date_edit = QDateEdit(QDate.currentDate())
        self.end_date_edit.setCalendarPopup(True)
        self.load_moex_button = QPushButton("Скачать и Сохранить")
        moex_layout.addWidget(QLabel("Тикер:"))
        moex_layout.addWidget(self.ticker_input)
        moex_layout.addWidget(QLabel("Начало:"))
        moex_layout.addWidget(self.start_date_edit)
        moex_layout.addWidget(QLabel("Конец:"))
        moex_layout.addWidget(self.end_date_edit)
        moex_layout.addWidget(self.load_moex_button)
        data_layout.addWidget(self.moex_widget)

        self.csv_widget = QWidget()
        csv_layout = QHBoxLayout(self.csv_widget)
        csv_layout.setContentsMargins(0,0,0,0)
        self.csv_path_label = QLabel("Файл не выбран")
        self.csv_path_label.setStyleSheet("color: #AAAAAA;")
        self.load_csv_button = QPushButton("Выбрать CSV...")
        csv_layout.addWidget(QLabel("Путь к файлу:"))
        csv_layout.addWidget(self.csv_path_label, 1)
        csv_layout.addWidget(self.load_csv_button)
        self.csv_widget.setVisible(False) 
        data_layout.addWidget(self.csv_widget)

        self.data_status_label = QLabel("Данные не загружены")
        self.data_status_label.setObjectName("StatusLabel") 
        data_layout.addWidget(self.data_status_label)
        main_layout.addWidget(data_group)

        # --- Секция 2: Параметры Обучения ---
        params_group = QGroupBox("2. Параметры Обучения")
        params_outer_layout = QVBoxLayout(params_group)

        params_layout_row1 = QHBoxLayout() 
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 1000)
        self.epochs_spinbox.setValue(config.DEFAULT_EPOCHS)
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setRange(1, 1024) 
        self.batch_spinbox.setValue(config.DEFAULT_BATCH_SIZE)
        
        params_layout_row1.addWidget(QLabel("Эпохи:"))
        params_layout_row1.addWidget(self.epochs_spinbox)
        params_layout_row1.addWidget(QLabel("Размер батча:"))
        params_layout_row1.addWidget(self.batch_spinbox)

        self.test_split_spinbox = QDoubleSpinBox()
        self.prediction_horizon_spinbox = QSpinBox()
        self.prediction_horizon_spinbox.setRange(1, 30) 
        self.prediction_horizon_spinbox.setValue(getattr(config, 'DEFAULT_PREDICTION_HORIZON', 1)) 
        
        params_layout_row1.addWidget(QLabel("Доля теста (%):")) 
        self.test_split_spinbox.setSuffix(" %") 
        self.test_split_spinbox.setDecimals(0) 
        self.test_split_spinbox.setRange(5, 50) 
        self.test_split_spinbox.setValue(int(config.DEFAULT_TEST_SPLIT * 100))

        params_layout_row1.addWidget(self.test_split_spinbox)
        params_layout_row1.addWidget(QLabel("Длина прогноза (дней):"))
        params_layout_row1.addWidget(self.prediction_horizon_spinbox)
        
        self.cb_use_differencing = QCheckBox("Использовать дифференцирование цен")
        self.cb_use_differencing.setToolTip(
            f"Если отмечено, модель будет обучаться на разностях цен колонки '{config.DIFFERENCING_COLUMN}' (порядок {config.DIFFERENCING_ORDER}).\n"
            f"Целевым признаком для модели станет '{config.DIFFERENCING_COLUMN}_diff{config.DIFFERENCING_ORDER}'.\n"
            "Это может помочь модели лучше улавливать изменения, а не абсолютные уровни."
        )
        self.cb_use_differencing.setChecked(config.USE_DIFFERENCING)
        params_layout_row1.addWidget(self.cb_use_differencing) 
        
        params_layout_row1.addStretch(1)
        params_outer_layout.addLayout(params_layout_row1)
        

        main_layout.addWidget(params_group)

        # --- Секция 3: Обучение Модели ---
        training_group = QGroupBox("3. Обучение Модели")
        training_layout = QVBoxLayout(training_group)

        button_progress_layout = QHBoxLayout()
        self.start_training_button = QPushButton("Начать Обучение")
        button_progress_layout.addWidget(self.start_training_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button_progress_layout.addWidget(self.progress_bar, 1)
        training_layout.addLayout(button_progress_layout)

        log_plot_layout = QHBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(150)
        self.log_output.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        log_plot_layout.addWidget(self.log_output, 1)

        self.training_plot_canvas = MplCanvas(self, width=5, height=3, dpi=100)
        self.training_plot_canvas.setMinimumHeight(250)
        self.training_plot_canvas.setMinimumWidth(350)
        self.training_plot_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        log_plot_layout.addWidget(self.training_plot_canvas, 1)
        training_layout.addLayout(log_plot_layout)
        main_layout.addWidget(training_group)

        # --- Секция 4: Результат Обучения ---
        result_group = QGroupBox("4. Результат Обучения (Actual vs Predicted на тесте)")
        result_layout = QVBoxLayout(result_group)
        self.eval_plot_webview = QWebEngineView()
        self.eval_plot_webview.setMinimumHeight(250)
        self.eval_plot_webview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.eval_plot_webview.setVisible(False) 
        result_layout.addWidget(self.eval_plot_webview)
        main_layout.addWidget(result_group)

        main_layout.addStretch(1) 
        logger.debug("UI TrainingTab создан.")

    def _connect_signals(self):
        self.rb_moex.toggled.connect(self._on_source_toggled)
        self.rb_csv.toggled.connect(self._on_source_toggled)
        self.load_moex_button.clicked.connect(self._on_load_moex_clicked)
        self.load_csv_button.clicked.connect(self._on_load_csv_clicked)
        self.start_training_button.clicked.connect(self._on_start_training_clicked)
        logger.debug("Сигналы TrainingTab подключены.")

    def _update_ui_state(self, is_loading=False, is_training=False):
        data_loaded = self._data_df is not None

        self.rb_moex.setEnabled(not is_loading and not is_training)
        self.rb_csv.setEnabled(not is_loading and not is_training)
        self.ticker_input.setEnabled(self.rb_moex.isChecked() and not is_loading and not is_training)
        self.start_date_edit.setEnabled(self.rb_moex.isChecked() and not is_loading and not is_training)
        self.end_date_edit.setEnabled(self.rb_moex.isChecked() and not is_loading and not is_training)
        self.load_moex_button.setEnabled(self.rb_moex.isChecked() and not is_loading and not is_training)
        self.load_csv_button.setEnabled(self.rb_csv.isChecked() and not is_loading and not is_training)

        self.epochs_spinbox.setEnabled(not is_training)
        self.batch_spinbox.setEnabled(not is_training)
        self.test_split_spinbox.setEnabled(not is_training)
        self.prediction_horizon_spinbox.setEnabled(not is_training)
        self.cb_use_differencing.setEnabled(not is_training)


        self.start_training_button.setEnabled(data_loaded and not is_training and not is_loading)
        self.progress_bar.setVisible(is_training)
        if not is_training:
            self.progress_bar.setValue(0)
        logger.debug(f"UI state updated: is_loading={is_loading}, is_training={is_training}, data_loaded={data_loaded}")

    @Slot(bool)
    def _on_source_toggled(self, checked):
        sender = self.sender()
        if sender == self.rb_moex and checked:
            self.moex_widget.setVisible(True)
            self.csv_widget.setVisible(False)
        elif sender == self.rb_csv and checked:
            self.moex_widget.setVisible(False)
            self.csv_widget.setVisible(True)
        self._update_ui_state()

    @Slot()
    def _on_load_moex_clicked(self):
        ticker = self.ticker_input.text().strip().upper()
        start_date = self.start_date_edit.date().toString("yyyy-MM-dd")
        end_date = self.end_date_edit.date().toString("yyyy-MM-dd")

        if not ticker:
            self._set_data_status("Ошибка: Введите тикер.", error=True)
            return

        self._update_ui_state(is_loading=True)
        self._set_data_status(f"Загрузка данных MOEX для {ticker}...")
        self.status_update.emit(f"Загрузка данных MOEX для {ticker}...")
        self.eval_plot_webview.setVisible(False)

        worker = DataWorker(source_type='moex', ticker=ticker, start_date=start_date, end_date=end_date)
        worker.signals.data_ready.connect(self._handle_data_ready)
        worker.signals.error.connect(self._handle_worker_error)
        worker.signals.finished.connect(lambda: self._update_ui_state(is_loading=False))
        QThreadPool.globalInstance().start(worker)

    @Slot()
    def _on_load_csv_clicked(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Выбрать CSV файл", config.DATA_DIR,
                                                  "CSV Files (*.csv);;All Files (*)", options=options)
        if filePath:
            self.csv_path_label.setText(os.path.basename(filePath))
            self._update_ui_state(is_loading=True)
            self._set_data_status(f"Загрузка данных из {os.path.basename(filePath)}...")
            self.status_update.emit(f"Загрузка данных из {os.path.basename(filePath)}...")
            self.eval_plot_webview.setVisible(False)

            worker = DataWorker(source_type='csv', csv_path=filePath)
            worker.signals.data_ready.connect(self._handle_data_ready)
            worker.signals.error.connect(self._handle_worker_error)
            worker.signals.finished.connect(lambda: self._update_ui_state(is_loading=False))
            QThreadPool.globalInstance().start(worker)

    @Slot()
    def _on_start_training_clicked(self):
        if self._data_df is None:
            self._set_data_status("Ошибка: Данные не загружены для обучения.", error=True)
            return

        epochs = self.epochs_spinbox.value()
        batch_size = self.batch_spinbox.value()
        test_split_percentage = self.test_split_spinbox.value()
        test_split_ratio = test_split_percentage / 100.0
        prediction_horizon = self.prediction_horizon_spinbox.value()
        use_differencing_from_ui = self.cb_use_differencing.isChecked()

        ticker = self.ticker_input.text().strip().upper() if self.rb_moex.isChecked() else self._extract_ticker_from_path(self._data_path)
        if not ticker: 
            ticker = "UNKNOWN_TICKER"
        
        logger.info(f"Запуск обучения для тикера '{ticker}' с параметрами: "
                    f"Эпохи={epochs}, Батч={batch_size}, Доля теста={test_split_ratio:.2f}, "
                    f"Длина прогноза={prediction_horizon} дней, "
                    f"Использовать дифференцирование (UI): {use_differencing_from_ui}.")

        self._update_ui_state(is_training=True)
        diff_status_msg = "с дифференцированием" if use_differencing_from_ui else "на абсолютных значениях"
        self._set_data_status(f"Запуск обучения для {ticker} ({diff_status_msg})...")
        self.status_update.emit(f"Запуск обучения для {ticker} ({diff_status_msg}, прогноз на {prediction_horizon} дней)...")

        self.log_output.clear()
        self.training_plot_canvas.clear_plot()
        self.eval_plot_webview.setVisible(False)
        self._training_history = {}

        worker = TrainingWorker(
            df_train=self._data_df.copy(), 
            epochs=epochs,
            batch_size=batch_size,
            ticker=ticker,
            test_split_ratio=test_split_ratio,
            prediction_horizon=prediction_horizon,
            use_differencing=use_differencing_from_ui
        )
        worker.signals.progress.connect(self._handle_training_progress)
        worker.signals.epoch_end.connect(self._handle_training_epoch)
        worker.signals.training_complete.connect(self._handle_training_complete)
        worker.signals.error.connect(self._handle_worker_error)
        worker.signals.finished.connect(lambda: self._update_ui_state(is_training=False))
        QThreadPool.globalInstance().start(worker)

    @Slot(pd.DataFrame, str)
    def _handle_data_ready(self, df, data_path):
        self._data_df = df
        self._data_path = data_path 
        filename = os.path.basename(data_path)
        status_msg = f"Данные загружены: {filename} ({len(df)} строк)"
        self._set_data_status(status_msg)
        self.status_update.emit(status_msg)
        
        if self.rb_csv.isChecked():
            extracted_ticker = self._extract_ticker_from_path(data_path)
            if extracted_ticker and (not self.ticker_input.text() or self.ticker_input.text() == config.DEFAULT_TICKER):
                self.ticker_input.setText(extracted_ticker)
                logger.debug(f"Тикер '{extracted_ticker}' извлечен из CSV и установлен в поле ввода.")
        self._update_ui_state()

    @Slot(int, int)
    def _handle_training_progress(self, epoch, total_epochs):
        self.progress_bar.setMaximum(total_epochs)
        self.progress_bar.setValue(epoch)
        status_msg = f"Обучение: Эпоха {epoch}/{total_epochs}"
        self.status_update.emit(status_msg)

    @Slot(int, dict)
    def _handle_training_epoch(self, epoch: int, logs: dict):
        log_items_for_ui = []
        for k, v in logs.items():
            if k.lower() == 'mape': 
                continue
            if k.lower() == 'val_mape': 
                continue
            if k.lower() == 'mape_last_step':
                continue
            log_items_for_ui.append(f"{k}={v:.4f}")
        
        if log_items_for_ui:
            log_entry = f"Эпоха {epoch}: " + ", ".join(log_items_for_ui)
            self.log_output.append(log_entry)
        else: 
            self.log_output.append(f"Эпоха {epoch}: (см. график потерь)")

        for key, value in logs.items():
            if key not in self._training_history:
                self._training_history[key] = []
            self._training_history[key].append(value)
        
        self.training_plot_canvas.plot_training_history(self._training_history)

    @Slot(str, object, object, object, object)
    def _handle_training_complete(self, model_name: str, model_dir: str | None,
                                  y_test_data: np.ndarray, y_pred_data: np.ndarray, 
                                  bilstm_metrics: dict): 
        
        logger.info(f"Обработка завершения обучения BiLSTM для '{model_name}'. Директория: {model_dir or 'НЕ СОХРАНЕНА'}")
        self.log_output.append(f"\n--- Результаты BiLSTM: {model_name} ---")

        if bilstm_metrics:
            mae_avg = bilstm_metrics.get('mae_avg', np.nan)
            rmse_avg = bilstm_metrics.get('rmse_avg', np.nan)
            mape_avg_custom = bilstm_metrics.get('mape_avg_custom', np.nan)
            mae_last = bilstm_metrics.get('mae_last', np.nan)
            rmse_last = bilstm_metrics.get('rmse_last', np.nan)
            mape_last_custom = bilstm_metrics.get('mape_last_custom', np.nan)
            
            horizon = 0
            if isinstance(y_test_data, np.ndarray) and y_test_data.ndim == 2:
                horizon = y_test_data.shape[1]
            
            if horizon > 0 :
                self.log_output.append(f"Метрики (усредненные по {horizon} шагам):")
                log_avg_list = []
                if not np.isnan(mae_avg): log_avg_list.append(f"MAE={mae_avg:.4f}")
                if not np.isnan(rmse_avg): log_avg_list.append(f"RMSE={rmse_avg:.4f}")
                if not np.isnan(mape_avg_custom): log_avg_list.append(f"MAPE_custom={mape_avg_custom:.2f}%")
                self.log_output.append("  " + (", ".join(log_avg_list) if log_avg_list else "Нет данных"))

                self.log_output.append(f"Метрики (для последнего шага - {horizon}):")
                log_last_list = []
                if not np.isnan(mae_last): log_last_list.append(f"MAE={mae_last:.4f}")
                if not np.isnan(rmse_last): log_last_list.append(f"RMSE={rmse_last:.4f}")
                if not np.isnan(mape_last_custom): log_last_list.append(f"MAPE_custom={mape_last_custom:.2f}%")
                self.log_output.append("  " + (", ".join(log_last_list) if log_last_list else "Нет данных"))
            else: 
                self.log_output.append("BiLSTM Оценка: Не удалось определить горизонт для детальных метрик.")
        else:
            self.log_output.append("BiLSTM Оценка: Результаты метрик не переданы.")

        y_test_for_plot = None
        y_pred_for_plot = None
        plot_title_suffix = ""
        horizon_for_plot_title = 0

        if isinstance(y_test_data, np.ndarray) and y_test_data.size > 0 and \
           isinstance(y_pred_data, np.ndarray) and y_pred_data.size > 0 and \
           y_test_data.shape == y_pred_data.shape:
            
            if y_test_data.ndim == 2 and y_test_data.shape[1] > 0:
                horizon_for_plot_title = y_test_data.shape[1]
                y_test_for_plot = y_test_data[:, -1] 
                y_pred_for_plot = y_pred_data[:, -1] 
                if horizon_for_plot_title > 1:
                    plot_title_suffix = f" (оценка шага {horizon_for_plot_title} из {horizon_for_plot_title})"
                else: 
                    plot_title_suffix = " (оценка шага 1 из 1)"
        
        if y_test_for_plot is not None and y_pred_for_plot is not None and \
           y_test_for_plot.size > 0: 
            eval_html_bilstm = plot_generator.create_evaluation_plot(
                y_test_main=y_test_for_plot,
                y_pred_main=y_pred_for_plot,
                model_name_main=f"BiLSTM: {model_name}{plot_title_suffix}"
            )
            if eval_html_bilstm:
                self.eval_plot_webview.setHtml(eval_html_bilstm)
                self.eval_plot_webview.setVisible(True)
                logger.debug(f"График оценки BiLSTM для '{model_name}{plot_title_suffix}' отображен.")
            else:
                log_msg_graph_err = f"Ошибка: Не удалось сгенерировать график оценки BiLSTM для '{model_name}{plot_title_suffix}'."
                self.log_output.append(log_msg_graph_err)
                logger.warning(log_msg_graph_err)
        else:
            logger.warning(f"Данные для графика оценки BiLSTM (y_test или y_pred) пусты или некорректны для '{model_name}{plot_title_suffix}'. График не будет показан.")
            self.eval_plot_webview.setHtml("<body style='background-color:#1E1E1E; color:#AAAAAA; font-family: sans-serif; text-align: center; padding-top: 50px;'><h2>BiLSTM: Нет данных для графика оценки.</h2><p>Тестовая выборка могла быть слишком мала.</p></body>")
            self.eval_plot_webview.setVisible(True)

        # --- Детализированный вывод результатов ARIMA ---
        arima_model_order = bilstm_metrics.get('arima_order', 'N/A')
        self.log_output.append(f"\n--- Результаты ARIMA (параметры {arima_model_order}) (сравнение): ---")
        
        if 'arima_error' in bilstm_metrics:
            self.log_output.append(f"  ARIMA Ошибка: {bilstm_metrics['arima_error']}")
        elif 'arima_mae_avg' in bilstm_metrics: # Проверяем наличие хотя бы одной метрики ARIMA
            arima_mae_avg = bilstm_metrics.get('arima_mae_avg', np.nan)
            arima_rmse_avg = bilstm_metrics.get('arima_rmse_avg', np.nan)
            arima_mape_avg = bilstm_metrics.get('arima_mape_avg_custom', np.nan)
            
            arima_mae_last = bilstm_metrics.get('arima_mae_last', np.nan)
            arima_rmse_last = bilstm_metrics.get('arima_rmse_last', np.nan)
            arima_mape_last = bilstm_metrics.get('arima_mape_last_custom', np.nan)
            
            # Используем 'horizon', который был определен для BiLSTM, т.к. он общий для прогноза
            if horizon > 0:
                self.log_output.append(f"Метрики (усредненные по {horizon} шагам):")
                log_arima_avg_list = []
                if not np.isnan(arima_mae_avg): log_arima_avg_list.append(f"MAE={arima_mae_avg:.4f}")
                if not np.isnan(arima_rmse_avg): log_arima_avg_list.append(f"RMSE={arima_rmse_avg:.4f}")
                if not np.isnan(arima_mape_avg): log_arima_avg_list.append(f"MAPE_custom={arima_mape_avg:.2f}%")
                self.log_output.append("  " + (", ".join(log_arima_avg_list) if log_arima_avg_list else "Нет данных"))

                self.log_output.append(f"Метрики (для последнего шага - {horizon}):")
                log_arima_last_list = []
                if not np.isnan(arima_mae_last): log_arima_last_list.append(f"MAE={arima_mae_last:.4f}")
                if not np.isnan(arima_rmse_last): log_arima_last_list.append(f"RMSE={arima_rmse_last:.4f}")
                if not np.isnan(arima_mape_last): log_arima_last_list.append(f"MAPE_custom={arima_mape_last:.2f}%")
                self.log_output.append("  " + (", ".join(log_arima_last_list) if log_arima_last_list else "Нет данных"))
            else:
                self.log_output.append("  ARIMA: Не удалось определить горизонт для детальных метрик (требуется информация от BiLSTM).")
        else:
            self.log_output.append("  ARIMA: Данные для сравнения отсутствуют или не были рассчитаны.")

        bilstm_save_status_msg = ""
        if model_dir is not None: 
            bilstm_save_status_msg = f"BiLSTM: Модель '{model_name}' обучена и сохранена."
            self.model_trained.emit(model_name, model_dir) 
            logger.info(f"BiLSTM: Модель '{model_name}' обучена и сохранена. Сигнал model_trained отправлен.")
        else:
            bilstm_save_status_msg = f"BiLSTM: Обучение для '{model_name}' завершено, но модель НЕ БЫЛА СОХРАНЕНА (см. логи)."
            logger.warning(f"BiLSTM: Модель '{model_name}' обучена, но НЕ сохранена.")
        
        self.log_output.append(f"\nСтатус: {bilstm_save_status_msg}")
        self._set_data_status(bilstm_save_status_msg) 
        self.status_update.emit(bilstm_save_status_msg) 
        self._update_ui_state(is_training=False)
        
    @Slot(str)
    def _handle_worker_error(self, error_message):
        logger.error(f"Получена ошибка от воркера: {error_message}")
        self._set_data_status(f"Ошибка: {error_message}", error=True)
        self.status_update.emit(f"Ошибка: {error_message}")
        
        if "данных" in error_message.lower() or "data" in error_message.lower():
            self._data_df = None
            self._data_path = None
        
        self._update_ui_state(is_loading=False, is_training=False)

    def _set_data_status(self, message, error=False):
        self.data_status_label.setText(message)
        if error:
            self.data_status_label.setObjectName("ErrorLabel")
            self.data_status_label.setStyleSheet("QLabel#ErrorLabel { color: #E74C3C; font-weight: bold; font-size: 9pt; }")
        else:
            self.data_status_label.setObjectName("StatusLabel")
            self.data_status_label.setStyleSheet("QLabel#StatusLabel { color: #00AEEF; font-size: 9pt; }")
        logger.debug(f"Статус данных обновлен: '{message}', ошибка={error}")

    def _extract_ticker_from_path(self, file_path: str | None) -> str | None:
        if not file_path:
            return None
        try:
            filename = os.path.basename(file_path)
            parts = filename.split('_')
            if len(parts) >= 1:
                ticker_candidate = parts[0]
                if ticker_candidate.isalpha() and ticker_candidate.isupper():
                    return ticker_candidate
        except Exception as e:
            logger.warning(f"Не удалось извлечь тикер из пути '{file_path}': {e}")
        return None