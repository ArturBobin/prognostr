from PySide6.QtCore import QObject, Signal, Slot, QRunnable, QThreadPool
import pandas as pd
import numpy as np 
import logging
import os
import json 
from datetime import datetime
import warnings # Для подавления предупреждений ARIMA

from app.utils import config
from app.core import data_loader, trainer, predictor, preprocessor
from app.core.trainer import restore_prices_from_differences
from app.plotting import plot_generator

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

class DataWorkerSignals(QObject):
    started = Signal()
    finished = Signal()
    error = Signal(str)
    data_ready = Signal(pd.DataFrame, str)

class DataWorker(QRunnable):
    def __init__(self, source_type: str, ticker: str = None, start_date: str = None, end_date: str = None, csv_path: str = None):
        super().__init__()
        self.signals = DataWorkerSignals()
        self.source_type = source_type
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.csv_path = csv_path
        self.setAutoDelete(True)

    @Slot()
    def run(self):
        logger.info(f"DataWorker запущен. Тип источника: '{self.source_type}', Тикер: '{self.ticker}', CSV: '{self.csv_path}'")
        self.signals.started.emit()
        df = None
        result_data_identifier = self.csv_path 
        error_msg = None
        try:
            if self.source_type == 'moex':
                if not all([self.ticker, self.start_date, self.end_date]):
                    error_msg = "Для загрузки с MOEX необходимо указать тикер, начальную и конечную даты."
                else:
                    df = data_loader.load_stock_data_moex(self.ticker, self.start_date, self.end_date)
                    if df is not None and not df.empty:
                        saved_path = data_loader.save_data_to_csv(df, self.ticker, self.start_date, self.end_date)
                        if saved_path:
                            result_data_identifier = saved_path
                        else:
                            error_msg = f"Данные для {self.ticker} загружены, но не удалось сохранить в CSV."
                            df = None 
                    elif df is not None and df.empty: 
                        error_msg = f"Данные для {self.ticker} с MOEX были загружены, но DataFrame пуст (возможно, нет торгов за период)."
                        df = None 
                    else: 
                        error_msg = f"Не удалось загрузить данные для {self.ticker} с MOEX."
            elif self.source_type == 'csv':
                if not self.csv_path:
                    error_msg = "Не указан путь к CSV файлу."
                else:
                    df = data_loader.load_data_from_csv(self.csv_path)
                    if df is None: 
                        error_msg = f"Не удалось загрузить данные из CSV файла: {self.csv_path}."
                    elif df.empty: 
                        error_msg = f"CSV файл {self.csv_path} загружен, но не содержит данных."
                        df = None 
            else:
                error_msg = f"Неизвестный тип источника данных: '{self.source_type}'."
        except Exception as e:
            error_msg = f"Непредвиденная ошибка в DataWorker ({self.source_type}): {e}"
            logger.error(error_msg, exc_info=config.DEBUG_MODE)
            df = None 

        if df is not None and not df.empty and result_data_identifier is not None:
            self.signals.data_ready.emit(df, result_data_identifier)
        else:
            final_error_msg = error_msg or "Неизвестная ошибка при загрузке/обработке данных."
            logger.error(f"DataWorker завершился с ошибкой: {final_error_msg}")
            self.signals.error.emit(final_error_msg)

        self.signals.finished.emit()
        logger.debug("DataWorker завершил выполнение.")


class TrainingWorkerSignals(QObject):
    started = Signal()
    finished = Signal()
    error = Signal(str)
    progress = Signal(int, int)
    epoch_end = Signal(int, dict)
    training_complete = Signal(str, object, object, object, object)


class TrainingWorker(QRunnable):
    def __init__(self, df_train: pd.DataFrame, epochs: int, batch_size: int, ticker: str,
                 test_split_ratio: float, prediction_horizon: int, use_differencing: bool):
        super().__init__()
        self.signals = TrainingWorkerSignals()
        self.df_train = df_train
        self.epochs = epochs
        self.batch_size = batch_size
        self.ticker = ticker
        self.test_split_ratio = test_split_ratio
        self.prediction_horizon = prediction_horizon
        self.use_differencing_flag = use_differencing
        self.setAutoDelete(True)
        
        
    def _emit_epoch_end_signal(self, epoch: int, logs: dict):
        self.signals.progress.emit(epoch, self.epochs)
        self.signals.epoch_end.emit(epoch, logs)

    @Slot()
    def run(self):
        logger.info(f"TrainingWorker запущен для тикера '{self.ticker}'. "
                    f"Эпох: {self.epochs}, Батч: {self.batch_size}, "
                    f"Доля теста: {self.test_split_ratio:.2f}, Горизонт прогноза: {self.prediction_horizon} дней."
                    f"Использовать дифференцирование: {self.use_differencing_flag}.")
        self.signals.started.emit()
        error_msg = None
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            diff_suffix = "_diff" if self.use_differencing_flag else ""
            model_name_candidate = f"{self.ticker}_h{self.prediction_horizon}{diff_suffix}_{timestamp}"
            model_save_dir = os.path.join(config.MODELS_DIR, model_name_candidate)

            train_result_bilstm = trainer.train_model(
                df_train_data=self.df_train.copy(),
                epochs=self.epochs,
                batch_size=self.batch_size,
                model_save_dir=model_save_dir,
                epoch_signal_emitter=self._emit_epoch_end_signal,
                test_split_ratio=self.test_split_ratio,
                prediction_horizon=self.prediction_horizon,
                use_differencing_override = self.use_differencing_flag
            )

            if not train_result_bilstm:
                error_msg = f"Процесс обучения BiLSTM для '{self.ticker}' (горизонт {self.prediction_horizon}) не вернул результат."
                logger.error(error_msg)
            else:
                actual_model_name_from_trainer, actual_model_dir, history_logs, \
                y_test_absolute_bilstm, y_pred_absolute_bilstm, bilstm_metrics_dict, \
                df_processed_from_trainer = train_result_bilstm
                
                # --- Начало блока ARIMA Walk-Forward Validation ---
                try:
                    logger.info(f"Запуск ARIMA Walk-Forward Validation (тикер: {self.ticker}, горизонт: {self.prediction_horizon}).")
                    
                    if df_processed_from_trainer is None or df_processed_from_trainer.empty:
                        raise ValueError("DataFrame (df_processed_from_trainer) для ARIMA пуст.")
                    if config.TARGET_FEATURE_ORIGINAL not in df_processed_from_trainer.columns:
                         raise KeyError(f"Колонка '{config.TARGET_FEATURE_ORIGINAL}' отсутствует в df_processed_from_trainer.")

                    # Используем весь обработанный временной ряд TARGET_FEATURE_ORIGINAL
                    # df_processed_from_trainer уже прошел dropna после расчета индикаторов/разностей
                    # и его длина соответствует данным, на которых BiLSTM обучалась и тестировалась
                    full_target_series_arima = df_processed_from_trainer[config.TARGET_FEATURE_ORIGINAL].copy()
                    
                    if len(full_target_series_arima) < 20: # Условный минимум для всего ряда
                        raise ValueError(f"Недостаточно данных ({len(full_target_series_arima)}) в full_target_series_arima для ARIMA.")

                    # Определяем общий тестовый период для ARIMA, он должен соответствовать тому,
                    # на котором оценивалась BiLSTM.
                    # Длина тестовой выборки BiLSTM (в количестве точек, а не последовательностей):
                    # len(y_test_absolute_bilstm) - это количество тестовых *последовательностей*
                    # Если y_test_absolute_bilstm не пуст, то y_test_absolute_bilstm.shape[0] - это кол-во окон
                    # А y_test_absolute_bilstm.shape[0] + self.prediction_horizon - 1 - это примерное кол-во точек в тесте
                    
                    
                    # Общий тренировочно-тестовый сплит для всего ряда full_target_series_arima
                    # Этот сплит определяет, где заканчивается "история" и начинается "тест" для walk-forward
                    n_total_points = len(full_target_series_arima)
                    # Индекс, с которого начинается общий тестовый период
                    test_period_start_index = int(n_total_points * (1 - self.test_split_ratio)) 
                    
                    # Данные до начала общего тестового периода
                    history_for_arima_initial_train = full_target_series_arima.iloc[:test_period_start_index]
                    # Данные, которые будут использоваться как "будущие" для walk-forward
                    test_data_for_arima_walk_forward = full_target_series_arima.iloc[test_period_start_index:]

                    if len(history_for_arima_initial_train) < config.ARIMA_MIN_TRAIN_SIZE:
                         raise ValueError(f"Начальный набор для обучения ARIMA слишком мал: {len(history_for_arima_initial_train)} "
                                          f"(требуется {config.ARIMA_MIN_TRAIN_SIZE}).")
                    if len(test_data_for_arima_walk_forward) < self.prediction_horizon:
                        raise ValueError(f"Общий тестовый набор для ARIMA Walk-Forward слишком мал: {len(test_data_for_arima_walk_forward)} "
                                         f"(требуется как минимум {self.prediction_horizon}).")

                    all_arima_predictions = []
                    all_arima_actuals = []
                    
                    current_train_window_arima = history_for_arima_initial_train.copy()
                    
                    # Шаг сдвига окна в Walk-Forward равен горизонту прогноза
                    walk_forward_step = self.prediction_horizon 

                    arima_order = config.ARIMA_DEFAULT_ORDER
                    
                    logger.info(f"ARIMA Walk-Forward: Начальный размер обучающего окна: {len(current_train_window_arima)}.")
                    logger.info(f"ARIMA Walk-Forward: Размер общего тестового периода: {len(test_data_for_arima_walk_forward)}.")
                    logger.info(f"ARIMA Walk-Forward: Шаг сдвига окна: {walk_forward_step}, Горизонт: {self.prediction_horizon}.")

                    warnings.simplefilter('ignore', ConvergenceWarning)
                    warnings.simplefilter('ignore', UserWarning)

                    for i in range(0, len(test_data_for_arima_walk_forward) - self.prediction_horizon + 1, walk_forward_step):
                        if len(current_train_window_arima) < config.ARIMA_MIN_TRAIN_SIZE_ITER: # Минимальный размер для итерации
                            logger.warning(f"ARIMA Walk-Forward: Пропуск итерации, размер текущего обучающего окна {len(current_train_window_arima)} "
                                           f"меньше {config.ARIMA_MIN_TRAIN_SIZE_ITER}.")
                            break 
                        
                        try:
                            arima_model = ARIMA(current_train_window_arima, order=arima_order, 
                                                enforce_stationarity=False, enforce_invertibility=False)
                            arima_model_fit = arima_model.fit()
                            
                            # Прогноз на self.prediction_horizon шагов
                            forecast_steps = arima_model_fit.forecast(steps=self.prediction_horizon)
                            
                            # Соответствующие фактические значения
                            actual_steps = test_data_for_arima_walk_forward.iloc[i : i + self.prediction_horizon]
                            
                            if len(forecast_steps) == self.prediction_horizon and len(actual_steps) == self.prediction_horizon:
                                all_arima_predictions.append(forecast_steps.values) # Берем .values для numpy array
                                all_arima_actuals.append(actual_steps.values)
                            else:
                                logger.warning(f"ARIMA Walk-Forward: Пропуск прогноза на итерации с i={i}. "
                                               f"Длина прогноза: {len(forecast_steps)}, длина факта: {len(actual_steps)}. "
                                               f"Ожидалось: {self.prediction_horizon}.")
                                # Если прогноз короче, возможно, достигли конца ряда test_data_for_arima_walk_forward
                                # или была ошибка в .forecast()
                                if len(forecast_steps) < self.prediction_horizon and len(actual_steps) < self.prediction_horizon :
                                    logger.info("Похоже, достигнут конец данных для walk-forward.")
                                    break


                        except Exception as wf_iter_exc:
                            logger.error(f"ARIMA Walk-Forward: Ошибка на итерации (i={i}, train_len={len(current_train_window_arima)}): {wf_iter_exc}", exc_info=False)

                        # Расширяем/сдвигаем окно обучения
                        # Тип окна: 'expanding'
                        # Добавляем 'walk_forward_step' новых точек из test_data_for_arima_walk_forward
                        next_chunk_to_add_idx_end = i + walk_forward_step
                        if next_chunk_to_add_idx_end <= len(test_data_for_arima_walk_forward):
                            chunk_to_add = test_data_for_arima_walk_forward.iloc[i : next_chunk_to_add_idx_end]
                            current_train_window_arima = pd.concat([current_train_window_arima, chunk_to_add])
                        else:
                            logger.info("ARIMA Walk-Forward: Достигнут конец тестовых данных для расширения окна.")
                            break
                    
                    warnings.simplefilter('default', ConvergenceWarning)
                    warnings.simplefilter('default', UserWarning)

                    if not all_arima_predictions or not all_arima_actuals:
                        raise ValueError("ARIMA Walk-Forward не сгенерировал ни одного прогноза.")

                    # Преобразуем списки массивов в единые 2D массивы
                    y_pred_arima_walk_forward = np.array(all_arima_predictions) # (num_forecast_windows, horizon)
                    y_true_arima_walk_forward = np.array(all_arima_actuals)   # (num_forecast_windows, horizon)

                    if y_pred_arima_walk_forward.shape != y_true_arima_walk_forward.shape or y_pred_arima_walk_forward.size == 0:
                         raise ValueError(f"Несоответствие форм или пустые массивы после ARIMA Walk-Forward. "
                                          f"Pred: {y_pred_arima_walk_forward.shape}, True: {y_true_arima_walk_forward.shape}")

                    # Расчет метрик на агрегированных результатах
                    def calculate_mape_custom_np(y_true_np, y_pred_np):
                        if y_true_np.size == 0: return np.nan
                        abs_y_true = np.abs(y_true_np)
                        safe_denominator = np.where(abs_y_true < np.finfo(float).eps, np.finfo(float).eps, abs_y_true)
                        return np.mean(np.abs((y_true_np - y_pred_np) / safe_denominator)) * 100

                    # Усредненные метрики по всем шагам и всем окнам
                    arima_mae_avg = mean_absolute_error(y_true_arima_walk_forward.flatten(), y_pred_arima_walk_forward.flatten())
                    arima_rmse_avg = np.sqrt(mean_squared_error(y_true_arima_walk_forward.flatten(), y_pred_arima_walk_forward.flatten()))
                    arima_mape_avg_custom = calculate_mape_custom_np(y_true_arima_walk_forward.flatten(), y_pred_arima_walk_forward.flatten())
                    
                    bilstm_metrics_dict['arima_mae_avg'] = arima_mae_avg
                    bilstm_metrics_dict['arima_rmse_avg'] = arima_rmse_avg
                    bilstm_metrics_dict['arima_mape_avg_custom'] = arima_mape_avg_custom
                    logger.info(f"ARIMA {arima_order} (Walk-Forward) метрики (усредненные по {self.prediction_horizon} шагам): "
                                f"MAE_avg={arima_mae_avg:.4f}, RMSE_avg={arima_rmse_avg:.4f}, MAPE_avg_custom={arima_mape_avg_custom:.2f}%")

                    # Метрики для последнего шага прогноза (усредненные по всем окнам)
                    if self.prediction_horizon > 0 and y_true_arima_walk_forward.shape[1] == self.prediction_horizon:
                        y_true_last_step_arima = y_true_arima_walk_forward[:, -1]
                        y_pred_last_step_arima = y_pred_arima_walk_forward[:, -1]
                        
                        arima_mae_last = mean_absolute_error(y_true_last_step_arima, y_pred_last_step_arima)
                        arima_rmse_last = np.sqrt(mean_squared_error(y_true_last_step_arima, y_pred_last_step_arima))
                        arima_mape_last_custom = calculate_mape_custom_np(y_true_last_step_arima, y_pred_last_step_arima)

                        bilstm_metrics_dict['arima_mae_last'] = arima_mae_last
                        bilstm_metrics_dict['arima_rmse_last'] = arima_rmse_last
                        bilstm_metrics_dict['arima_mape_last_custom'] = arima_mape_last_custom
                        logger.info(f"ARIMA {arima_order} (Walk-Forward) метрики (для последнего шага - {self.prediction_horizon}): "
                                    f"MAE_last={arima_mae_last:.4f}, RMSE_last={arima_rmse_last:.4f}, MAPE_last_custom={arima_mape_last_custom:.2f}%")
                    
                    bilstm_metrics_dict['arima_order'] = str(arima_order)
                    bilstm_metrics_dict['arima_walk_forward_windows'] = y_pred_arima_walk_forward.shape[0]


                except Exception as arima_exc:
                    logger.error(f"Ошибка при выполнении ARIMA Walk-Forward Validation: {arima_exc}", exc_info=config.DEBUG_MODE)
                    bilstm_metrics_dict['arima_error'] = f"Walk-Forward Error: {str(arima_exc)}"
                # --- Конец блока ARIMA ---

                if actual_model_dir is not None:
                    logger.info(f"Обучение BiLSTM (горизонт {self.prediction_horizon}) успешно завершено. "
                                f"Модель '{actual_model_name_from_trainer}' сохранена в '{actual_model_dir}'.")
                    self.signals.training_complete.emit(actual_model_name_from_trainer, actual_model_dir,
                                                        y_test_absolute_bilstm, y_pred_absolute_bilstm, bilstm_metrics_dict)
                else:
                    logger.warning(f"Обучение BiLSTM для '{actual_model_name_from_trainer}' (горизонт {self.prediction_horizon}) завершено, "
                                   f"НО модель НЕ БЫЛА СОХРАНЕНА.")
                    self.signals.training_complete.emit(actual_model_name_from_trainer, None,
                                                        y_test_absolute_bilstm, y_pred_absolute_bilstm, bilstm_metrics_dict)
        except Exception as e:
            error_msg = f"Непредвиденная ошибка в TrainingWorker ({self.ticker}, горизонт {self.prediction_horizon}): {e}"
            logger.error(error_msg, exc_info=True)

        if error_msg:
            self.signals.error.emit(error_msg)
        self.signals.finished.emit()
        logger.debug(f"TrainingWorker для '{self.ticker}' (горизонт {self.prediction_horizon}) завершил выполнение.")


class PredictionWorkerSignals(QObject):
    started = Signal()
    finished = Signal()
    error = Signal(str)
    prediction_ready = Signal(list, str, float, pd.DataFrame, str)


class PredictionWorker(QRunnable):
    def __init__(self, ticker: str, model_dir: str, indicators_state: dict):
        super().__init__()
        self.signals = PredictionWorkerSignals()
        self.ticker = ticker
        self.model_dir = model_dir
        self.indicators_state = indicators_state
        self.setAutoDelete(True)

    @Slot()
    def run(self):
        model_base_name = os.path.basename(self.model_dir)
        logger.info(f"PredictionWorker запущен для тикера '{self.ticker}', модель из директории: '{model_base_name}'")
        self.signals.started.emit()
        
        error_msg = None
        predicted_values_from_model = None 
        final_absolute_predictions_list = None
        
        plot_html_output = ""
        last_actual_price_original_output = 0.0
        df_for_plot_output = pd.DataFrame() 
        
        model_prediction_horizon = None
        model_used_differencing = False
        model_target_feature_original_name = config.TARGET_FEATURE_ORIGINAL

        try:
            metadata_path = os.path.join(self.model_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    model_prediction_horizon = metadata.get('prediction_horizon')
                    model_used_differencing = metadata.get('used_differencing', False)
                    model_target_feature_original_name = metadata.get('target_feature_original', config.TARGET_FEATURE_ORIGINAL)

                    if model_prediction_horizon:
                         logger.info(f"Загружены метаданные для модели '{model_base_name}'. "
                                     f"Горизонт прогноза модели: {model_prediction_horizon} дней. "
                                     f"Использовалось дифференцирование: {model_used_differencing}. "
                                     f"Исходная цель: '{model_target_feature_original_name}'.")
                    else:
                        error_msg = f"Ключ 'prediction_horizon' отсутствует или некорректен в {metadata_path}."
                except Exception as meta_err:
                    error_msg = f"Не удалось загрузить или прочитать metadata.json для модели '{model_base_name}': {meta_err}"
            else:
                error_msg = f"Файл metadata.json не найден для модели '{model_base_name}'. Невозможно определить параметры модели."

            if error_msg:
                logger.error(error_msg)
                self.signals.error.emit(error_msg)
                self.signals.finished.emit()
                return

            end_date = datetime.now().date()
            end_date_str = end_date.strftime('%Y-%m-%d')
            days_needed_for_model_input = config.MIN_DAYS_FOR_PREDICTION 
            non_trading_day_factor = 1.8 
            extra_buffer_days = 20      
            days_to_load_calendar = int(days_needed_for_model_input * non_trading_day_factor) + extra_buffer_days
            start_date = end_date - pd.Timedelta(days=days_to_load_calendar)
            start_date_str = start_date.strftime('%Y-%m-%d')

            logger.info(f"Загрузка свежих данных для {self.ticker} с {start_date_str} по {end_date_str} "
                        f"(требуется ~{days_needed_for_model_input} торг. дней для входа модели).")
            df_latest_raw = data_loader.load_stock_data_moex(self.ticker, start_date_str, end_date_str)

            if df_latest_raw is None or df_latest_raw.empty:
                error_msg = f"Не удалось загрузить свежие данные для {self.ticker} (период: {start_date_str} - {end_date_str})."
            elif len(df_latest_raw) < days_needed_for_model_input:
                error_msg = (f"Загружено недостаточно торговых дней ({len(df_latest_raw)}) для {self.ticker}. "
                             f"Требуется как минимум {days_needed_for_model_input} дней для входа модели (с учетом индикаторов и TIME_STEPS).")
            else:
                preprocess_result = preprocessor.preprocess_for_prediction(df_latest_raw.copy(), self.model_dir)
                
                if preprocess_result is None:
                    error_msg = "Ошибка предобработки данных для предсказания."
                else:
                    (input_sequence, 
                     last_actual_price_original_output,
                     df_for_plot_output,
                     last_known_original_for_restore
                    ) = preprocess_result
                    
                    if input_sequence is None:
                        error_msg = "Ошибка: входная последовательность None после предобработки."
                    else:
                        predicted_values_from_model = predictor.make_prediction(input_sequence, self.model_dir)

                        if predicted_values_from_model is None or not isinstance(predicted_values_from_model, list):
                            error_msg = f"Ошибка при выполнении предсказания моделью для {self.ticker} (не получен список)."
                        elif len(predicted_values_from_model) != model_prediction_horizon:
                             error_msg = (f"Модель вернула {len(predicted_values_from_model)} предсказаний, "
                                          f"ожидалось {model_prediction_horizon} на основе метаданных.")
                             predicted_values_from_model = None 
                        else:
                            logger.info(f"Получен прогноз (до возможного восстановления) на {model_prediction_horizon} дней для {self.ticker}: {predicted_values_from_model}")
                            
                            if model_used_differencing:
                                if last_known_original_for_restore is None:
                                    error_msg = "Ошибка: модель использовала дифференцирование, но не получено значение для восстановления абсолютных цен."
                                    final_absolute_predictions_list = None
                                else:
                                    logger.info(f"Модель использовала дифференцирование. Восстановление абсолютных цен из предсказанных разностей, начиная с {last_known_original_for_restore:.4f}...")
                                    restored_array = restore_prices_from_differences(
                                        np.array(predicted_values_from_model).reshape(1, -1),
                                        np.array([last_known_original_for_restore]),
                                        model_prediction_horizon
                                    )
                                    final_absolute_predictions_list = restored_array.flatten().tolist()
                                    logger.info(f"Восстановленные абсолютные цены: {final_absolute_predictions_list}")
                            else: 
                                final_absolute_predictions_list = predicted_values_from_model
                                logger.info(f"Модель предсказывала абсолютные цены. Восстановление не требуется.")
                            
                            if final_absolute_predictions_list:
                                plot_html_output = plot_generator.create_prediction_plot(
                                    df_history=df_for_plot_output, 
                                    list_of_predicted_values=final_absolute_predictions_list,
                                    ticker=self.ticker,
                                    target_column_name_for_plot=model_target_feature_original_name,
                                    **self.indicators_state
                                )
                                if plot_html_output is None:
                                    logger.warning(f"Прогноз для {self.ticker} получен, но график не сгенерирован.")
        
        except Exception as e:
            error_msg = f"Критическая ошибка в PredictionWorker ({self.ticker}): {e}"
            logger.error(error_msg, exc_info=config.DEBUG_MODE)
            final_absolute_predictions_list = None
            plot_html_output = ""
            df_for_plot_output = pd.DataFrame()

        if final_absolute_predictions_list is not None and not df_for_plot_output.empty:
            self.signals.prediction_ready.emit(
                final_absolute_predictions_list,
                plot_html_output,
                last_actual_price_original_output,
                df_for_plot_output,
                self.ticker
            )
        else:
            final_error_msg = error_msg or f"Неизвестная ошибка предсказания для {self.ticker}."
            logger.error(f"PredictionWorker для {self.ticker} с ошибкой: {final_error_msg}")
            self.signals.error.emit(final_error_msg)

        self.signals.finished.emit()
        logger.debug(f"PredictionWorker для '{self.ticker}' завершил выполнение.")