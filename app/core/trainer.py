import tensorflow as tf
import numpy as np
import pandas as pd
import os
import json
import logging
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.metrics import mean_squared_error, mean_absolute_error
from app.core import model_builder, preprocessor
from app.utils import config

logger = logging.getLogger(__name__)

class EpochEndSignalCallback(Callback):
    def __init__(self, signal_emitter_func):
        super().__init__()
        self.signal_emitter_func = signal_emitter_func

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        serializable_logs = {k: float(v) if isinstance(v, (np.float32, np.float64, np.ndarray)) else v for k, v in logs.items()}
        self.signal_emitter_func(epoch + 1, serializable_logs)
        logger.debug(f"Epoch {epoch + 1} ended. Logs: {serializable_logs}")

class DebugMapeCallback(Callback):
    def __init__(self, validation_data, target_feature_name, log_freq=5):
        super().__init__()
        self.y_val_scaled = validation_data[1] if validation_data and len(validation_data) > 1 else None
        self.target_feature_name = target_feature_name
        self.log_freq = log_freq
        if self.y_val_scaled is not None:
            y_val_for_stats = self.y_val_scaled[:, 0] if self.y_val_scaled.ndim == 2 else self.y_val_scaled
            
            logger.debug(f"DebugMapeCallback: y_val_scaled (первые 5, шаг 1, цель: {self.target_feature_name}): {y_val_for_stats[:5].flatten()}")
            min_val, max_val = np.min(y_val_for_stats), np.max(y_val_for_stats)
            zeros_count = np.sum(np.isclose(y_val_for_stats, 0.0))
            near_zeros_count = np.sum(np.isclose(y_val_for_stats, 0.0, atol=1e-5)) # atol для сравнения с нулем
            logger.info(f"DebugMapeCallback: Статистика по y_val_scaled (шаг 1, цель: {self.target_feature_name}): "
                        f"Min={min_val:.6f}, Max={max_val:.6f}, Нулей: {zeros_count}, Близких к нулю: {near_zeros_count}/{len(y_val_for_stats)}.")
        else:
            logger.warning(f"DebugMapeCallback: Валидационные данные (y_val_scaled) не предоставлены (цель: {self.target_feature_name}).")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if epoch == 0 or (epoch + 1) % self.log_freq == 0:
            log_parts = []
            for m in ['val_mape', 'mape', 'val_mae_last_step', 'mae_last_step', 'val_mape_last_step', 'mape_last_step', 'val_loss', 'loss']:
                if m in logs: log_parts.append(f"{m}={logs[m]:.4f}")
            
            if self.y_val_scaled is not None:
                y_val_for_stats_log = self.y_val_scaled[:, 0] if self.y_val_scaled.ndim == 2 else self.y_val_scaled
                log_parts.append(f"min(y_val_s_step1({self.target_feature_name}))={np.min(y_val_for_stats_log):.6f}")
            
            if log_parts: logger.debug(f"DebugMapeCallback (Epoch {epoch+1}, Цель: {self.target_feature_name}): " + ", ".join(log_parts))


def restore_prices_from_differences(predicted_diffs: np.ndarray, 
                                    last_known_actual_prices_before_each_sequence: np.ndarray,
                                    prediction_horizon: int) -> np.ndarray:
    """
    Восстанавливает абсолютные цены из предсказанных разностей.
    predicted_diffs: (num_samples, prediction_horizon) - предсказанные (немасштабированные) разности.
    last_known_actual_prices_before_each_sequence: (num_samples,) - массив последних известных
                                                     абсолютных цен ПЕРЕД началом каждой из num_samples
                                                     прогнозных последовательностей.
    prediction_horizon: int - количество шагов в прогнозе.
    Возвращает: (num_samples, prediction_horizon) - восстановленные абсолютные цены.
    """
    num_samples = predicted_diffs.shape[0]
    restored_prices = np.zeros_like(predicted_diffs)

    if predicted_diffs.ndim == 1: # Если только одна предсказанная последовательность
        predicted_diffs = predicted_diffs.reshape(1, -1)
        if last_known_actual_prices_before_each_sequence.ndim == 0:
             last_known_actual_prices_before_each_sequence = np.array([last_known_actual_prices_before_each_sequence])


    if num_samples != len(last_known_actual_prices_before_each_sequence):
        logger.error(f"Ошибка восстановления цен: количество выборок в predicted_diffs ({num_samples}) "
                     f"не совпадает с количеством начальных цен ({len(last_known_actual_prices_before_each_sequence)}).")
        # Возвращаем нули
        return restored_prices 

    for i in range(num_samples):
        current_base_price = last_known_actual_prices_before_each_sequence[i]
        for h in range(prediction_horizon):
            current_base_price += predicted_diffs[i, h]
            restored_prices[i, h] = current_base_price
            
    logger.debug(f"Восстановлены абсолютные цены из разностей. "
                 f"Форма predicted_diffs: {predicted_diffs.shape}, "
                 f"Форма restored_prices: {restored_prices.shape}")
    return restored_prices


def train_model(df_train_data: pd.DataFrame, epochs: int, batch_size: int,
                model_save_dir: str, epoch_signal_emitter,
                test_split_ratio: float, prediction_horizon: int,
                use_differencing_override: bool
               ) -> tuple | None:

    
    model_name_for_log = os.path.basename(model_save_dir)
    try:
        logger.info(f"Начало процесса обучения для '{model_name_for_log}'. "
                    f"Эпох: {epochs}, Батч: {batch_size}, Доля теста: {test_split_ratio:.2f}, Горизонт: {prediction_horizon}")
        logger.info(f"Фактическое использование дифференцирования для этого обучения: {use_differencing_override}.")
        
        # Определяем TARGET_FEATURE на основе use_differencing_override для этого сеанса
        current_target_feature = config.TARGET_FEATURE_ORIGINAL
        if use_differencing_override:
            current_target_feature = f'{config.DIFFERENCING_COLUMN}_diff{config.DIFFERENCING_ORDER}'
        logger.info(f"Целевой признак для этого обучения: '{current_target_feature}'.")


        preprocess_result = preprocessor.preprocess_for_training(
            df_train_data, model_save_dir, test_split_ratio, prediction_horizon,
            use_differencing_runtime=use_differencing_override,
            current_target_feature_runtime=current_target_feature
        )

        if preprocess_result is None:
            logger.error(f"Предобработка данных для обучения '{model_name_for_log}' не удалась.")
            return None 

        # Распаковываем результат от preprocessor
        (X_train, y_train, X_test, y_test, 
         feature_scaler, target_scaler, 
         last_original_target_value_in_train, # Последнее абс. значение в конце трейн выборки
         y_test_original_absolute_sequences,  # (num_test_samples, horizon) - абс. значения для каждого y_test
         df_processed
        ) = preprocess_result
        
        logger.info(f"Данные для '{model_name_for_log}' успешно предобработаны.")
        # Логируем с current_target_feature
        logger.debug(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape} (цель: {current_target_feature})")
        logger.debug(f"  X_test shape: {X_test.shape}, y_test shape: {y_test.shape} (цель: {current_target_feature})")
        if use_differencing_override:
            logger.debug(f"  last_original_target_value_in_train: {last_original_target_value_in_train}")
            logger.debug(f"  y_test_original_absolute_sequences shape: {y_test_original_absolute_sequences.shape}")


        if X_train.size == 0 or y_train.size == 0:
            logger.error(f"Обучающая выборка пуста для '{model_name_for_log}'.")
            return None
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = model_builder.build_bilstm_model(input_shape, prediction_horizon)
        
        early_stopping_monitor = config.EARLY_STOPPING_MONITOR
        model_metric_names = [m.name if hasattr(m, 'name') else m for m in model.metrics]
        logger.debug(f"Метрики модели для EarlyStopping: {model_metric_names}")
        
        potential_monitors_val = ['val_mape_last_step', 'val_mae_last_step', 'val_loss']
        potential_monitors_train = ['mape_last_step', 'mae_last_step', 'loss']

        if X_test.size > 0 and y_test.size > 0:
            for monitor_candidate in potential_monitors_val:
                if monitor_candidate in model_metric_names or monitor_candidate == 'val_loss':
                    early_stopping_monitor = monitor_candidate
                    logger.info(f"EarlyStopping будет мониторить: '{early_stopping_monitor}' (валидационная).")
                    break
        else:
            logger.warning(f"Тестовая выборка для '{model_name_for_log}' пуста. EarlyStopping будет мониторить метрики на тренировочных данных.")
            for monitor_candidate in potential_monitors_train:
                 if monitor_candidate in model_metric_names or monitor_candidate == 'loss':
                    early_stopping_monitor = monitor_candidate
                    logger.info(f"EarlyStopping будет мониторить: '{early_stopping_monitor}' (тренировочная).")
                    break
        
        early_stopping = EarlyStopping(
            monitor=early_stopping_monitor,
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        )
        epoch_signal_callback = EpochEndSignalCallback(signal_emitter_func=epoch_signal_emitter)
        
        validation_data_for_fit = None
        if X_test.size > 0 and y_test.size > 0:
            validation_data_for_fit = (X_test, y_test)
        
        debug_mape_callback = DebugMapeCallback(validation_data=validation_data_for_fit, target_feature_name=current_target_feature)
        callbacks_list = [early_stopping, epoch_signal_callback, debug_mape_callback]

        logger.info(f"Начало обучения модели '{model_name_for_log}' (model.fit) на цели '{current_target_feature}'...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data_for_fit,
            callbacks=callbacks_list,
            verbose=0 
        )
        logger.info(f"Обучение модели '{model_name_for_log}' завершено.")
        
        # Инициализация переменных для результатов
        y_pred_for_ui_plot = np.array([]) 
        y_test_for_ui_plot = np.array([])
        bilstm_metrics = {}

        if X_test.size > 0 and y_test.size > 0:
            logger.info(f"Оценка модели '{model_name_for_log}' на тестовых данных (цель: '{current_target_feature}')...")
            
            # 1. Получаем предсказания модели (масштабированные значения TARGET_FEATURE)
            y_pred_scaled = model.predict(X_test, verbose=0) # shape: (num_samples, horizon)
            
            # 2. Инвертируем масштабирование для предсказанных и истинных значений TARGET_FEATURE
            # y_pred_unscaled_target будет содержать немасштабированные разности или цены
            y_pred_unscaled_target = target_scaler.inverse_transform(y_pred_scaled)
            # y_test_unscaled_target будет содержать немасштабированные истинные разности или цены
            y_test_unscaled_target = target_scaler.inverse_transform(y_test) # y_test был (num_samples, horizon)

            if use_differencing_override:
                logger.info(f"Восстановление абсолютных цен из разностей для оценки (цель: '{config.TARGET_FEATURE_ORIGINAL}')...")
                
                if y_test_original_absolute_sequences.size > 0 and y_test_unscaled_target.size > 0:
                    # base_prices_for_pred_restore это цены ПЕРЕД первой разностью для каждого окна
                    # y_test_original_absolute_sequences[i, 0] = base_price[i] + y_test_unscaled_target[i, 0]
                    # => base_price[i] = y_test_original_absolute_sequences[i, 0] - y_test_unscaled_target[i, 0]
                    base_prices_for_pred_restore = y_test_original_absolute_sequences[:, 0] - y_test_unscaled_target[:, 0]
                    
                    y_pred_for_ui_plot = restore_prices_from_differences(
                        y_pred_unscaled_target, # предсказанные разности
                        base_prices_for_pred_restore, # база для начала восстановления каждой последовательности
                        prediction_horizon
                    )
                    y_test_for_ui_plot = y_test_original_absolute_sequences # Истинные абсолютные значения
                else:
                    logger.warning("Недостаточно данных для восстановления абсолютных цен на тесте (y_test_original_absolute_sequences или y_test_unscaled_target пусты).")
                    y_pred_for_ui_plot = np.array([])
                    y_test_for_ui_plot = np.array([])

            else:
                y_pred_for_ui_plot = y_pred_unscaled_target
                y_test_for_ui_plot = y_test_unscaled_target

            # Расчет метрик на y_pred_for_ui_plot и y_test_for_ui_plot (абсолютные цены)
            if y_pred_for_ui_plot.size > 0 and y_test_for_ui_plot.size > 0 and y_pred_for_ui_plot.shape == y_test_for_ui_plot.shape:
                mae_avg = mean_absolute_error(y_test_for_ui_plot, y_pred_for_ui_plot)
                rmse_avg = np.sqrt(mean_squared_error(y_test_for_ui_plot, y_pred_for_ui_plot))
                
                mape_steps_avg = []
                for step_idx in range(prediction_horizon):
                    y_true_s = y_test_for_ui_plot[:, step_idx]
                    y_pred_s = y_pred_for_ui_plot[:, step_idx]
                    abs_y_true_s = np.abs(y_true_s)
                    safe_denom_s = np.where(abs_y_true_s < 1e-8, 1e-8, abs_y_true_s)
                    mape_s_val = np.mean(np.abs((y_true_s - y_pred_s) / safe_denom_s)) * 100
                    mape_steps_avg.append(mape_s_val)
                mape_avg_custom = np.mean(mape_steps_avg) if mape_steps_avg else np.nan
                
                logger.info(f"  Метрики на АБСОЛЮТНЫХ ценах (усредненные по {prediction_horizon} шагам): "
                            f"MAE_avg={mae_avg:.4f}, RMSE_avg={rmse_avg:.4f}, MAPE_avg_custom={mape_avg_custom:.2f}%")
                bilstm_metrics.update({'mae_avg': mae_avg, 'rmse_avg': rmse_avg, 'mape_avg_custom': mape_avg_custom})

                y_test_last_abs = y_test_for_ui_plot[:, -1]
                y_pred_last_abs = y_pred_for_ui_plot[:, -1]
                mae_last = mean_absolute_error(y_test_last_abs, y_pred_last_abs)
                rmse_last = np.sqrt(mean_squared_error(y_test_last_abs, y_pred_last_abs))
                abs_y_test_last = np.abs(y_test_last_abs)
                safe_denom_last = np.where(abs_y_test_last < 1e-8, 1e-8, abs_y_test_last)
                mape_last_custom = np.mean(np.abs((y_test_last_abs - y_pred_last_abs) / safe_denom_last)) * 100
                
                logger.info(f"  Метрики на АБСОЛЮТНЫХ ценах (для последнего шага {prediction_horizon}): "
                            f"MAE_last={mae_last:.4f}, RMSE_last={rmse_last:.4f}, MAPE_last_custom={mape_last_custom:.2f}%")
                bilstm_metrics.update({'mae_last': mae_last, 'rmse_last': rmse_last, 'mape_last_custom': mape_last_custom})

                # Добавим метрики из history Keras
                if validation_data_for_fit: # Если была валидация
                    bilstm_metrics['keras_val_loss_target'] = history.history.get('val_loss', [np.nan])[-1]
                    bilstm_metrics['keras_val_mae_target'] = history.history.get('val_mae', [np.nan])[-1] # Усредненная MAE по шагам для TARGET_FEATURE
                    bilstm_metrics['keras_val_mape_target'] = history.history.get('val_mape', [np.nan])[-1] # Усредненная MAPE
                    bilstm_metrics['keras_val_mae_last_step_target'] = history.history.get('val_mae_last_step', [np.nan])[-1]
                    bilstm_metrics['keras_val_mape_last_step_target'] = history.history.get('val_mape_last_step', [np.nan])[-1]
            else:
                logger.warning(f"Не удалось рассчитать финальные метрики на абсолютных ценах из-за несоответствия форм или пустых массивов.")
        else: 
            logger.warning(f"Тестовая выборка была пуста для '{model_name_for_log}'. Оценка не проводилась.")
            nan_metrics_keys = ['mae_avg', 'rmse_avg', 'mape_avg_custom', 'mae_last', 'rmse_last', 'mape_last_custom',
                                'keras_val_loss_target', 'keras_val_mae_target', 'keras_val_mape_target',
                                'keras_val_mae_last_step_target', 'keras_val_mape_last_step_target']
            bilstm_metrics.update({k: np.nan for k in nan_metrics_keys})

        # Сохранение модели и метаданных
        model_saved_successfully = False
        actual_model_dir_for_return = None
        model_filename = 'bilstm_predictor.keras'
        model_path = os.path.join(model_save_dir, model_filename)

        try:
            if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            model.save(model_path)
            logger.info(f"Финальная модель '{model_name_for_log}' сохранена: {model_path}")
            
            metadata = {
                "model_name": model_name_for_log,
                "prediction_horizon": prediction_horizon,
                "time_steps": config.TIME_STEPS,
                "features_input": config.FEATURES,
                "target_feature_model": current_target_feature,
                "target_feature_original": config.TARGET_FEATURE_ORIGINAL,
                "used_differencing": use_differencing_override,
                "differencing_column": config.DIFFERENCING_COLUMN if use_differencing_override else None,
                "differencing_order": config.DIFFERENCING_ORDER if use_differencing_override else None,
                "loss_function": "custom_weighted_mse",
                "loss_weight_base": config.LOSS_WEIGHT_BASE,
                "training_epochs_run": len(history.history.get('loss', [])),
                "test_split_ratio_used": test_split_ratio,
                "early_stopping_monitor_config": config.EARLY_STOPPING_MONITOR,
                "early_stopping_monitor_actual": early_stopping_monitor,
                "metrics_on_test_absolute_prices": bilstm_metrics 
            }
            metadata_path = os.path.join(model_save_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4, default=lambda x: None if pd.isna(x) else x)
            logger.info(f"Метаданные модели сохранены: {metadata_path}")

            model_saved_successfully = True
            actual_model_dir_for_return = model_save_dir
        except Exception as save_err:
            logger.error(f"Ошибка сохранения модели/метаданных для '{model_name_for_log}': {save_err}", exc_info=config.DEBUG_MODE)

        if model_saved_successfully:
            # Передаем y_test_for_ui_plot и y_pred_for_ui_plot (абсолютные цены)
            return (model_name_for_log, actual_model_dir_for_return, history.history, 
                    y_test_for_ui_plot, y_pred_for_ui_plot, bilstm_metrics,
                    df_processed)
        else: # Если сохранение не удалось, но обучение прошло
             return (model_name_for_log, None, history.history, 
                    y_test_for_ui_plot, y_pred_for_ui_plot, bilstm_metrics,
                    df_processed)

    except Exception as e:
        logger.error(f"Критическая ошибка в процессе обучения для '{model_name_for_log}': {e}", exc_info=True)
        return None