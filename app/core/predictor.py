# app/core/predictor.py
import tensorflow as tf
import numpy as np
import joblib
import os
import json
import logging
from app.utils import config
from app.core.model_builder import custom_weighted_mse, mae_last_step, mape_last_step

logger = logging.getLogger(__name__)

# Загружает метаданные, возвращает список предсказаний
def make_prediction(input_sequence: np.ndarray, model_dir: str) -> list[float] | None:
    model_name_for_log = os.path.basename(model_dir)
    logger.debug(f"Попытка сделать предсказание с использованием модели: '{model_name_for_log}' из директории: {model_dir}")

    if not os.path.isdir(model_dir):
        logger.error(f"Директория модели не найдена: {model_dir}")
        return None

    model_path = os.path.join(model_dir, 'bilstm_predictor.keras')
    target_scaler_path = os.path.join(model_dir, 'target_scaler.pkl')
    metadata_path = os.path.join(model_dir, 'metadata.json')

    if not os.path.exists(model_path):
        logger.error(f"Файл модели не найден: {model_path}")
        return None
    if not os.path.exists(target_scaler_path):
        logger.error(f"Файл целевого скейлера не найден: {target_scaler_path}")
        return None
    if not os.path.exists(metadata_path):
        logger.error(f"Файл метаданных 'metadata.json' не найден: {metadata_path}")
        return None

    try:
        # Загрузка метаданных
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        prediction_horizon = metadata.get('prediction_horizon')
        if prediction_horizon is None or not isinstance(prediction_horizon, int) or prediction_horizon <= 0:
            logger.error(f"Некорректное или отсутствующее значение 'prediction_horizon' в метаданных: {metadata_path}")
            return None
        logger.info(f"Метаданные загружены. Модель '{model_name_for_log}' предсказывает на {prediction_horizon} шагов.")

        # Передаем custom_objects при загрузке
        custom_objects_dict = {
            "custom_weighted_mse": custom_weighted_mse,
            "mae_last_step": mae_last_step,
            "mape_last_step": mape_last_step
        }

        # Загрузка модели
        logger.debug(f"Загрузка модели из: {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.debug(f"Модель '{model_name_for_log}' успешно загружена.")
        
        # Проверка выходного слоя модели
        expected_output_shape = (None, prediction_horizon)
        if model.output_shape != expected_output_shape:
            logger.warning(f"Выходной слой модели '{model.output_shape}' не соответствует ожидаемому '{expected_output_shape}' "
                           f"на основе prediction_horizon={prediction_horizon} из метаданных. Возможна ошибка.")

        # Загрузка скейлера
        logger.debug(f"Загрузка целевого скейлера из: {target_scaler_path}")
        target_scaler = joblib.load(target_scaler_path)
        logger.debug(f"Целевой скейлер для '{model_name_for_log}' успешно загружен.")

        logger.debug(f"Выполнение предсказания для последовательности формы: {input_sequence.shape}")
        predicted_scaled = model.predict(input_sequence, verbose=0) # Выход: (1, prediction_horizon)
        logger.debug(f"Предсказанные масштабированные значения (форма {predicted_scaled.shape}): {predicted_scaled}")

        predicted_prices_array = target_scaler.inverse_transform(predicted_scaled) # Выход: (1, prediction_horizon)
        predicted_prices_list = predicted_prices_array.flatten().tolist() # Преобразуем в список float
        
        logger.info(f"Предсказанные значения (после inverse_transform) для '{model_name_for_log}' на {prediction_horizon} шагов: "
                    f"{[f'{p:.4f}' for p in predicted_prices_list]}")

        return predicted_prices_list

    except Exception as e:
        logger.error(f"Ошибка во время выполнения предсказания (модель: '{model_name_for_log}'): {e}", exc_info=config.DEBUG_MODE)
        return None