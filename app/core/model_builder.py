# app/core/model_builder.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.utils import register_keras_serializable 
import logging
from app.utils import config 


logger = logging.getLogger(__name__)

# --- Кастомные объекты Keras ---

# Декорируем кастомные объекты
@register_keras_serializable(package="Custom", name="get_custom_weights_fn")
def get_custom_weights(horizon: tf.Tensor | int, base: float = 1.15, normalize_sum_to_horizon: bool = True) -> tf.Tensor:
    if not isinstance(horizon, tf.Tensor):
        horizon = tf.cast(horizon, tf.int32)
    raw_weights = tf.pow(base, tf.range(0, tf.cast(horizon, tf.float32), dtype=tf.float32))
    if normalize_sum_to_horizon:
        sum_raw_weights = tf.reduce_sum(raw_weights)
        weights = (raw_weights / tf.maximum(sum_raw_weights, 1e-9)) * tf.cast(horizon, tf.float32)
    else:
        weights = raw_weights
    return weights


@register_keras_serializable(package="CustomLosses", name="weighted_mse")
def custom_weighted_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    horizon = tf.shape(y_true)[1] 
    weight_base = getattr(config, 'LOSS_WEIGHT_BASE', 1.15) 
    weights = get_custom_weights(horizon, base=weight_base, normalize_sum_to_horizon=True)
    squared_errors = tf.square(y_true - y_pred)
    weighted_squared_errors = squared_errors * weights
    return tf.reduce_mean(tf.reduce_mean(weighted_squared_errors, axis=1))


@register_keras_serializable(package="CustomMetrics", name="mae_ls")
def mae_last_step(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    last_step_true = y_true[:, -1]
    last_step_pred = y_pred[:, -1]
    absolute_errors = tf.abs(last_step_true - last_step_pred)
    return tf.reduce_mean(absolute_errors)


@register_keras_serializable(package="CustomMetrics", name="mape_ls")
def mape_last_step(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    last_step_true = y_true[:, -1]
    last_step_pred = y_pred[:, -1]
    abs_true = tf.abs(last_step_true)
    epsilon = tf.keras.backend.epsilon() 
    denominator = tf.maximum(abs_true, epsilon)
    absolute_percentage_errors = tf.abs((last_step_true - last_step_pred) / denominator) * 100.0
    return tf.reduce_mean(absolute_percentage_errors)

# --- Построение модели ---

def build_bilstm_model(input_shape_tuple: tuple, prediction_horizon: int) -> tf.keras.Model:
    logger.debug(f"Создание модели BiLSTM (Functional API) с input_shape={input_shape_tuple}, prediction_horizon={prediction_horizon}")

    inputs = Input(shape=input_shape_tuple, name="input_layer")

    x = Bidirectional(LSTM(units=256, return_sequences=True, name="bilstm_1"))(inputs)
    x = Dropout(getattr(config, 'DROPOUT_RATE', 0.2), name="dropout_1")(x)

    x = Bidirectional(LSTM(units=128, return_sequences=True, name="bilstm_2"))(x)
    x = Dropout(getattr(config, 'DROPOUT_RATE', 0.2), name="dropout_2")(x)

    x = Bidirectional(LSTM(units=64, return_sequences=False, name="bilstm_3"))(x) 
    x = Dropout(getattr(config, 'DROPOUT_RATE', 0.2), name="dropout_3")(x)
    
    outputs = Dense(units=prediction_horizon, name="output_layer")(x)
    
    model_name = f"BiLSTM_Stock_Predictor_h{prediction_horizon}"
    model = Model(inputs=inputs, outputs=outputs, name=model_name)

    model.compile(
        optimizer='adam', 
        loss=custom_weighted_mse, 
        metrics=[
            'mae', # Стандартная MAE (усредненная Keras'ом по всем шагам)
            'mape',# Стандартная MAPE (усредненная Keras'ом по всем шагам)
            mae_last_step, 
            mape_last_step 
        ]
    )

    logger.info(f"Модель '{model_name}' создана и скомпилирована с custom_weighted_mse и метриками для последнего шага.")

    if config.DEBUG_MODE:
        stringlist = []
        model.summary(print_fn=lambda s, **kwargs: stringlist.append(s))
        model_summary = "\n".join(stringlist)
        logger.debug(f"Структура модели:\n{model_summary}")

    return model