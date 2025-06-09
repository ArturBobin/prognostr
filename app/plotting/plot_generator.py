# app/plotting/plot_generator.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
from app.utils import config
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

def create_evaluation_plot(
    y_test_main: np.ndarray,    # Ожидаются абсолютные значения
    y_pred_main: np.ndarray,    # Ожидаются абсолютные значения
    model_name_main: str
) -> str | None:
    if y_test_main is None or y_pred_main is None or len(y_test_main) != len(y_pred_main) or len(y_test_main) == 0:
        logger.error(f"Некорректные или пустые данные для графика оценки ({model_name_main}). "
                     f"y_test: {type(y_test_main)}, y_pred: {type(y_pred_main)}")
        return None
    try:
        y_test_main_np = np.asarray(y_test_main).flatten()
        y_pred_main_np = np.asarray(y_pred_main).flatten()

        if y_test_main_np.size == 0 or y_pred_main_np.size == 0 :
             logger.error(f"Массивы для графика оценки пусты после преобразования ({model_name_main}).")
             return None
        
        mae_main = mean_absolute_error(y_test_main_np, y_pred_main_np)
        rmse_main = np.sqrt(mean_squared_error(y_test_main_np, y_pred_main_np))
        
        abs_y_test_main = np.abs(y_test_main_np)
        safe_denominator_main = np.maximum(abs_y_test_main, np.finfo(float).eps)
        mape_main = np.mean(np.abs((y_test_main_np - y_pred_main_np) / safe_denominator_main)) * 100
        
        title_model_part = model_name_main
        title = (f"Оценка модели '{title_model_part}' на тестовых данных<br>"
                 f"MAE: {mae_main:.2f} | RMSE: {rmse_main:.2f} | MAPE: {mape_main:.2f}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(y_test_main_np))), y=y_test_main_np,
            mode='lines', name=f'Реальные значения', line=dict(color='cyan')
        ))
        
        pred_legend_name = model_name_main.split(':')[0].strip() if ':' in model_name_main else model_name_main
        
        fig.add_trace(go.Scatter(
            x=list(range(len(y_pred_main_np))), y=y_pred_main_np,
            mode='lines', name=f'Прогноз {pred_legend_name}', line=dict(color='orange')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Временные шаги (тестовая выборка)',
            yaxis_title=f"Цена ({config.TARGET_FEATURE_ORIGINAL})",
            template='plotly_dark',
            legend_title_text='Данные',
            height=230,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        graph_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        logger.debug(f"График оценки для {model_name_main} сгенерирован.")
        return graph_html
    except Exception as e:
        logger.error(f"Ошибка создания графика оценки: {e}", exc_info=config.DEBUG_MODE)
        return None

def create_prediction_plot(
    df_history: pd.DataFrame,
    list_of_predicted_values: list[float], # Ожидается список АБСОЛЮТНЫХ предсказанных цен
    ticker: str,
    target_column_name_for_plot: str = config.TARGET_FEATURE_ORIGINAL, # Колонка для исторических цен
    show_sma_20: bool = False,
    show_sma_50: bool = False,
    show_bb: bool = False,
    show_rsi: bool = False
) -> str | None:
    if df_history is None or df_history.empty:
        logger.error("Невозможно построить график прогноза: DataFrame истории пуст.")
        return None
    if not list_of_predicted_values:
        logger.error("Невозможно построить график прогноза: список предсказанных значений пуст.")
        return None
    
    if target_column_name_for_plot not in df_history.columns:
        logger.error(f"Невозможно построить график прогноза: целевая колонка '{target_column_name_for_plot}' "
                     f"для отображения истории отсутствует в df_history. Доступные колонки: {df_history.columns.tolist()}")
        return (f"<body style='color:red; background-color:#1E1E1E;'>"
                f"Ошибка: Колонка '{target_column_name_for_plot}' не найдена в исторических данных.</body>")

    try:
        # Берем хвост исторических данных для отображения
        plot_data_history = df_history.tail(config.HISTORY_DAYS_TO_PLOT).copy()
        if plot_data_history.empty:
            logger.warning("DataFrame для исторической части графика прогноза пуст после взятия tail.")
            return "<body style='color:red; background-color:#1E1E1E;'>Нет исторических данных для отображения.</body>"
        
        logger.debug(f"Данные для графика прогноза ({ticker}): {len(plot_data_history)} строк истории, "
                     f"{len(list_of_predicted_values)} предсказанных шагов. "
                     f"Колонка для истории: '{target_column_name_for_plot}'.")

        last_history_date = plot_data_history.index[-1]
        prediction_dates = pd.bdate_range(start=last_history_date + pd.tseries.offsets.BDay(1), 
                                          periods=len(list_of_predicted_values))
        
        fig = make_subplots(specs=[[{"secondary_y": show_rsi}]]) # Создаем subplot, если RSI нужен

        # 1. Исторические данные (используем target_column_name_for_plot)
        fig.add_trace(go.Scatter(
            x=plot_data_history.index, 
            y=plot_data_history[target_column_name_for_plot],
            mode='lines',
            name=f'{target_column_name_for_plot} (история)',
            line=dict(color='cyan', width=2)
        ), secondary_y=False) 
        
        # 2. Предсказанные данные (list_of_predicted_values - это уже АБСОЛЮТНЫЕ цены)
        last_history_actual_value = plot_data_history[target_column_name_for_plot].iloc[-1]
        series_last_actual = pd.Series([last_history_actual_value], index=[last_history_date])
        series_predictions = pd.Series(list_of_predicted_values, index=prediction_dates)
        combined_prediction_line_y = pd.concat([series_last_actual, series_predictions])
        combined_prediction_line_x = combined_prediction_line_y.index

        fig.add_trace(go.Scatter(
            x=combined_prediction_line_x, 
            y=combined_prediction_line_y, 
            mode='lines',
            name=f'Прогноз на {len(list_of_predicted_values)} дней',
            line=dict(color='#FFD700', width=2),
            marker=dict(color='#FFD700', size=6)
        ), secondary_y=False)

        # 3. Технические индикаторы (на исторических данных, plot_data_history)
        if show_sma_20 and 'SMA_20' in plot_data_history.columns:
            fig.add_trace(go.Scatter(
                x=plot_data_history.index, y=plot_data_history['SMA_20'], mode='lines',
                name='SMA(20)', line=dict(color='magenta', width=1, dash='dot')
            ), secondary_y=False)
        if show_sma_50 and 'SMA_50' in plot_data_history.columns:
             fig.add_trace(go.Scatter(
                 x=plot_data_history.index, y=plot_data_history['SMA_50'], mode='lines',
                 name='SMA(50)', line=dict(color='yellow', width=1, dash='dash') 
             ), secondary_y=False)
        if show_bb and all(c in plot_data_history.columns for c in ['BB_Upper_20', 'BB_Lower_20', 'BB_Middle_20']):
             fig.add_trace(go.Scatter( 
                 x=plot_data_history.index, y=plot_data_history['BB_Middle_20'], mode='lines',
                 name='BB Средняя(20)', line=dict(color='rgba(255,105,180,0.6)', width=1, dash='longdash'),
                 legendgroup="bollinger"
             ), secondary_y=False)
             fig.add_trace(go.Scatter( 
                 x=plot_data_history.index, y=plot_data_history['BB_Upper_20'], mode='lines',
                 name='BB Верхняя', line=dict(color='rgba(152,251,152,0.5)', width=1),
                 legendgroup="bollinger", showlegend=False
             ), secondary_y=False)
             fig.add_trace(go.Scatter( 
                 x=plot_data_history.index, y=plot_data_history['BB_Lower_20'], mode='lines',
                 name='BB Нижняя', line=dict(color='rgba(152,251,152,0.5)', width=1),
                 fill='tonexty', fillcolor='rgba(152,251,152,0.1)', 
                 legendgroup="bollinger", showlegend=False
             ), secondary_y=False)
        
        # Настройка основной оси Y (используем target_column_name_for_plot)
        fig.update_layout(
            yaxis=dict(
                title_text=f"Цена ({target_column_name_for_plot}), RUB",
                gridcolor='rgba(255, 255, 255, 0.1)',
            )
        )

        # 4. RSI (если выбран)
        if show_rsi:
            if 'RSI_14' in plot_data_history.columns:
                fig.add_trace(go.Scatter(
                    x=plot_data_history.index, y=plot_data_history['RSI_14'], mode='lines',
                    name='RSI(14)', line=dict(color='orange', width=1.5)
                ), secondary_y=True)
                
                fig.update_layout(
                    yaxis2=dict( # Настройка вторичной оси Y для RSI
                        title=dict(text="RSI", font=dict(color='orange')),
                        range=[0, 100], 
                        overlaying='y', 
                        side='right',   
                        showgrid=False, 
                        tickfont=dict(color='orange')
                    )
                )
                # ИСПРАВЛЕНО ЗДЕСЬ: Используем fig.add_annotation для точного позиционирования
                # Сначала рисуем линии без встроенных аннотаций
                fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, secondary_y=True)
                fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, secondary_y=True)

                # Затем добавляем аннотации отдельно
                fig.add_annotation(
                    y=70,                       # Y-координата на оси RSI
                    x=0.98,                     # X-координата (0 до 1, от левого к правому краю бумаги)
                    xref="paper",               # X-координата относительно всей области графика
                    yref="y2",                  # Y-координата относительно вторичной оси Y (RSI)
                    text="Перекупленность (70)",
                    showarrow=False,
                    font=dict(color="red", size=10), # Уменьшим размер шрифта для компактности
                    bgcolor="rgba(30,30,30,0.75)", # Фон для лучшей читаемости
                    yanchor="bottom",             # Якорь текста снизу, чтобы текст был НАД y=70
                    xanchor="right"               # Якорь текста справа
                )
                
                fig.add_annotation(
                    y=30,                       # Y-координата на оси RSI
                    x=0.98,                     # X-координата
                    xref="paper",
                    yref="y2",
                    text="Перепроданность (30)",
                    showarrow=False,
                    font=dict(color="green", size=10),
                    bgcolor="rgba(30,30,30,0.75)",
                    yanchor="top",                # Якорь текста сверху, чтобы текст был ПОД y=30
                    xanchor="right"
                )
            else:
                logger.warning(f"RSI включен для {ticker}, но колонка 'RSI_14' отсутствует в исторических данных.")

        # Общие настройки макета графика
        prediction_end_date_str = prediction_dates[-1].strftime('%Y-%m-%d')
        title_text = (f'Анализ и Прогноз для {ticker.upper()} ({target_column_name_for_plot})<br>'
                      f'Прогноз до: {prediction_end_date_str} ({len(list_of_predicted_values)} дн.)')

        fig.update_layout(
            title=title_text, 
            xaxis_title='Дата',
            template='plotly_dark', 
            legend_title_text='Данные и Индикаторы', 
            height=350, # Немного увеличена высота для лучшего размещения аннотаций
            xaxis_rangeslider_visible=False, 
            hovermode='x unified', 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=50, t=80, b=50) 
        )
        
        graph_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        logger.debug(f"График прогноза ({len(list_of_predicted_values)} шагов) с индикаторами для {ticker} сгенерирован (RSI: {show_rsi}). "
                     f"Использована колонка истории: '{target_column_name_for_plot}'.")
        return graph_html

    except Exception as e:
        logger.error(f"Ошибка создания графика прогноза для {ticker}: {e}", exc_info=config.DEBUG_MODE)
        return None