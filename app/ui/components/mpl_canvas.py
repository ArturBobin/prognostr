# app/ui/components/mpl_canvas.py

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy
import logging

matplotlib.rc('axes', edgecolor='#AAAAAA', labelcolor='#EAEAEA', titlecolor='#EAEAEA')
matplotlib.rc('xtick', color='#AAAAAA', labelsize=8)
matplotlib.rc('ytick', color='#AAAAAA', labelsize=8)
matplotlib.rc('figure', facecolor='#252526', edgecolor='#252526')
matplotlib.rc('axes', facecolor='#2D2D2D')
matplotlib.rc('grid', color='#4A4A4A', linestyle=':')
matplotlib.rc('legend', facecolor='#2D2D2D', edgecolor='#5A5A5A', labelcolor='#EAEAEA', fontsize=8)
matplotlib.rc('lines', linewidth=1.5)


class MplCanvas(FigureCanvasQTAgg):
    """Встраиваемый холст Matplotlib для PyQt/PySide."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        try:
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            self.axes = self.fig.add_subplot(111) # Основная ось Y (для Loss)
            self.ax2 = None
            super(MplCanvas, self).__init__(self.fig)
            self.setParent(parent)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.updateGeometry()
            logging.debug("MplCanvas инициализирован.")
        except Exception as e:
             logging.error(f"Ошибка инициализации MplCanvas: {e}", exc_info=True)

    def plot_training_history(self, history_dict):
        """Отрисовывает графики обучения (Loss, MAE/MAPE)."""
        try:
            # Полная очистка перед отрисовкой
            self.axes.clear()
            if self.ax2:
                self.ax2.clear()
                self.ax2.set_visible(False)

            epochs = range(1, len(history_dict.get('loss', [])) + 1)
            if not epochs: # Если нет данных для эпох, ничего не рисуем
                self.draw_idle()
                return

            lines = [] # Для сбора линий для легенды
            labels = [] # Для сбора меток для легенды

            # График потерь (Loss) на основной оси self.axes
            if 'loss' in history_dict:
                line, = self.axes.plot(epochs, history_dict['loss'], 'b-', label='Training Loss')
                lines.append(line)
                labels.append(line.get_label())
            if 'val_loss' in history_dict:
                line, = self.axes.plot(epochs, history_dict['val_loss'], 'r--', label='Validation Loss')
                lines.append(line)
                labels.append(line.get_label())
            
            self.axes.set_ylabel('Loss', color='#EAEAEA')
            self.axes.grid(True) # Сетка для основной оси

            # Дополнительная ось Y для метрики (MAE/MAPE)
            metric_key = None
            val_metric_key = None
            # Ищем первую доступную пару метрик
            possible_metrics = [
                ('mae', 'val_mae'), 
                ('mean_absolute_error', 'val_mean_absolute_error'),
                ('mape', 'val_mape'), 
                ('mean_absolute_percentage_error', 'val_mean_absolute_percentage_error')
            ]
            
            for m_key, val_m_key in possible_metrics:
                if m_key in history_dict and val_m_key in history_dict:
                    metric_key = m_key
                    val_metric_key = val_m_key
                    break
            
            if metric_key and val_metric_key:
                if self.ax2 is None: # Создаем вторую ось только если ее нет
                    self.ax2 = self.axes.twinx()
                else:
                    self.ax2.set_visible(True) # Делаем видимой, если уже есть

                if metric_key in history_dict: # Доп. проверка
                    line, = self.ax2.plot(epochs, history_dict[metric_key], 'g-', label=f'Training {metric_key.upper()}')
                    lines.append(line)
                    labels.append(line.get_label())
                if val_metric_key in history_dict: # Доп. проверка
                    line, = self.ax2.plot(epochs, history_dict[val_metric_key], 'y--', label=f'Validation {metric_key.upper()}') # Используем metric_key для названия
                    lines.append(line)
                    labels.append(line.get_label())
                
                self.ax2.set_ylabel(metric_key.upper(), color='#EAEAEA')
                self.ax2.tick_params(axis='y', labelcolor='#EAEAEA')
            elif self.ax2: # Если метрики нет, а ax2 был создан ранее
                self.ax2.set_visible(False)


            # Настройка основной оси X
            self.axes.set_xlabel('Epoch', color='#EAEAEA')
            self.axes.set_title('Training History', color='#EAEAEA', fontsize=10)
            self.axes.tick_params(axis='x', labelcolor='#EAEAEA')
            self.axes.tick_params(axis='y', labelcolor='#EAEAEA') # Для основной оси (Loss)

            if lines:
                self.axes.legend(lines, labels, loc='best')

            self.fig.tight_layout()
            self.draw_idle()
            logging.debug("График обучения обновлен.")
        except Exception as e:
             logging.error(f"Ошибка обновления графика обучения: {e}", exc_info=True)

    def clear_plot(self):
        """Очищает график."""
        try:
            self.axes.clear()
            if self.ax2:
                self.ax2.clear()
                self.ax2.set_visible(False)

            self.axes.set_xlabel('Epoch', color='#EAEAEA')
            self.axes.set_ylabel('Value', color='#EAEAEA')
            self.axes.set_title('Training History', color='#EAEAEA', fontsize=10)
            self.axes.grid(True)
            
            self.draw_idle()
            logging.debug("График обучения очищен.")
        except Exception as e:
            logging.error(f"Ошибка очистки графика обучения: {e}", exc_info=True)