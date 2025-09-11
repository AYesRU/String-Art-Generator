import tkinter as tk

# Словарь, в котором хранятся все тексты для всплывающих подсказок.
# Ключи соответствуют названиям виджетов или их логическому назначению.
TOOLTIPS = {
    'load_image': "Открыть файл изображения (JPG, PNG).\nРекомендуются контрастные портреты.",
    'pins': "Количество точек (гвоздей) по периметру.\nБольше = выше детализация.",
    'lines': "Общее количество линий (нитей).\nБольше = темнее и плотнее изображение.",
    'thread_thickness': "Толщина нитей для отображения и экспорта.\nНе влияет на скорость генерации.",
    
    'num_islands': "Количество независимых популяций (потоков).\nУскоряет поиск, используя больше ядер CPU.",
    'population_size': "Количество вариантов на каждом 'острове'.\nБольше = лучше поиск, но медленнее.",
    'stages': "Количество стадий оптимизации.\nАлгоритм последовательно уточняет результат.",
    'mutation_rate': "Вероятность случайных изменений (%)\nПомогает избегать 'застревания' на плохом решении.",
    'migration_interval': "Частота обмена лучшими вариантами между островами (в поколениях).",
    
    'start': "Начать процесс генерации.",
    'pause': "Приостановить/возобновить генерацию.",
    'stop': "Остановить генерацию и сохранить текущий лучший результат.",
    
    'prune': "Удалить из пути лишние линии вида A->B->A,\nсокращая итоговую длину нити.",
    'export_txt': "Сохранить последовательность номеров гвоздей для ручной работы.",
    'export_png': "Сохранить итоговую картинку в высоком разрешении (2000x2000px).",
    'export_svg': "Сохранить результат в векторном формате SVG,\nидеальном для масштабирования и печати.",
}

class ToolTip:
    """
    Класс для создания всплывающих подсказок для виджетов Tkinter.
    """
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        # Привязываем события "мышь вошла" и "мышь ушла" к методам показа/скрытия.
        self.widget.bind("<Enter>", self.showtip)
        self.widget.bind("<Leave>", self.hidetip)

    def showtip(self, event=None):
        """Показывает окно с подсказкой."""
        if self.tipwindow or not self.text:
            return
        # Получаем координаты виджета, чтобы расположить подсказку рядом.
        x, y, _, _ = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 20
        
        # Создаем новое окно Toplevel без рамок.
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{int(x)}+{int(y)}")
        
        label = tk.Label(
            tw, text=self.text, justify=tk.LEFT,
            background="#ffffe0", relief=tk.SOLID, borderwidth=1,
            font=("tahoma", "8", "normal")
        )
        label.pack(ipadx=4)

    def hidetip(self, event=None):
        """Уничтожает окно с подсказкой."""
        if self.tipwindow:
            self.tipwindow.destroy()
        self.tipwindow = None