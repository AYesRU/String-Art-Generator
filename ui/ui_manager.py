import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from typing import Dict, Any, Optional
from PIL import Image, ImageTk

from .app_state import AppState
from .utils import ToolTip, TOOLTIPS

if False:
    from .evolution_coordinator import EvolutionCoordinator
    from main import StringArtApp


class UIManager:
    """
    Отвечает за создание, компоновку и обновление всех виджетов UI.
    Не содержит логики вычислений, только управляет отображением.
    """
    def __init__(self, root: 'StringArtApp', state: AppState):
        self.root = root
        self.state = state
        self.coordinator: Optional['EvolutionCoordinator'] = None
        
        self.widgets: Dict[str, tk.Widget] = {}
        self.result_canvases: Dict[Any, tk.Canvas] = {}
        # Словари для хранения ссылок на объекты PhotoImage, чтобы их не удалил сборщик мусора.
        self.result_photo_refs: Dict[Any, Any] = {}
        self.main_island_progress_bars: Dict[int, ttk.Progressbar] = {}
        self.main_island_labels: Dict[int, ttk.Label] = {}
        
        self.chart_needs_update: bool = False
        self.ax: Optional[plt.Axes] = None
        self.chart_canvas: Optional[FigureCanvasTkAgg] = None

    def create_widgets(self):
        """Создает и размещает все элементы интерфейса."""
        # Главный разделитель окна по вертикали (верхняя часть / нижняя часть).
        main_pane = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # Верхняя часть, разделенная по горизонтали (панель управления / область отображения).
        top_area = ttk.PanedWindow(main_pane, orient=tk.HORIZONTAL)
        main_pane.add(top_area, weight=3)
        
        self._create_control_panel(top_area)
        self._create_display_area(top_area)
        
        # Нижняя часть для отображения прогресса и статистики.
        progress_frame = ttk.LabelFrame(main_pane, text="Прогресс Эволюции", height=250)
        main_pane.add(progress_frame, weight=1)
        
        self._create_stats_and_chart_area(progress_frame)

    def _create_control_panel(self, parent: ttk.PanedWindow):
        """Создает левую панель с настройками и кнопками управления."""
        container = ttk.Frame(parent, width=350)
        parent.add(container, weight=0)
        
        # Разделяем панель на две колонки для более компактного размещения.
        left_col = ttk.Frame(container)
        left_col.pack(side=tk.LEFT, fill=tk.Y, anchor='n', padx=(5, 2), expand=True)

        right_col = ttk.Frame(container)
        right_col.pack(side=tk.LEFT, fill=tk.Y, anchor='n', padx=(2, 5), expand=True)
        
        self._create_settings_section(left_col)
        self._create_algorithm_section(left_col)
        self._create_control_section(right_col)
        self._create_export_section(right_col)

    def _create_display_area(self, parent: ttk.PanedWindow):
        """Создает правую область для отображения исходного изображения и результатов."""
        display_area = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        parent.add(display_area, weight=1)
        
        # Область для исходного изображения.
        target_frame = ttk.Frame(display_area, width=300)
        display_area.add(target_frame, weight=1)
        canvas = tk.Canvas(target_frame, bg='white', relief='sunken', borderwidth=1)
        canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        canvas.bind("<Configure>", self.root.redraw_target_canvas)
        self.widgets['target_canvas'] = canvas
        
        # Область с вкладками для результатов.
        result_area = ttk.Frame(display_area)
        display_area.add(result_area, weight=3)
        notebook = ttk.Notebook(result_area)
        notebook.pack(fill=tk.BOTH, expand=True)
        self.widgets['result_notebook'] = notebook

    def _create_section(self, parent: tk.Widget, title: str) -> ttk.LabelFrame:
        """Вспомогательная функция для создания секции с рамкой и заголовком."""
        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.pack(fill=tk.X, pady=5, padx=5, anchor='n')
        return frame

    def _create_settings_section(self, parent: tk.Widget):
        """Создает секцию "Настройки изображения"."""
        frame = self._create_section(parent, "1. Настройки изображения")
        btn = ttk.Button(frame, text="Загрузить изображение") 
        btn.pack(fill=tk.X, pady=5)
        ToolTip(btn, TOOLTIPS['load_image'])
        self.widgets['load_image_button'] = btn
        
        sliders = [
            ('pins', "Гвозди", (100, 400)),
            ('lines', "Нити", (500, 8000)),
            ('thread_thickness', "Толщина (экспорт)", (1, 5)),
            ('thread_scale', "Масштаб нитей", (0.1, 5.0))
        ]
        for name, text, limits in sliders:
            self._create_slider_row(frame, name, text, limits)
        self.widgets['settings_frame'] = frame

    def _create_algorithm_section(self, parent: tk.Widget):
        """Создает секцию "Параметры алгоритма"."""
        frame = self._create_section(parent, "2. Параметры алгоритма")
        sliders = [
            ('num_islands', "Острова", (2, 10)),
            ('population_size', "Популяция", (10, 200)),
            ('stages', "Этапы", (2, 25)),
            ('mutation_rate', "Мутация (%)", (0, 100)),
            ('migration_interval', "Интервал миграции", (5, 50))
        ]
        for name, text, limits in sliders:
            self._create_slider_row(frame, name, text, limits)
        self.widgets['algo_frame'] = frame

    def _create_control_section(self, parent: tk.Widget):
        """Создает секцию "Управление" с кнопками Запуск/Пауза/Стоп."""
        frame = self._create_section(parent, "3. Управление")
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        buttons = [
            ('start', "▶ Запуск", self.coordinator.start_generation, tk.NORMAL),
            ('pause', "❚❚ Пауза", self.coordinator.toggle_pause, tk.DISABLED),
            ('stop', "■ Стоп", self.coordinator.stop_generation, tk.DISABLED)
        ]
        for name, text, command, state in buttons:
            btn = ttk.Button(btn_frame, text=text, command=command, state=state)
            btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
            ToolTip(btn, TOOLTIPS[name])
            self.widgets[f'{name}_button'] = btn

    def _create_export_section(self, parent: tk.Widget):
        """Создает секцию "Оптимизация и Экспорт"."""
        frame = self._create_section(parent, "4. Оптимизация и Экспорт")
        
        export_buttons = [
            ('prune', "Оптимизировать путь", self.coordinator.prune_chromosome),
            ('export_txt', "Экспорт пути (.txt)", self.coordinator.export_txt),
            ('export_png', "Экспорт изображения (.png)", self.coordinator.export_png),
            ('export_svg', "Экспорт в SVG (.svg)", self.coordinator.export_svg),
        ]
        
        for name, text, command in export_buttons:
            btn = ttk.Button(frame, text=text, command=command, state=tk.DISABLED)
            btn.pack(fill=tk.X, pady=2)
            self.widgets[f'{name}_button'] = btn
            ToolTip(btn, TOOLTIPS[name])

        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=5)
        
        save_buttons = [
            ('save_progress', "Сохранить прогресс (.pkl)", self.coordinator.save_progress),
            ('load_progress', "Загрузить прогресс (.pkl)", self.coordinator.load_progress)
        ]
        
        for name, text, command in save_buttons:
            state = tk.DISABLED if name == 'save_progress' else tk.NORMAL
            btn = ttk.Button(frame, text=text, command=command, state=state)
            btn.pack(fill=tk.X, pady=2)
            self.widgets[f'{name}_button'] = btn
            ToolTip(btn, TOOLTIPS[name])

    def _create_stats_and_chart_area(self, parent: tk.Widget):
        """Создает нижнюю панель со статистикой, прогресс-барами и графиком."""
        bottom_panel = ttk.Frame(parent)
        bottom_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Левая часть с текстовой статистикой и прогресс-барами.
        left_panel = ttk.Frame(bottom_panel)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        
        self.widgets['status_label'] = ttk.Label(left_panel, text="Ожидание...")
        self.widgets['status_label'].pack(anchor='w')
        self.widgets['fitness_label'] = ttk.Label(left_panel, text="")
        self.widgets['fitness_label'].pack(anchor='w', pady=(0, 5))
        self.widgets['compute_mode_label'] = ttk.Label(left_panel, text="Режим: --", font=('Segoe UI', 8))
        self.widgets['compute_mode_label'].pack(anchor='w', pady=(0, 10))
        
        for key, text in [('time', "Время:"), ('eta', "Осталось (ETA):"), ('gens', "Поколений:"), ('mig', "Миграций:")]:
            self.widgets[key+'_label'] = self._create_stat_row(left_panel, text)

        self.widgets['main_progress_bars_frame'] = ttk.Frame(left_panel)
        self.widgets['main_progress_bars_frame'].pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Правая часть с графиком Matplotlib.
        fig = Figure(facecolor='#f0f0f0')
        self.ax = fig.add_subplot(111)
        fig.tight_layout(pad=0.5)
        self.chart_canvas = FigureCanvasTkAgg(fig, master=bottom_panel)
        self.chart_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _create_slider_row(self, parent: tk.Widget, name: str, text: str, limits: tuple):
        """Вспомогательная функция для создания строки с подписью, слайдером и полем ввода."""
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=2)
        lbl = ttk.Label(f, text=f"{text}:", width=17)
        lbl.pack(side=tk.LEFT)
        entry = ttk.Entry(f, textvariable=self.state.vars[name], width=5)
        entry.pack(side=tk.RIGHT)
        
        is_double = isinstance(self.state.vars[name], tk.DoubleVar)
        command = (lambda v, n=name: self.state.vars[n].set(float(v))) if is_double else (lambda v, n=name: self.state.vars[n].set(int(float(v))))
        
        scale = ttk.Scale(f, from_=limits[0], to=limits[1], variable=self.state.vars[name], orient=tk.HORIZONTAL, command=command)
        scale.pack(fill=tk.X, expand=True, padx=5)
        for widget in [lbl, entry, scale]:
            ToolTip(widget, TOOLTIPS[name])
            
    def _create_stat_row(self, parent, text):
        """Вспомогательная функция для создания строки статистики."""
        row_frame = ttk.Frame(parent)
        row_frame.pack(fill='x')
        ttk.Label(row_frame, text=text, font=('Segoe UI', 9, 'bold'), width=15, anchor='w').pack(side=tk.LEFT)
        value_label = ttk.Label(row_frame, text="--", font=('Segoe UI', 9), anchor='w')
        value_label.pack(side=tk.LEFT)
        return value_label

    def on_result_canvas_resize(self, event, key):
        """
        Обработчик изменения размера холста с результатом.
        Запрашивает перерисовку изображения в фоновом потоке.
        """
        chromosome = self.coordinator.best_chromosomes_by_key.get(key)
        if chromosome:
            # Запрашиваем рендер в стандартном разрешении (500px).
            # Он будет асинхронно отрендерен и затем отмасштабирован под размер холста.
            render_data = (chromosome, self.state.get_config()['pins'], self.state.vars['thread_scale'].get())
            self.coordinator.request_canvas_update(key, render_data, {'size': 500})

    def setup_for_new_run(self, num_islands):
        """Подготавливает UI к новому запуску: создает вкладки и прогресс-бары."""
        self.setup_notebook_tabs(num_islands)
        self.setup_progress_bars(num_islands)
        self.chart_needs_update = True
        self.update_chart()

    def setup_notebook_tabs(self, num_islands):
        """Создает или пересоздает вкладки для отображения результатов."""
        notebook = self.widgets['result_notebook']
        for i in notebook.tabs(): notebook.forget(i)
        self.result_canvases.clear()
        
        self._create_notebook_tab('overall', "Лучший результат", notebook)
        
        for i in range(num_islands):
            self._create_notebook_tab(i, f"Остров {i+1}", notebook)
        
        # Создаем вкладку для анализа расхождений.
        vis_tab = ttk.Frame(notebook)
        notebook.add(vis_tab, text="Анализ Расхождений")
        # Обновленный, более информативный текст.
        ssim_frame = ttk.LabelFrame(vis_tab, text="Карта расхождений: где нитей много (красный) или мало (синий)", padding=5)
        ssim_frame.pack(fill=tk.BOTH, expand=True)
        ssim_canvas = tk.Canvas(ssim_frame, bg='white', highlightthickness=0)
        ssim_canvas.pack(fill=tk.BOTH, expand=True)
        self.result_canvases['ssim_map'] = ssim_canvas
        
        if self.state.overall_best_chromosome:
            self.root.after(100, lambda: self.on_result_canvas_resize(None, 'overall'))

    def _create_notebook_tab(self, key, text, notebook):
        """Вспомогательная функция для создания одной вкладки с холстом."""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=text)
        canvas = tk.Canvas(tab, bg='white', highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.bind("<Configure>", lambda e, k=key: self.on_result_canvas_resize(e, k))
        self.result_canvases[key] = canvas

    def setup_progress_bars(self, num_islands):
        """Создает или пересоздает прогресс-бары для каждого острова."""
        pb_frame = self.widgets['main_progress_bars_frame']
        for widget in pb_frame.winfo_children(): widget.destroy()
        self.main_island_progress_bars.clear()
        self.main_island_labels.clear()
        
        for i in range(num_islands):
            frame = ttk.Frame(pb_frame)
            frame.pack(fill=tk.X, expand=True, pady=1)
            label = ttk.Label(frame, text="Ожидание...", width=15)
            label.pack(side=tk.LEFT, padx=(0, 5))
            self.main_island_labels[i] = label
            pb = ttk.Progressbar(frame, orient='horizontal', mode='determinate')
            pb.pack(fill=tk.X, expand=True)
            self.main_island_progress_bars[i] = pb

    def update_chart(self):
        """Обновляет график эволюции фитнеса."""
        if not self.ax or not self.chart_canvas: return
        self.ax.clear()
        num_islands = self.state.get_config()['num_islands']
        colors = plt.cm.viridis(np.linspace(0, 1, num_islands)) if num_islands > 0 else []
        
        for i in range(num_islands):
            data = self.state.chart_data_by_island.get(i, [])
            if data:
                gens, fitnesses = zip(*data)
                self.ax.plot(gens, fitnesses, label=f'Остров {i+1}', color=colors[i])
        
        self.ax.set_title("Эволюция фитнеса")
        self.ax.set_xlabel("Поколение")
        self.ax.set_ylabel("Фитнес")
        if any(self.state.chart_data_by_island.values()): self.ax.legend(loc='upper left')
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.set_facecolor('white')
        self.chart_canvas.figure.tight_layout(pad=0.5)
        self.chart_canvas.draw()
        self.chart_needs_update = False

    def toggle_ui_state(self, is_running: bool):
        """
        Переключает состояние всех интерактивных элементов UI
        (включает/выключает их) в зависимости от того, идет ли процесс генерации.
        """
        self.state.is_running = is_running
        state = tk.DISABLED if is_running else tk.NORMAL
        
        # Блокируем/разблокируем все слайдеры и поля ввода.
        for frame_key in ['settings_frame', 'algo_frame']:
            frame = self.widgets.get(frame_key)
            if frame:
                for child in frame.winfo_children():
                    try:
                        child.configure(state=state)
                        for sub_child in child.winfo_children():
                             sub_child.configure(state=state)
                    except (tk.TclError, AttributeError):
                        pass # Игнорируем виджеты, у которых нет свойства 'state'.

        self.widgets['start_button'].config(state=state)
        self.widgets['load_progress_button'].config(state=state)
        
        control_state = tk.NORMAL if is_running else tk.DISABLED
        self.widgets['stop_button'].config(state=control_state)
        self.widgets['pause_button'].config(state=control_state)

        # Кнопки экспорта активны только когда процесс завершен и есть результат.
        export_state = tk.NORMAL if not is_running and self.state.overall_best_chromosome else tk.DISABLED
        export_keys = ['prune_button', 'export_txt_button', 'export_png_button', 'save_progress_button', 'export_svg_button']
        for key in export_keys:
            if key in self.widgets: self.widgets[key].config(state=export_state)
            
        if not is_running:
            self.widgets['pause_button'].config(text="❚❚ Пауза")
            for i, pb in self.main_island_progress_bars.items():
                pb['value'] = 0
                if i in self.main_island_labels: self.main_island_labels[i].config(text=f"Завершено")

    def update_island_progress(self, island_id: int, stage: int, patience: int, limit: int):
        """Обновляет текст и значение прогресс-бара для конкретного острова."""
        if island_id in self.main_island_labels:
            self.main_island_labels[island_id].config(text=f"Этап {stage + 1}: {patience}/{limit}")
        if island_id in self.main_island_progress_bars:
            pb = self.main_island_progress_bars[island_id]
            pb['maximum'] = limit
            pb['value'] = patience

    def update_best_fitness_display(self, fitness: float):
        """Обновляет текстовое поле с лучшим значением фитнеса."""
        self.widgets['fitness_label'].config(text=f"Лучший фитнес: {fitness:.6f}")

    def update_stats_display(self, current_state: AppState):
        """Обновляет статистику (время, ETA и т.д.)."""
        elapsed = time.time() - current_state.start_time
        self.widgets['time_label'].config(text=time.strftime('%H:%M:%S', time.gmtime(elapsed)))
        self.widgets['gens_label'].config(text=f"{sum(current_state.generation_counters.values())}")
        self.widgets['mig_label'].config(text=f"{current_state.migration_count}")
        
        # Расчет ETA (примерное оставшееся время).
        total_progress = 0
        num_islands = current_state.get_config()['num_islands']
        total_stages = current_state.get_config()['stages']
        if not num_islands or not total_stages: return
        
        for i in range(num_islands):
            stage, limit = current_state.island_progress_data.get(i, (0, 1))
            pb = self.main_island_progress_bars.get(i)
            patience = pb['value'] if pb else 0
            progress_per_stage = 1.0 / total_stages
            total_progress += stage * progress_per_stage + (patience / limit) * progress_per_stage
        
        avg_progress = total_progress / num_islands
        if avg_progress > 0.01:
            eta = (elapsed / avg_progress) * (1 - avg_progress) if avg_progress < 1 else 0
            self.widgets['eta_label'].config(text=time.strftime('%H:%M:%S', time.gmtime(eta)))
        else:
            self.widgets['eta_label'].config(text="Расчет...")

    def update_canvas_image(self, key: Any, img_pil: Image.Image):
        """Отображает переданное изображение PIL на указанном холсте."""
        canvas = self.result_canvases.get(key)
        if not canvas: return
        
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 2 or h < 2: return # Не пытаемся рисовать на свернутом холсте.
        
        # Масштабируем изображение под размер холста с сохранением пропорций.
        img_pil.thumbnail((w, h), Image.Resampling.LANCZOS)
        photo_ref = ImageTk.PhotoImage(img_pil)
        # Важно сохранить ссылку, иначе изображение не отобразится.
        self.result_photo_refs[key] = photo_ref
        canvas.delete("all")
        canvas.create_image(w//2, h//2, anchor=tk.CENTER, image=photo_ref)

    def update_title(self):
        """Обновляет заголовок окна, добавляя имя открытого файла."""
        base_title = "String Art Generator"
        if self.state.original_image_path:
            filename = os.path.basename(self.state.original_image_path)
            self.root.title(f"{base_title} - [{filename}]")
        else:
            self.root.title(base_title)
