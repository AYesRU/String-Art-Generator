# -*- coding: utf-8 -*-
"""
Графический интерфейс String Art Studio Pro (Clean Light Theme),
потокобезопасный координатор синтеза, интерактивный плеер и живая аналитика Matplotlib.
"""

import logging
import os
import queue
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_stringart"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from . import (
    APP_TITLE,
    CANVAS_SHAPES,
    COLOR_MODES,
    IMAGE_SIZE,
    PRESETS,
    TOOLTIPS,
)
from .audio import SAPI_AVAILABLE, AudioAssistant
from .engine import (
    CUDA_DEVICE_NAME,
    GPU_ENABLED_FLAG,
    SCIPY_AVAILABLE,
    FitnessCalculator,
    GreedyInitializer,
    IslandManagerThread,
    prune_chromosome_loops,
)
from .exporters import (
    export_csv,
    export_drill_template_svg,
    export_gcode_cnc,
    export_html_assistant,
    export_instructions_txt,
    export_png_ultra_hd,
    export_svg_vector,
)
from .geometry import CanvasManager
from .preprocessing import auto_recommend_parameters, prepare_target_image_bundle
from .radon import SKIMAGE_AVAILABLE, RadonGuide


def enable_windows_high_dpi() -> None:
    """Включает поддержку четкого масштабирования HiDPI на Windows."""
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


class AppState:
    """Хранилище параметров и текущего состояния синтеза/плеера."""

    def __init__(self):
        self.vars: Dict[str, tk.Variable] = {
            "preset": tk.StringVar(value="Авто (по изображению)"),
            "clahe_clip": tk.DoubleVar(value=1.3),
            "gamma": tk.DoubleVar(value=0.82),
            "canvas_shape": tk.StringVar(value="Круг"),
            "color_mode": tk.StringVar(value="Монохром (Черная нить)"),
            "physical_diameter": tk.DoubleVar(value=50.0),
            "pins": tk.IntVar(value=260),
            "lines": tk.IntVar(value=1400),
            "line_weight": tk.IntVar(value=72),
            "thread_thickness": tk.IntVar(value=1),
            "fearless_factor": tk.IntVar(value=94),
            "num_islands": tk.IntVar(value=min(4, max(2, (os.cpu_count() or 4) // 2))),
            "population_size": tk.IntVar(value=24),
            "stages": tk.IntVar(value=5),
            "mutation_rate": tk.IntVar(value=18),
            "migration_interval": tk.IntVar(value=20),
            "structural_metric": tk.StringVar(
                value="Преобразование расстояний" if SCIPY_AVAILABLE else "Контуры"
            ),
            "contour_weight": tk.IntVar(value=55),
            "memetic_enabled": tk.BooleanVar(value=True),
            "memetic_intensity": tk.IntVar(value=20),
            "memetic_depth": tk.IntVar(value=25),
            "radon_enabled": tk.BooleanVar(value=True),
            "radon_influence": tk.IntVar(value=35),
            "step_scrubber": tk.IntVar(value=1400),
            "active_layer": tk.StringVar(value="Черная нить"),
            "show_active_chord": tk.BooleanVar(value=True),
            "voice_enabled": tk.BooleanVar(value=False),
            "split_compare": tk.IntVar(value=0),
        }
        self._defaults: Dict[str, Any] = {k: v.get() for k, v in self.vars.items()}
        self.original_image: Optional[Image.Image] = None
        self.original_image_path: Optional[str] = None
        self.target_image_tk: Optional[Any] = None
        self.preprocessed_target_np: Optional[np.ndarray] = None
        self.preprocessed_rgb_np: Optional[np.ndarray] = None
        self.layers: List[Dict[str, Any]] = []
        self.active_layer_idx: int = 0
        self.overall_best_chromosome: Optional[List[int]] = None
        self.overall_best_fitness: float = -1.0
        self.layer_best_fitnesses: Dict[int, float] = {}
        self.omp_baseline_fitness: float = 0.0
        self.is_running: bool = False
        self.is_paused: bool = False
        self.start_time: float = 0.0
        self.migration_count: int = 0
        self.generation_counters: Dict[int, int] = {}
        self.chart_data_by_island: Dict[int, List[Tuple[int, float]]] = {}
        self.island_best_fitnesses: Dict[int, float] = {}
        self.island_progress_data: Dict[int, Tuple[int, int]] = {}
        self.aggregated_mut_stats: Dict[str, Dict[str, int]] = {}
        self.state_lock = threading.Lock()

    def safe_get(self, key: str) -> Any:
        try:
            return self.vars[key].get()
        except (tk.TclError, ValueError, KeyError):
            return self._defaults.get(key)

    def get_config(self) -> Dict[str, Any]:
        return {k: self.safe_get(k) for k in self.vars}

    def reset_for_new_run(self, reset_layers: bool = True) -> None:
        if reset_layers:
            self.layers = []
            self.active_layer_idx = 0
            self.overall_best_chromosome = None
        self.overall_best_fitness = -1.0
        self.layer_best_fitnesses = {}
        self.omp_baseline_fitness = 0.0
        self.migration_count = 0
        self.is_paused = False
        self.start_time = time.time()
        num_islands = max(1, int(self.safe_get("num_islands")))
        self.chart_data_by_island = {i: [] for i in range(num_islands)}
        self.island_best_fitnesses = {i: -1.0 for i in range(num_islands)}
        self.generation_counters = {i: 0 for i in range(num_islands)}
        self.island_progress_data = {i: (0, 1) for i in range(num_islands)}
        self.aggregated_mut_stats = {}


class ToolTip:
    """Компактная всплывающая подсказка при наведении на элемент интерфейса."""

    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tipwindow: Optional[tk.Toplevel] = None
        self.widget.bind("<Enter>", self.showtip, add="+")
        self.widget.bind("<Leave>", self.hidetip, add="+")

    def showtip(self, event=None) -> None:
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 22
        y = self.widget.winfo_rooty() + 22
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{int(x)}+{int(y)}")
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            foreground="#1f2328",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Segoe UI", 8, "normal"),
        )
        label.pack(ipadx=6, ipady=3)

    def hidetip(self, event=None) -> None:
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


class UIManager:
    """Управляет виджетами, светлой темой, графиками Matplotlib и холстами предпросмотра."""

    def __init__(self, root: "StringArtApp", state: AppState):
        self.root = root
        self.state = state
        self.coordinator: Optional["EvolutionCoordinator"] = None
        self.widgets: Dict[str, tk.Widget] = {}
        self.result_canvases: Dict[Any, tk.Canvas] = {}
        self.result_photo_refs: Dict[Any, Any] = {}
        self.raw_pil_images: Dict[Any, Image.Image] = {}
        self.main_island_progress_bars: Dict[int, ttk.Progressbar] = {}
        self.main_island_labels: Dict[int, ttk.Label] = {}
        self.chart_needs_update: bool = False
        self.ax_fitness: Optional[plt.Axes] = None
        self.ax_mut: Optional[plt.Axes] = None
        self.chart_canvas: Optional[FigureCanvasTkAgg] = None
        self._slider_limits: Dict[str, Tuple[float, float, bool]] = {}

    def create_widgets(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        BG_MAIN = "#f6f8fa"
        BG_SIDE = "#ffffff"
        TEXT_MAIN = "#1f2328"
        TEXT_SUB = "#57606a"
        ACCENT = "#0969da"
        BORDER = "#d0d7de"

        self.root.configure(bg=BG_MAIN)
        style.configure(".", background=BG_MAIN, foreground=TEXT_MAIN, font=("Segoe UI", 9))
        style.configure("TFrame", background=BG_MAIN)
        style.configure("Card.TFrame", background=BG_SIDE)
        style.configure("TLabel", background=BG_MAIN, foreground=TEXT_MAIN)
        style.configure("TLabelframe", background=BG_SIDE, bordercolor=BORDER)
        style.configure("TLabelframe.Label", background=BG_SIDE, foreground=TEXT_MAIN, font=("Segoe UI", 9, "bold"))

        style.configure("TButton", background="#f6f8fa", foreground=TEXT_MAIN, borderwidth=1, relief="solid", padding=4)
        style.map(
            "TButton",
            background=[("active", "#eaeef2"), ("pressed", "#d0d7de")],
            foreground=[("active", ACCENT)],
        )
        style.configure("Primary.TButton", background=ACCENT, foreground="#ffffff", font=("Segoe UI", 9, "bold"), padding=5)
        style.map(
            "Primary.TButton",
            background=[("active", "#0550ae"), ("disabled", "#94a3b8")],
            foreground=[("disabled", "#ffffff")],
        )

        style.configure("TNotebook", background=BG_MAIN, tabmargins=[2, 2, 2, 0])
        style.configure("TNotebook.Tab", background="#e1e4e8", foreground=TEXT_SUB, padding=[10, 5], font=("Segoe UI", 9))
        style.map("TNotebook.Tab", background=[("selected", "#ffffff")], foreground=[("selected", ACCENT)])

        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor="#e1e4e8", borderwidth=0)

        main_pane = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        top_area = ttk.PanedWindow(main_pane, orient=tk.HORIZONTAL)
        main_pane.add(top_area, weight=4)

        self._create_control_panel(top_area)
        self._create_display_area(top_area)

        progress_frame = ttk.LabelFrame(main_pane, text="Дашборд Эволюции и Аналитика Алгоритма", height=220)
        main_pane.add(progress_frame, weight=1)
        self._create_stats_and_chart_area(progress_frame)

    def _create_control_panel(self, parent: ttk.PanedWindow) -> None:
        outer = ttk.Frame(parent, width=560)
        parent.add(outer, weight=0)

        canvas = tk.Canvas(outer, bg="#f6f8fa", highlightthickness=0, width=550)
        v_scroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        container = ttk.Frame(canvas)

        container_id = canvas.create_window((0, 0), window=container, anchor="nw")
        canvas.configure(yscrollcommand=v_scroll.set)

        def _on_frame_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfigure(container_id, width=max(520, event.width))

        container.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            if canvas.winfo_height() < container.winfo_reqheight():
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        left_col = ttk.Frame(container)
        left_col.pack(side=tk.LEFT, fill=tk.Y, anchor="n", padx=(4, 2), expand=True)
        right_col = ttk.Frame(container)
        right_col.pack(side=tk.LEFT, fill=tk.Y, anchor="n", padx=(2, 4), expand=True)

        self._create_image_settings_section(left_col)
        self._create_geometry_section(left_col)
        self._create_algorithm_section(left_col)

        self._create_fitness_and_radon_section(right_col)
        self._create_inspector_section(right_col)
        self._create_control_and_export_section(right_col)

    def _create_display_area(self, parent: ttk.PanedWindow) -> None:
        display_area = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        parent.add(display_area, weight=1)

        target_frame = ttk.LabelFrame(display_area, text="Целевой шаблон (с живой предобработкой)", width=340)
        display_area.add(target_frame, weight=0)
        canvas = tk.Canvas(target_frame, bg="#ffffff", relief="sunken", borderwidth=1)
        canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        canvas.bind("<Configure>", self.root.redraw_target_canvas)
        self.widgets["target_canvas"] = canvas

        results_area = ttk.Frame(display_area)
        display_area.add(results_area, weight=1)
        results_notebook = ttk.Notebook(results_area)
        results_notebook.pack(fill=tk.BOTH, expand=True)
        results_notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.widgets["results_notebook"] = results_notebook

    def _on_tab_changed(self, _event=None) -> None:
        if not self.coordinator:
            return
        notebook = self.widgets["results_notebook"]
        sel = notebook.select()
        if not sel:
            return
        idx = notebook.index(sel)
        if idx == 0:
            self.on_result_canvas_resize(None, "overall")
        elif 1 <= idx <= int(self.state.safe_get("num_islands")):
            self.on_result_canvas_resize(None, idx - 1)

    def _create_section(self, parent: tk.Widget, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=6)
        frame.pack(fill=tk.X, pady=3, padx=3, anchor="n")
        return frame

    def _create_image_settings_section(self, parent: tk.Widget) -> None:
        frame = self._create_section(parent, "1. Изображение, Пресеты и Авто")
        btn = ttk.Button(frame, text="📁 Загрузить изображение (Ctrl+O)")
        btn.pack(fill=tk.X, pady=2)
        ToolTip(btn, TOOLTIPS["load_image"])
        self.widgets["load_image_button"] = btn

        f_pre = ttk.Frame(frame)
        f_pre.pack(fill=tk.X, pady=2)
        ttk.Label(f_pre, text="Пресет:", width=9).pack(side=tk.LEFT)
        preset_cb = ttk.Combobox(
            f_pre,
            textvariable=self.state.vars["preset"],
            values=list(PRESETS.keys()),
            state="readonly",
        )
        preset_cb.pack(fill=tk.X, expand=True)
        preset_cb.bind("<<ComboboxSelected>>", self.root.on_preset_selected)
        ToolTip(preset_cb, TOOLTIPS["preset"])
        self.widgets["preset_combo"] = preset_cb

        lbl_info = ttk.Label(
            frame,
            text="Анализ: загрузите фото...",
            font=("Segoe UI", 8),
            foreground="#333333",
            wraplength=245,
        )
        lbl_info.pack(fill=tk.X, pady=2)
        self.widgets["analysis_label"] = lbl_info

        btn_auto = ttk.Button(
            frame,
            text="🔮 Авто-подбор (без каши)",
            command=self.root.apply_auto_recommendations,
        )
        btn_auto.pack(fill=tk.X, pady=2)

        self._create_double_slider_row(frame, "gamma", "Светлота (Гамма)", (0.50, 1.40), command=self.root.redraw_target_canvas)
        self._create_double_slider_row(frame, "clahe_clip", "Контраст (CLAHE)", (0.0, 3.5), command=self.root.redraw_target_canvas)
        self.widgets["image_settings_frame"] = frame

    def _create_geometry_section(self, parent: tk.Widget) -> None:
        frame = self._create_section(parent, "2. Геометрия, Цвет и Нить")
        f_shape = ttk.Frame(frame)
        f_shape.pack(fill=tk.X, pady=2)
        ttk.Label(f_shape, text="Форма холста:", width=15).pack(side=tk.LEFT)
        shape_combo = ttk.Combobox(
            f_shape,
            textvariable=self.state.vars["canvas_shape"],
            values=CANVAS_SHAPES,
            state="readonly",
        )
        shape_combo.pack(fill=tk.X, expand=True)
        shape_combo.bind("<<ComboboxSelected>>", self.root.redraw_target_canvas)
        ToolTip(shape_combo, TOOLTIPS["canvas_shape"])

        f_color = ttk.Frame(frame)
        f_color.pack(fill=tk.X, pady=2)
        ttk.Label(f_color, text="Режим цвета:", width=15).pack(side=tk.LEFT)
        color_combo = ttk.Combobox(
            f_color,
            textvariable=self.state.vars["color_mode"],
            values=COLOR_MODES,
            state="readonly",
        )
        color_combo.pack(fill=tk.X, expand=True)
        ToolTip(color_combo, TOOLTIPS["color_mode"])

        self._create_slider_row(frame, "pins", "Гвозди", (120, 480))
        self._create_double_slider_row(frame, "physical_diameter", "Размер (см)", (20.0, 120.0))
        self._create_slider_row(frame, "lines", "Макс. нитей", (400, 4500))
        self._create_slider_row(frame, "line_weight", "Чернота нити (%)", (40, 95), command=self._on_visual_param_change)
        self._create_slider_row(frame, "thread_thickness", "Толщина нити", (1, 3), command=self._on_visual_param_change)
        self._create_slider_row(frame, "fearless_factor", "Защита белого (%)", (70, 100))
        self.widgets["geometry_frame"] = frame

    def _create_algorithm_section(self, parent: tk.Widget) -> None:
        frame = self._create_section(parent, "3. Островной ГА (Полировка)")
        sliders = [
            ("num_islands", "Острова (потоки)", (1, max(2, os.cpu_count() or 4))),
            ("population_size", "Популяция", (8, 80)),
            ("stages", "Этапы доводки", (2, 12)),
            ("mutation_rate", "Мутация (%)", (5, 70)),
            ("migration_interval", "Миграция (пок.)", (5, 50)),
        ]
        for name, text, limits in sliders:
            self._create_slider_row(frame, name, text, limits)
        self.widgets["algo_frame"] = frame

    def _create_fitness_and_radon_section(self, parent: tk.Widget) -> None:
        frame = self._create_section(parent, "4. Радон FBP и Дельта-Поиск")
        f_struct = ttk.Frame(frame)
        f_struct.pack(fill=tk.X, pady=2)
        ttk.Label(f_struct, text="Метрика краев:", width=15).pack(side=tk.LEFT)
        struct_values = ["Контуры"]
        if SCIPY_AVAILABLE:
            struct_values.insert(0, "Преобразование расстояний")
        struct_combo = ttk.Combobox(
            f_struct,
            textvariable=self.state.vars["structural_metric"],
            values=struct_values,
            state="readonly",
        )
        struct_combo.pack(fill=tk.X, expand=True)
        ToolTip(struct_combo, TOOLTIPS["structural_metric"])
        self._create_slider_row(frame, "contour_weight", "Вес контуров (%)", (10, 90))

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=4)
        check_mem = ttk.Checkbutton(
            frame,
            text="Включить быстрый дельта-поиск",
            variable=self.state.vars["memetic_enabled"],
        )
        check_mem.pack(anchor="w")
        ToolTip(check_mem, TOOLTIPS["memetic_enabled"])
        self._create_slider_row(frame, "memetic_intensity", "Интенсивность (%)", (5, 50))
        self._create_slider_row(frame, "memetic_depth", "Глубина поиска", (5, 80))

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=4)
        check_radon = ttk.Checkbutton(
            frame,
            text="Наведение Радона (Ramp FBP)",
            variable=self.state.vars["radon_enabled"],
        )
        check_radon.pack(anchor="w")
        ToolTip(check_radon, TOOLTIPS["radon_enabled"])
        self._create_slider_row(frame, "radon_influence", "Сила Радона (%)", (0, 100))
        self.widgets["fitness_frame"] = frame

    def _create_inspector_section(self, parent: tk.Widget) -> None:
        frame = self._create_section(parent, "5. Интерактивный Плеер и Инспектор")

        f_lay = ttk.Frame(frame)
        f_lay.pack(fill=tk.X, pady=2)
        ttk.Label(f_lay, text="Активный слой:", width=15).pack(side=tk.LEFT)
        layer_cb = ttk.Combobox(
            f_lay,
            textvariable=self.state.vars["active_layer"],
            values=["Черная нить"],
            state="readonly",
        )
        layer_cb.pack(fill=tk.X, expand=True)
        layer_cb.bind("<<ComboboxSelected>>", self._on_active_layer_changed)
        ToolTip(layer_cb, TOOLTIPS["active_layer"])
        self.widgets["active_layer_combo"] = layer_cb

        self._create_slider_row(frame, "step_scrubber", "Слой плетения", (1, 4500), command=self._on_scrubber_slider_move)

        f_step = ttk.Frame(frame)
        f_step.pack(fill=tk.X, pady=2)
        self.widgets["player_step_label"] = ttk.Label(
            f_step, text="Ход: --", font=("Segoe UI", 8, "bold"), foreground="#0969da"
        )
        self.widgets["player_step_label"].pack(side=tk.LEFT)

        nav = ttk.Frame(frame)
        nav.pack(fill=tk.X, pady=2)
        btn_prev = ttk.Button(nav, text="◀ Назад", command=lambda: self.root.step_player(-1))
        btn_auto = ttk.Button(nav, text="▶ Авто", command=self.root.toggle_auto_play)
        btn_next = ttk.Button(nav, text="Вперед ▶", command=lambda: self.root.step_player(1))
        btn_prev.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        btn_auto.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        btn_next.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        self.widgets["auto_play_button"] = btn_auto

        opts = ttk.Frame(frame)
        opts.pack(fill=tk.X, pady=2)
        chk_neon = ttk.Checkbutton(
            opts,
            text="✨ Подсветка нити",
            variable=self.state.vars["show_active_chord"],
            command=self._on_visual_param_change,
        )
        chk_neon.pack(side=tk.LEFT)
        ToolTip(chk_neon, TOOLTIPS["show_active_chord"])

        chk_voice = ttk.Checkbutton(
            opts,
            text="🔊 Голос (SAPI)",
            variable=self.state.vars["voice_enabled"],
            command=self.root.on_voice_toggle,
        )
        if not SAPI_AVAILABLE:
            chk_voice.config(state=tk.DISABLED)
        chk_voice.pack(side=tk.RIGHT)
        ToolTip(chk_voice, TOOLTIPS["voice_enabled"])

        self._create_slider_row(frame, "split_compare", "Шторка До/После %", (0, 100), command=self._on_visual_param_change)
        self.widgets["inspector_frame"] = frame

    def _create_control_and_export_section(self, parent: tk.Widget) -> None:
        frame = self._create_section(parent, "6. Управление и Экспорт")
        btn_start = ttk.Button(
            frame,
            text="▶ Старт (CSR OMP + Острова)",
            style="Primary.TButton",
            command=self.coordinator.start_generation,
        )
        btn_start.pack(fill=tk.X, pady=2)
        ToolTip(btn_start, TOOLTIPS["start"])
        self.widgets["start_button"] = btn_start

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=2)
        for name, text, command, state in [
            ("polish", "✨ Доводка", self.coordinator.start_evolution_only, tk.NORMAL),
            ("pause", "❚❚ Пауза", self.coordinator.toggle_pause, tk.DISABLED),
            ("stop", "■ Стоп", self.coordinator.stop_generation, tk.DISABLED),
        ]:
            btn = ttk.Button(btn_row, text=text, command=command, state=state)
            btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
            ToolTip(btn, TOOLTIPS.get(name, ""))
            self.widgets[f"{name}_button"] = btn

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=4)

        btn_prune = ttk.Button(
            frame,
            text="✂ Оптимизировать путь (убрать петли)",
            command=self.coordinator.prune_chromosome,
            state=tk.DISABLED,
        )
        btn_prune.pack(fill=tk.X, pady=1)
        ToolTip(btn_prune, TOOLTIPS["prune"])
        self.widgets["prune_button"] = btn_prune

        btn_html = ttk.Button(
            frame,
            text="🌐 Автономный веб-ассистент (.html)",
            command=self.coordinator.export_html,
            state=tk.DISABLED,
        )
        btn_html.pack(fill=tk.X, pady=2)
        ToolTip(btn_html, TOOLTIPS["export_html"])
        self.widgets["export_html_button"] = btn_html

        grid = ttk.Frame(frame)
        grid.pack(fill=tk.X, pady=1)
        export_pairs = [
            ("export_txt", "📄 Схема (.txt)", self.coordinator.export_txt, 0, 0),
            ("export_csv", "📊 Таблица (.csv)", self.coordinator.export_csv, 0, 1),
            ("export_png", "🖼 Ultra HD (.png)", self.coordinator.export_png, 1, 0),
            ("export_svg", "📐 Вектор (.svg)", self.coordinator.export_svg, 1, 1),
            ("export_template", "⭕ Шаблон 1:1 (.svg)", self.coordinator.export_template, 2, 0),
            ("export_gcode", "⚙ ЧПУ (.gcode)", self.coordinator.export_gcode, 2, 1),
        ]
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        for name, text, cmd, r, c in export_pairs:
            btn = ttk.Button(grid, text=text, command=cmd, state=tk.DISABLED)
            btn.grid(row=r, column=c, sticky="ew", padx=1, pady=1)
            ToolTip(btn, TOOLTIPS.get(name, ""))
            self.widgets[f"{name}_button"] = btn

    def _create_stats_and_chart_area(self, parent: tk.Widget) -> None:
        bottom_panel = ttk.Frame(parent)
        bottom_panel.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left_panel = ttk.Frame(bottom_panel, width=295)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)

        self.widgets["status_label"] = ttk.Label(
            left_panel, text="Ожидание запуска...", font=("Segoe UI", 9, "bold"), wraplength=280
        )
        self.widgets["status_label"].pack(anchor="w")
        self.widgets["fitness_label"] = ttk.Label(left_panel, text="Точность (Фитнес): --", foreground="#0969da")
        self.widgets["fitness_label"].pack(anchor="w", pady=(0, 2))

        mode_str = f"PyTorch CUDA ({CUDA_DEVICE_NAME}) + CSR" if GPU_ENABLED_FLAG else "SciPy CSR + NumPy (CPU)"
        self.widgets["compute_mode_label"] = ttk.Label(
            left_panel, text=f"Ядро: {mode_str}", font=("Segoe UI", 8), foreground="#57606a"
        )
        self.widgets["compute_mode_label"].pack(anchor="w", pady=(0, 4))

        for key, text in [
            ("time", "Время:"),
            ("eta", "Осталось (ETA):"),
            ("gens", "Поколений:"),
            ("mig", "Миграций:"),
            ("len", "Длина нити:"),
        ]:
            self.widgets[key + "_label"] = self._create_stat_row(left_panel, text)

        self.widgets["main_progress_bars_frame"] = ttk.Frame(left_panel)
        self.widgets["main_progress_bars_frame"].pack(fill=tk.BOTH, expand=True, pady=3)

        fig = Figure(figsize=(7, 2.2), facecolor="#ffffff")
        self.ax_fitness = fig.add_subplot(121)
        self.ax_mut = fig.add_subplot(122)
        fig.tight_layout(pad=1.2)
        self.chart_canvas = FigureCanvasTkAgg(fig, master=bottom_panel)
        self.chart_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=4, pady=2)

    def _create_slider_row(
        self,
        parent: tk.Widget,
        name: str,
        text: str,
        limits: Tuple[int, int],
        command: Optional[Callable] = None,
    ) -> None:
        self._slider_limits[name] = (float(limits[0]), float(limits[1]), True)
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=1)
        lbl = ttk.Label(f, text=f"{text}:", width=15)
        lbl.pack(side=tk.LEFT)
        entry = ttk.Entry(f, textvariable=self.state.vars[name], width=6)
        entry.pack(side=tk.RIGHT)
        var = self.state.vars[name]

        def cmd(v):
            var.set(int(round(float(v))))
            if command:
                command()

        def on_entry_commit(_event=None):
            try:
                val = int(round(float(entry.get().replace(",", "."))))
            except ValueError:
                val = int(self.state.safe_get(name))
            val = max(int(limits[0]), min(int(limits[1]), val))
            var.set(val)
            if command:
                command()

        entry.bind("<FocusOut>", on_entry_commit)
        entry.bind("<Return>", on_entry_commit)

        scale = ttk.Scale(f, from_=limits[0], to=limits[1], variable=var, orient=tk.HORIZONTAL, command=cmd)
        scale.pack(fill=tk.X, expand=True, padx=4)
        self.widgets[f"{name}_scale"] = scale
        tip = TOOLTIPS.get(name, "")
        ToolTip(lbl, tip)
        ToolTip(entry, tip)
        ToolTip(scale, tip)

    def _create_double_slider_row(
        self,
        parent: tk.Widget,
        name: str,
        text: str,
        limits: Tuple[float, float],
        command: Optional[Callable] = None,
    ) -> None:
        self._slider_limits[name] = (float(limits[0]), float(limits[1]), False)
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=1)
        lbl = ttk.Label(f, text=f"{text}:", width=15)
        lbl.pack(side=tk.LEFT)
        entry = ttk.Entry(f, textvariable=self.state.vars[name], width=6)
        entry.pack(side=tk.RIGHT)
        var = self.state.vars[name]

        def cmd(v):
            var.set(round(float(v), 2))
            if command:
                command()

        def on_entry_commit(_event=None):
            try:
                val = float(entry.get().replace(",", "."))
            except ValueError:
                val = float(self.state.safe_get(name))
            val = round(max(float(limits[0]), min(float(limits[1]), val)), 2)
            var.set(val)
            if command:
                command()

        entry.bind("<FocusOut>", on_entry_commit)
        entry.bind("<Return>", on_entry_commit)

        scale = ttk.Scale(f, from_=limits[0], to=limits[1], variable=var, orient=tk.HORIZONTAL, command=cmd)
        scale.pack(fill=tk.X, expand=True, padx=4)
        self.widgets[f"{name}_scale"] = scale
        tip = TOOLTIPS.get(name, "")
        ToolTip(lbl, tip)
        ToolTip(entry, tip)
        ToolTip(scale, tip)

    def _create_stat_row(self, parent: tk.Widget, text: str) -> ttk.Label:
        row_frame = ttk.Frame(parent)
        row_frame.pack(fill="x")
        ttk.Label(row_frame, text=text, font=("Segoe UI", 8, "bold"), width=15, anchor="w").pack(side=tk.LEFT)
        value_label = ttk.Label(row_frame, text="--", font=("Segoe UI", 8), anchor="w")
        value_label.pack(side=tk.LEFT)
        return value_label

    def update_layer_selector(self) -> None:
        cb = self.widgets.get("active_layer_combo")
        if not isinstance(cb, ttk.Combobox):
            return
        if not self.state.layers:
            cb["values"] = ["Черная нить"]
            self.state.vars["active_layer"].set("Черная нить")
            return
        names = [l.get("name", f"Слой {i+1}") for i, l in enumerate(self.state.layers)]
        cb["values"] = names
        idx = min(self.state.active_layer_idx, len(names) - 1)
        self.state.active_layer_idx = max(0, idx)
        self.state.vars["active_layer"].set(names[self.state.active_layer_idx])

    def _on_active_layer_changed(self, _event=None) -> None:
        if not self.state.layers:
            return
        sel_name = self.state.vars["active_layer"].get()
        for idx, layer in enumerate(self.state.layers):
            if layer.get("name") == sel_name:
                self.state.active_layer_idx = idx
                path = layer.get("path", [])
                max_steps = max(1, len(path) - 1)
                self.state.overall_best_chromosome = path
                scale = self.widgets.get("step_scrubber_scale")
                if isinstance(scale, ttk.Scale):
                    scale.configure(to=max_steps)
                self.state.vars["step_scrubber"].set(max_steps)
                self.update_player_step_label()
                self._on_visual_param_change()
                break

    def _on_scrubber_slider_move(self, *_args) -> None:
        self.update_player_step_label()
        self._on_visual_param_change()

    def update_player_step_label(self) -> None:
        lbl = self.widgets.get("player_step_label")
        if not lbl:
            return
        if not self.state.layers or self.state.active_layer_idx >= len(self.state.layers):
            lbl.config(text="Ход: --")
            return
        path = self.state.layers[self.state.active_layer_idx].get("path", [])
        if len(path) < 2:
            lbl.config(text="Ход: --")
            return
        total = len(path) - 1
        step = max(1, min(total, int(self.state.safe_get("step_scrubber"))))
        lbl.config(text=f"Шаг {step}/{total}: гвоздь {path[step - 1]} → {path[step]}")

    def _on_visual_param_change(self, *_args) -> None:
        if self.coordinator and self.state.layers:
            config = self.state.get_config()
            self.coordinator.request_canvas_update(
                "overall",
                (list(self.state.layers), config["pins"], config["thread_thickness"]),
                {"force": True},
            )

    def on_result_canvas_resize(self, _event, key: Any) -> None:
        canvas = self.result_canvases.get(key)
        if not canvas:
            return
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 20 or h < 20:
            return
        if self.coordinator and key in self.coordinator.best_chromosomes_by_key:
            data_obj = self.coordinator.best_chromosomes_by_key[key]
            config = self.state.get_config()
            self.coordinator.request_canvas_update(key, (data_obj, config["pins"], config["thread_thickness"]), {})
        elif key in self.raw_pil_images:
            self.update_canvas_image(key, self.raw_pil_images[key])

    def setup_for_new_run(self, num_islands: int) -> None:
        self.setup_notebook_tabs(num_islands)
        self.setup_progress_bars(num_islands)
        self.chart_needs_update = True
        self.update_chart()

    def setup_notebook_tabs(self, num_islands: int) -> None:
        notebook = self.widgets["results_notebook"]
        for tab_id in notebook.tabs():
            notebook.forget(tab_id)
        self.result_canvases.clear()
        self.result_photo_refs.clear()

        self._create_notebook_tab("overall", "★ Лучший результат", notebook)
        for i in range(num_islands):
            self._create_notebook_tab(i, f"Остров {i + 1}", notebook)

        self._create_analysis_canvas_tab(notebook, "diff_map", "Карта Ошибок (Кр/Син)")
        self._create_analysis_canvas_tab(notebook, "density_heatmap", "Плотность нитей (Heatmap)")
        self._create_analysis_canvas_tab(notebook, "fit_saliency", "Карта Значимости (DoG)")
        self._create_analysis_canvas_tab(notebook, "radon_sinogram", "Радон + FBP")

    def _create_analysis_canvas_tab(self, parent_notebook: ttk.Notebook, key: str, title: str) -> None:
        frame = ttk.Frame(parent_notebook)
        parent_notebook.add(frame, text=title)
        canvas = tk.Canvas(frame, bg="#18181b", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        canvas.bind("<Configure>", lambda e, k=key: self.on_result_canvas_resize(e, k))
        self.result_canvases[key] = canvas

    def _create_notebook_tab(self, key: Any, text: str, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=text)
        canvas = tk.Canvas(tab, bg="#ffffff", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.bind("<Configure>", lambda e, k=key: self.on_result_canvas_resize(e, k))
        self.result_canvases[key] = canvas

    def setup_progress_bars(self, num_islands: int) -> None:
        pb_frame = self.widgets["main_progress_bars_frame"]
        for widget in pb_frame.winfo_children():
            widget.destroy()
        self.main_island_progress_bars.clear()
        self.main_island_labels.clear()
        for i in range(num_islands):
            frame = ttk.Frame(pb_frame)
            frame.pack(fill=tk.X, expand=False, pady=1)
            label = ttk.Label(frame, text=f"Остров {i + 1}: старт", width=19, font=("Segoe UI", 8))
            label.pack(side=tk.LEFT, padx=(0, 4))
            self.main_island_labels[i] = label
            pb = ttk.Progressbar(frame, orient="horizontal", mode="determinate")
            pb.pack(fill=tk.X, expand=True)
            self.main_island_progress_bars[i] = pb

    def update_chart(self) -> None:
        if not self.ax_fitness or not self.ax_mut or not self.chart_canvas:
            return
        self.ax_fitness.clear()
        self.ax_mut.clear()
        self.ax_fitness.set_facecolor("#ffffff")
        self.ax_mut.set_facecolor("#ffffff")

        num_islands = max(1, int(self.state.safe_get("num_islands")))
        colors = plt.cm.tab10(np.linspace(0, 0.8, num_islands))
        has_lines = False
        for i in range(num_islands):
            data = self.state.chart_data_by_island.get(i, [])
            if data:
                gens, fitnesses = zip(*data)
                self.ax_fitness.plot(gens, fitnesses, label=f"Остров {i + 1}", color=colors[i], linewidth=1.5)
                has_lines = True
        if self.state.omp_baseline_fitness > 0:
            self.ax_fitness.axhline(self.state.omp_baseline_fitness, color="gray", linestyle=":", label="OMP База")
            has_lines = True

        self.ax_fitness.set_title("Рост точности (Фитнес)", fontsize=9)
        self.ax_fitness.set_xlabel("Поколения", fontsize=8)
        self.ax_fitness.tick_params(labelsize=8)
        if has_lines:
            self.ax_fitness.legend(loc="lower right", fontsize=7)
        self.ax_fitness.grid(True, linestyle="--", alpha=0.6, color="#e1e4e8")

        if self.state.aggregated_mut_stats:
            names = list(self.state.aggregated_mut_stats.keys())
            rates = [
                100.0
                * self.state.aggregated_mut_stats[n]["success"]
                / max(1, self.state.aggregated_mut_stats[n]["attempts"])
                for n in names
            ]
            bars = self.ax_mut.barh(names, rates, color="#0969da", height=0.55)
            self.ax_mut.set_xlim(0, max(35.0, max(rates) * 1.15))
            for bar, r in zip(bars, rates):
                self.ax_mut.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2, f"{r:.1f}%", va="center", fontsize=7)
        self.ax_mut.set_title("КПД операторов мутации (%)", fontsize=9)
        self.ax_mut.tick_params(labelsize=8)
        self.ax_mut.grid(True, axis="x", linestyle="--", alpha=0.6, color="#e1e4e8")

        self.chart_canvas.figure.tight_layout(pad=1.0)
        self.chart_canvas.draw()
        self.chart_needs_update = False

    def _set_widget_tree_state(self, root_widget: Optional[tk.Widget], state: str) -> None:
        if root_widget is None:
            return
        for child in root_widget.winfo_children():
            if isinstance(child, ttk.Combobox):
                child.configure(state="disabled" if state == tk.DISABLED else "readonly")
            elif isinstance(child, (ttk.Scale, ttk.Entry, ttk.Button, ttk.Checkbutton)):
                try:
                    child.configure(state=state)
                except tk.TclError:
                    pass
            elif isinstance(child, (ttk.Frame, ttk.LabelFrame)):
                self._set_widget_tree_state(child, state)

    def toggle_ui_state(self, is_running: bool) -> None:
        self.state.is_running = is_running
        state = tk.DISABLED if is_running else tk.NORMAL
        for frame_key in ["image_settings_frame", "algo_frame", "fitness_frame", "geometry_frame"]:
            self._set_widget_tree_state(self.widgets.get(frame_key), state)

        self.widgets["load_image_button"].config(state=state)
        self.widgets["start_button"].config(state=state)
        if "polish_button" in self.widgets:
            self.widgets["polish_button"].config(state=state)

        control_state = tk.NORMAL if is_running else tk.DISABLED
        self.widgets["stop_button"].config(state=control_state)
        self.widgets["pause_button"].config(state=control_state)

        has_path = any(len(l.get("path", [])) >= 2 for l in self.state.layers) or bool(
            self.state.overall_best_chromosome and len(self.state.overall_best_chromosome) >= 2
        )
        export_state = tk.NORMAL if (not is_running and has_path) else tk.DISABLED
        for key in [
            "prune_button",
            "export_txt_button",
            "export_csv_button",
            "export_gcode_button",
            "export_png_button",
            "export_svg_button",
            "export_template_button",
            "export_html_button",
        ]:
            if key in self.widgets:
                self.widgets[key].config(state=export_state)

        if not is_running:
            self.widgets["pause_button"].config(text="❚❚ Пауза")
            for i, pb in self.main_island_progress_bars.items():
                pb["value"] = 0
                if i in self.main_island_labels:
                    self.main_island_labels[i].config(text=f"Остров {i + 1}: готово")

    def update_island_progress(self, island_id: int, stage: int, patience: int, limit: int) -> None:
        if island_id in self.main_island_labels:
            pct = int((patience / max(1, limit)) * 100)
            total_st = max(1, int(self.state.safe_get("stages")))
            self.main_island_labels[island_id].config(text=f"Ост.{island_id + 1} Эт.{stage + 1}/{total_st}: {pct}%")
        if island_id in self.main_island_progress_bars:
            pb = self.main_island_progress_bars[island_id]
            pb["maximum"] = max(1, limit)
            pb["value"] = patience

    def update_best_fitness_display(self, fitness: float) -> None:
        n_lines = sum(max(0, len(l.get("path", [])) - 1) for l in self.state.layers)
        if n_lines == 0:
            n_lines = max(0, len(self.state.overall_best_chromosome or []) - 1)
        self.widgets["fitness_label"].config(text=f"Точность: {fitness:.5f} ({n_lines} нитей)")

    def update_thread_length_display(self, length_in_meters: float) -> None:
        dia = float(self.state.safe_get("physical_diameter") or 50.0)
        self.widgets["len_label"].config(text=f"{length_in_meters:.1f} м (на Ø{dia:.0f}см)")

    def update_stats_display(self, current_state: AppState) -> None:
        elapsed = time.time() - current_state.start_time
        self.widgets["time_label"].config(text=time.strftime("%H:%M:%S", time.gmtime(elapsed)))
        self.widgets["gens_label"].config(text=f"{sum(current_state.generation_counters.values())}")
        self.widgets["mig_label"].config(text=f"{current_state.migration_count}")

        num_islands = max(1, int(current_state.safe_get("num_islands")))
        total_stages = max(1, int(current_state.safe_get("stages")))
        total_progress = 0.0
        for i in range(num_islands):
            stage, limit = current_state.island_progress_data.get(i, (0, 1))
            pb = self.main_island_progress_bars.get(i)
            patience = float(pb["value"]) if pb else 0.0
            prog_per_stage = 1.0 / total_stages
            total_progress += (stage * prog_per_stage) + (patience / max(1.0, float(limit))) * prog_per_stage

        avg_progress = total_progress / num_islands
        if avg_progress > 0.02:
            eta = (elapsed / avg_progress) * max(0.0, 1.0 - avg_progress)
            self.widgets["eta_label"].config(text=time.strftime("%H:%M:%S", time.gmtime(eta)))
        else:
            self.widgets["eta_label"].config(text="Расчет...")

    def update_canvas_image(self, key: Any, img_pil: Image.Image) -> None:
        self.raw_pil_images[key] = img_pil
        canvas = self.result_canvases.get(key)
        if not canvas:
            return
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 10 or h < 10:
            return

        if key == "radon_sinogram":
            # Сохраняем соотношение сторон широкой панели Синограмма + FBP
            scale = min((w - 8) / max(1, img_pil.width), (h - 8) / max(1, img_pil.height))
            nw = max(40, int(round(img_pil.width * scale)))
            nh = max(40, int(round(img_pil.height * scale)))
            img_shown = img_pil.resize((nw, nh), Image.Resampling.LANCZOS)
        else:
            display_size = max(50, min(w, h) - 8)
            if img_pil.width != display_size or img_pil.height != display_size:
                img_shown = img_pil.resize((display_size, display_size), Image.Resampling.LANCZOS)
            else:
                img_shown = img_pil

            if key == "overall" and self.state.preprocessed_target_np is not None:
                split_pct = int(self.state.safe_get("split_compare") or 0)
                if split_pct > 0:
                    split_x = int(display_size * (split_pct / 100.0))
                    is_color = "CMYK" in str(self.state.safe_get("color_mode"))
                    if is_color and self.state.preprocessed_rgb_np is not None:
                        target_src = self.state.preprocessed_rgb_np
                        comp_np = np.array(img_shown.convert("RGB"))
                        target_resized = cv2.resize(target_src, (display_size, display_size), interpolation=cv2.INTER_AREA)
                        comp_np[:, :split_x] = target_resized[:, :split_x]
                        cv2.line(comp_np, (split_x, 0), (split_x, display_size), (9, 105, 218), 2)
                    else:
                        target_resized = cv2.resize(
                            self.state.preprocessed_target_np, (display_size, display_size), interpolation=cv2.INTER_AREA
                        )
                        comp_np = np.array(img_shown.convert("RGB"))
                        target_rgb = cv2.cvtColor(target_resized, cv2.COLOR_GRAY2RGB)
                        comp_np[:, :split_x] = target_rgb[:, :split_x]
                        cv2.line(comp_np, (split_x, 0), (split_x, display_size), (9, 105, 218), 2)
                    img_shown = Image.fromarray(comp_np)

        photo_ref = ImageTk.PhotoImage(img_shown)
        self.result_photo_refs[key] = photo_ref
        canvas.delete("all")
        canvas.create_image(w // 2, h // 2, anchor=tk.CENTER, image=photo_ref)

    def update_title(self) -> None:
        base_title = f"{APP_TITLE} (CSR OMP + Island GA + Radon FBP + CUDA)"
        if self.state.original_image_path:
            filename = os.path.basename(self.state.original_image_path)
            self.root.title(f"{base_title} — [{filename}]")
        else:
            self.root.title(base_title)


class EvolutionCoordinator:
    """Координирует подготовку данных, многопоточный синтез, рендеринг и экспорт."""

    def __init__(
        self,
        root: "StringArtApp",
        state: AppState,
        ui_manager: UIManager,
        queues: Dict[str, queue.Queue],
        audio_assistant: AudioAssistant,
    ):
        self.root = root
        self.state = state
        self.ui_manager = ui_manager
        self.queues = queues
        self.audio_assistant = audio_assistant
        self.manager_thread: Optional[IslandManagerThread] = None
        self.omp_stop_flag: bool = False
        self.omp_pause_event = threading.Event()
        self.omp_pause_event.set()
        self.active_fitness_calc: Optional[FitnessCalculator] = None

        # Потокобезопасный буфер рендеринга с интеллектуальным сбросом устаревших кадров
        self._render_cond = threading.Condition()
        self._pending_renders: Dict[Any, Tuple[Any, Dict[str, Any]]] = {}
        self._renderer_stop = False
        self.render_results_queue: queue.Queue = queue.Queue()
        self.best_chromosomes_by_key: Dict[Any, Any] = {}
        self.stats_update_job: Optional[str] = None
        self.canvas_manager: Optional[CanvasManager] = None
        self.radon_guide: Optional[RadonGuide] = None
        self._last_analytic_map_time: float = 0.0
        self._start_renderer_thread()

    def _start_renderer_thread(self) -> None:
        threading.Thread(target=self._renderer_worker, name="RendererThread", daemon=True).start()

    def shutdown(self) -> None:
        self._renderer_stop = True
        with self._render_cond:
            self._render_cond.notify_all()

    def _renderer_worker(self) -> None:
        while not self._renderer_stop:
            task = None
            key = None
            with self._render_cond:
                while not self._pending_renders and not self._renderer_stop:
                    self._render_cond.wait(timeout=0.5)
                if self._renderer_stop:
                    break
                # Приоритет отдаем главному виду 'overall'
                key = "overall" if "overall" in self._pending_renders else next(iter(self._pending_renders))
                task = self._pending_renders.pop(key)

            if task is None or self.canvas_manager is None:
                continue

            try:
                data, render_params = task
                layers_or_chromo, _, thread_thickness = data
                out_size = render_params.get("output_size", 550)
                line_weight = render_params.get("line_weight", 72)
                max_step = render_params.get("max_step", None)
                active_layer_idx = render_params.get("active_layer_idx", 0)
                show_active = render_params.get("show_active_chord", False)

                if isinstance(layers_or_chromo, list) and layers_or_chromo and isinstance(layers_or_chromo[0], dict):
                    layers_data = layers_or_chromo
                else:
                    layers_data = [{"name": "Черная нить", "color_hex": "#111111", "path": list(layers_or_chromo)}]

                active_chord = None
                if show_active and active_layer_idx < len(layers_data):
                    p = layers_data[active_layer_idx].get("path", [])
                    total_l = len(p) - 1
                    step_idx = total_l if max_step is None else min(total_l, max(1, int(max_step)))
                    if 0 < step_idx < len(p) and max_step is not None and max_step < total_l:
                        active_chord = (p[step_idx - 1], p[step_idx])

                img_pil = self.canvas_manager.render_layers_to_pil(
                    layers_data,
                    thickness=thread_thickness,
                    line_weight_percent=line_weight,
                    max_step=max_step,
                    output_size=out_size,
                    active_layer_idx=active_layer_idx,
                    active_chord=active_chord,
                )
                self.render_results_queue.put((key, img_pil))

                # Асинхронно обновляем карты ошибок/плотности без торможения UI-потока
                now = time.perf_counter()
                if (
                    key == "overall"
                    and self.active_fitness_calc is not None
                    and len(layers_data) == 1
                    and (now - self._last_analytic_map_time > 0.45 or render_params.get("force_maps"))
                ):
                    self._last_analytic_map_time = now
                    path_main = layers_data[0].get("path", [])
                    if len(path_main) >= 2:
                        _, diff_map, depth_2d = self.active_fitness_calc.calculate_single_fitness_and_maps(path_main)
                        self._push_analytic_maps(diff_map, depth_2d)
            except Exception:
                logging.error("Ошибка в потоке рендеринга", exc_info=True)

    def _cancel_stats_job(self) -> None:
        if self.stats_update_job:
            try:
                self.root.after_cancel(self.stats_update_job)
            except Exception:
                pass
            self.stats_update_job = None

    def start_evolution_only(self) -> None:
        """Запуск параллельной островной эволюции на текущем пути без перезапуска OMP."""
        if self.state.original_image is None:
            messagebox.showwarning("Внимание", "Пожалуйста, сначала загрузите изображение.")
            return
        if not self.state.overall_best_chromosome or len(self.state.overall_best_chromosome) < 4:
            messagebox.showinfo("Информация", "Базовый путь еще не создан. Запускаем полный цикл OMP + Острова...")
            self.start_generation()
            return

        config = self.state.get_config()
        self._cancel_stats_job()
        self.omp_stop_flag = False
        self.omp_pause_event.set()
        self.state.reset_for_new_run(reset_layers=False)
        self.ui_manager.toggle_ui_state(True)
        self.best_chromosomes_by_key.clear()
        self.ui_manager.setup_for_new_run(config["num_islands"])
        self.stats_update_job = self.root.after(1000, self._update_progress_stats)
        threading.Thread(
            target=self._run_islands_only_worker, args=(config,), name="IslandsOnlyWorker", daemon=True
        ).start()

    def _run_islands_only_worker(self, config: Dict[str, Any]) -> None:
        try:
            self._send_status(f"Подготовка фитнес-калькулятора для {config['num_islands']} островов...")
            bundle = prepare_target_image_bundle(
                self.state.original_image,
                IMAGE_SIZE,
                clahe_clip=config["clahe_clip"],
                gamma=config["gamma"],
                shape=config["canvas_shape"],
                color_mode=config["color_mode"],
            )
            self.state.preprocessed_target_np = bundle["gray"]
            self.state.preprocessed_rgb_np = bundle["rgb"]

            cm = self._ensure_canvas_manager(config)
            active_idx = min(self.state.active_layer_idx, len(bundle["channels"]) - 1)
            ch_data = bundle["channels"][max(0, active_idx)]

            fc = FitnessCalculator(
                bundle["gray"],
                bundle["edges"],
                bundle["saliency"],
                ch_data["importance"],
                cm,
                config,
                self.queues,
                target_darkness_override=ch_data["darkness"],
            )
            self.active_fitness_calc = fc
            base_chromosome = list(self.state.overall_best_chromosome)
            self.state.omp_baseline_fitness = fc.calculate_single_fitness(base_chromosome)
            self._send_status(f"Запуск {config['num_islands']} параллельных островов эволюции...")
            self.manager_thread = IslandManagerThread(
                config,
                self.queues,
                fc,
                base_chromosome,
                self.radon_guide,
                layer_idx=active_idx,
                layer_name=ch_data["name"],
                emit_done=True,
            )
            self.manager_thread.start()
        except Exception:
            logging.error("Ошибка при запуске эволюционной доводки", exc_info=True)
            self.queues["status_queue"].put({"type": "error", "data": f"Ошибка при доводке: {traceback.format_exc()}"})

    def _ensure_canvas_manager(self, config: Dict[str, Any]) -> CanvasManager:
        cm = self.canvas_manager
        if (
            cm is None
            or cm.num_pins != config["pins"]
            or cm.shape != config["canvas_shape"]
            or abs(cm.physical_cm - config["physical_diameter"]) > 1e-3
        ):
            cm = CanvasManager(
                IMAGE_SIZE,
                config["pins"],
                shape=config["canvas_shape"],
                physical_cm=config["physical_diameter"],
            )
            self.canvas_manager = cm
        cm.build_line_tables()
        return cm

    def start_generation(self) -> None:
        if self.state.original_image is None:
            messagebox.showwarning("Внимание", "Пожалуйста, сначала загрузите изображение.")
            return
        config = self.state.get_config()
        self._cancel_stats_job()
        self.omp_stop_flag = False
        self.omp_pause_event.set()
        self.state.reset_for_new_run(reset_layers=True)
        self.ui_manager.toggle_ui_state(True)
        self.best_chromosomes_by_key.clear()
        self.ui_manager.setup_for_new_run(config["num_islands"])
        self.stats_update_job = self.root.after(1000, self._update_progress_stats)
        threading.Thread(
            target=self._prepare_and_run_evolution, args=(config,), name="PreparationThread", daemon=True
        ).start()

    def _prepare_and_run_evolution(self, config: Dict[str, Any]) -> None:
        try:
            self._send_status("Анализ изображения, геометрии и построение таблиц хорд...")
            bundle = prepare_target_image_bundle(
                self.state.original_image,
                IMAGE_SIZE,
                clahe_clip=config["clahe_clip"],
                gamma=config["gamma"],
                shape=config["canvas_shape"],
                color_mode=config["color_mode"],
            )
            self.state.preprocessed_target_np = bundle["gray"]
            self.state.preprocessed_rgb_np = bundle["rgb"]

            cm = self._ensure_canvas_manager(config)

            radon_guide = None
            self.radon_guide = None
            if config.get("radon_enabled", True):
                self._send_status("Вычисление томографической синограммы Радона (Ramp FBP)...")
                radon_guide = RadonGuide(bundle["gray"], bundle["edges"], cm, config)
                self.radon_guide = radon_guide
                if radon_guide.sinogram is not None:
                    self.queues["status_queue"].put({
                        "type": "radon_map",
                        "data": (radon_guide.sinogram, radon_guide.fbp_recon),
                    })

            self.queues["status_queue"].put({"type": "fit_func_maps", "data": {"saliency": bundle["saliency"]}})

            channels = bundle["channels"]
            is_cmyk = len(channels) > 1
            accumulated_layers: List[Dict[str, Any]] = []

            if not is_cmyk:
                ch = channels[0]
                fc = FitnessCalculator(
                    bundle["gray"],
                    bundle["edges"],
                    bundle["saliency"],
                    ch["importance"],
                    cm,
                    config,
                    self.queues,
                    target_darkness_override=ch["darkness"],
                )
                self.active_fitness_calc = fc
                self._send_status("Этап 1: Высокоточный CSR OMP-синтез черной нитью...")
                greedy = GreedyInitializer(
                    cm, fc.target_darkness_flat, fc.importance_flat, config, radon_guide, stop_Check=self._check_omp_stop_or_pause
                )

                def greedy_cb(status: str, path: List[int]) -> None:
                    layers_snap = [{"name": ch["name"], "color_hex": ch["color_hex"], "path": path}]
                    self.queues["status_queue"].put({"type": "greedy_update", "data": {"text": status, "layers": layers_snap, "active_idx": 0}})

                base_chromosome = greedy.run(config["lines"], greedy_cb, layer_label=ch["name"])
                if not base_chromosome or len(base_chromosome) < 2:
                    self.queues["status_queue"].put({"type": "error", "data": "Не удалось построить начальный путь."})
                    return

                self.state.layers = [{"name": ch["name"], "color_hex": ch["color_hex"], "path": list(base_chromosome)}]
                omp_fit, diff_map, depth_2d = fc.calculate_single_fitness_and_maps(base_chromosome)
                self.state.omp_baseline_fitness = omp_fit
                self.state.layer_best_fitnesses[0] = omp_fit
                self.queues["status_queue"].put({
                    "type": "island_update",
                    "data": {"id": 0, "chromosome": base_chromosome, "fitness": omp_fit, "mut_stats": {}, "layer_idx": 0},
                })
                self._push_analytic_maps(diff_map, depth_2d)

                if self.omp_stop_flag:
                    self.queues["status_queue"].put({"type": "done", "data": "Остановлено на этапе OMP."})
                    return

                self.manager_thread = IslandManagerThread(
                    config, self.queues, fc, base_chromosome, radon_guide, layer_idx=0, layer_name=ch["name"], emit_done=True
                )
                self.manager_thread.start()
            else:
                # 4-слойный цветной CMYK синтез + островная полировка по слоям
                for c_idx, ch in enumerate(channels):
                    if self.omp_stop_flag:
                        break
                    fc = FitnessCalculator(
                        bundle["gray"],
                        bundle["edges"],
                        bundle["saliency"],
                        ch["importance"],
                        cm,
                        config,
                        self.queues,
                        target_darkness_override=ch["darkness"],
                    )
                    self.active_fitness_calc = fc
                    self._send_status(f"CMYK слой {c_idx + 1}/4 ({ch['name']}): CSR OMP-синтез...")
                    greedy = GreedyInitializer(
                        cm, fc.target_darkness_flat, fc.importance_flat, config, radon_guide, stop_Check=self._check_omp_stop_or_pause
                    )

                    def greedy_cb_cmyk(status: str, path: List[int], idx=c_idx, ch_meta=ch) -> None:
                        snap = list(accumulated_layers)
                        snap.append({"name": ch_meta["name"], "color_hex": ch_meta["color_hex"], "path": path})
                        self.queues["status_queue"].put({
                            "type": "greedy_update",
                            "data": {"text": f"Слой {idx + 1}/4 ({ch_meta['name']}): {status}", "layers": snap, "active_idx": idx},
                        })

                    base_path = greedy.run(config["lines"], greedy_cb_cmyk, layer_label=ch["name"])
                    if len(base_path) >= 2 and not self.omp_stop_flag:
                        omp_layer_fit = fc.calculate_single_fitness(base_path)
                        self.state.omp_baseline_fitness = omp_layer_fit
                        self.state.layer_best_fitnesses[c_idx] = omp_layer_fit - 1e-6
                        snap_pre = list(accumulated_layers) + [{"name": ch["name"], "color_hex": ch["color_hex"], "path": list(base_path)}]
                        self.queues["status_queue"].put({
                            "type": "greedy_update",
                            "data": {"text": f"Слой {c_idx + 1}/4 ({ch['name']}): эволюционная доводка...", "layers": snap_pre, "active_idx": c_idx},
                        })
                        cmyk_stages = max(2, config.get("stages", 5) // 2)
                        mgr = IslandManagerThread(
                            config,
                            self.queues,
                            fc,
                            base_path,
                            radon_guide,
                            layer_idx=c_idx,
                            layer_name=ch["name"],
                            emit_done=False,
                            stages_override=cmyk_stages,
                            patience_scale=0.55,
                        )
                        self.manager_thread = mgr
                        mgr.start()
                        mgr.join()
                        if mgr.best_result and len(mgr.best_result.get("chromosome", [])) >= 2:
                            base_path = list(mgr.best_result["chromosome"])

                    accumulated_layers.append({"name": ch["name"], "color_hex": ch["color_hex"], "path": base_path})
                    self.queues["status_queue"].put({
                        "type": "greedy_update",
                        "data": {"text": f"Слой {ch['name']} готов", "layers": list(accumulated_layers), "active_idx": c_idx},
                    })

                msg = "Остановлено пользователем." if self.omp_stop_flag else "Цветной CMYK синтез и полировка завершены!"
                self.queues["status_queue"].put({"type": "done", "data": msg})

        except Exception:
            logging.error("Ошибка при подготовке эволюции", exc_info=True)
            self.queues["status_queue"].put({"type": "error", "data": f"Ошибка при подготовке: {traceback.format_exc()}"})

    def _check_omp_stop_or_pause(self) -> bool:
        while not self.omp_pause_event.is_set() and not self.omp_stop_flag:
            time.sleep(0.06)
        return self.omp_stop_flag

    def _push_analytic_maps(self, diff_map: np.ndarray, depth_2d: np.ndarray) -> None:
        self.render_results_queue.put(("diff_map", self._create_fast_diff_pil(diff_map)))
        self.render_results_queue.put(("density_heatmap", self._create_fast_heatmap_pil(depth_2d)))

    def _create_fast_diff_pil(self, diff_map: np.ndarray) -> Image.Image:
        h, w = diff_map.shape
        rgb = np.full((h, w, 3), 255, dtype=np.uint8)
        max_v = max(0.05, float(np.max(np.abs(diff_map))))
        norm = np.clip(diff_map / max_v, -1.0, 1.0)

        pos = norm > 0
        inten_pos = (norm[pos] * 255).astype(np.uint8)
        rgb[pos, 1] = 255 - inten_pos
        rgb[pos, 2] = 255 - inten_pos

        neg = norm < 0
        inten_neg = ((-norm[neg]) * 255).astype(np.uint8)
        rgb[neg, 0] = 255 - inten_neg
        rgb[neg, 1] = 255 - inten_neg
        return Image.fromarray(rgb)

    def _create_fast_heatmap_pil(self, depth_2d: np.ndarray) -> Image.Image:
        max_d = max(0.1, float(np.max(depth_2d)))
        norm_u8 = np.clip((depth_2d / max_d) * 255.0, 0, 255).astype(np.uint8)
        colored_bgr = cv2.applyColorMap(norm_u8, cv2.COLORMAP_INFERNO)
        return Image.fromarray(cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB))

    def stop_generation(self) -> None:
        self.omp_stop_flag = True
        self.omp_pause_event.set()
        self._cancel_stats_job()
        if self.manager_thread and self.manager_thread.is_alive():
            self.manager_thread.stop()
            self._send_status("Остановка потоков...")
        self.root.after(250, self._finalize_run)

    def _finalize_run(self) -> None:
        self._cancel_stats_job()
        self.ui_manager.toggle_ui_state(False)

    def toggle_pause(self) -> None:
        if not self.state.is_running:
            return
        self.state.is_paused = not self.state.is_paused
        if self.state.is_paused:
            self.omp_pause_event.clear()
            if self.manager_thread and self.manager_thread.is_alive():
                self.manager_thread.pause()
            self.ui_manager.widgets["pause_button"].config(text="► Продолжить")
            self._send_status("Пауза...")
        else:
            self.omp_pause_event.set()
            if self.manager_thread and self.manager_thread.is_alive():
                self.manager_thread.resume()
            self.ui_manager.widgets["pause_button"].config(text="❚❚ Пауза")
            self.stats_update_job = self.root.after(1000, self._update_progress_stats)

    def process_queues(self) -> None:
        try:
            while not self.queues["status_queue"].empty():
                msg = self.queues["status_queue"].get_nowait()
                handler = self._get_message_handler(msg.get("type"))
                handler(msg.get("data"))

            while not self.render_results_queue.empty():
                key, img_pil = self.render_results_queue.get_nowait()
                self.ui_manager.update_canvas_image(key, img_pil)
        except queue.Empty:
            pass
        finally:
            if not self._renderer_stop:
                self.root.after(60, self.process_queues)

    def _get_message_handler(self, msg_type: Optional[str]) -> Callable:
        return {
            "greedy_update": self._handle_greedy_update,
            "status": self._handle_status,
            "error": self._handle_error,
            "gpu_status": self._handle_gpu_status,
            "generation_tick": self._handle_generation_tick,
            "island_update": self._handle_island_update,
            "done": self._handle_done,
            "migration": self._handle_migration,
            "fit_func_maps": self._handle_fit_func_maps,
            "radon_map": self._handle_radon_map,
        }.get(msg_type or "", lambda _d: None)

    def _handle_greedy_update(self, data: Dict[str, Any]) -> None:
        self.ui_manager.widgets["status_label"].config(text=data["text"])
        layers_snap = data["layers"]
        active_idx = data.get("active_idx", 0)
        self.state.layers = layers_snap
        self.state.active_layer_idx = active_idx
        active_path = layers_snap[active_idx].get("path", []) if active_idx < len(layers_snap) else []
        self.state.overall_best_chromosome = active_path
        self.best_chromosomes_by_key["overall"] = layers_snap
        self.ui_manager.update_layer_selector()

        max_steps = max(1, len(active_path) - 1)
        scale = self.ui_manager.widgets.get("step_scrubber_scale")
        if isinstance(scale, ttk.Scale):
            scale.configure(to=max(10, max_steps))
        self.state.vars["step_scrubber"].set(max_steps)
        self.ui_manager.update_player_step_label()

        config = self.state.get_config()
        self.request_canvas_update(
            "overall",
            (layers_snap, config["pins"], config["thread_thickness"]),
            {"ignore_scrubber": True},
        )

    def _handle_status(self, data: str) -> None:
        self.ui_manager.widgets["status_label"].config(text=data)

    def _handle_error(self, data: str) -> None:
        self._cancel_stats_job()
        messagebox.showerror("Ошибка", data)
        self.ui_manager.widgets["status_label"].config(text="Ошибка!")
        self.ui_manager.toggle_ui_state(False)

    def _handle_gpu_status(self, data: bool) -> None:
        mode = f"PyTorch CUDA ({CUDA_DEVICE_NAME}) + CSR" if data else "SciPy CSR + NumPy (CPU)"
        self.ui_manager.widgets["compute_mode_label"].config(text=f"Ядро: {mode}")

    def _handle_generation_tick(self, data: Tuple[int, int, int, int]) -> None:
        island_id, stage, patience, limit = data
        total_gens = self.state.generation_counters.get(island_id, 0) + 1
        self.state.generation_counters[island_id] = total_gens
        self.state.island_progress_data[island_id] = (stage, limit)
        self.ui_manager.update_island_progress(island_id, stage, patience, limit)
        if total_gens % 6 == 0:
            cur_f = self.state.island_best_fitnesses.get(island_id, 0.0)
            if cur_f > 0:
                self.state.chart_data_by_island.setdefault(island_id, []).append(
                    (sum(self.state.generation_counters.values()), cur_f)
                )
                self.ui_manager.chart_needs_update = True

    def _handle_island_update(self, data: Dict[str, Any]) -> None:
        island_id, chromosome, fitness = data["id"], data["chromosome"], data["fitness"]
        layer_idx = data.get("layer_idx", self.state.active_layer_idx)
        mut_stats = data.get("mut_stats", {})
        if mut_stats:
            self.state.aggregated_mut_stats = mut_stats
            self.ui_manager.chart_needs_update = True

        self.state.island_best_fitnesses[island_id] = fitness
        self.best_chromosomes_by_key[island_id] = chromosome
        config = self.state.get_config()

        notebook = self.ui_manager.widgets["results_notebook"]
        sel = notebook.select()
        if sel and notebook.index(sel) == island_id + 1:
            self.request_canvas_update(island_id, (chromosome, config["pins"], config["thread_thickness"]), {})

        if fitness > self.state.layer_best_fitnesses.get(layer_idx, -1.0):
            self.state.layer_best_fitnesses[layer_idx] = fitness
            self.state.overall_best_fitness = fitness
            self.state.overall_best_chromosome = list(chromosome)
            if layer_idx < len(self.state.layers):
                self.state.layers[layer_idx]["path"] = list(chromosome)
            elif layer_idx == len(self.state.layers):
                self.state.layers.append({"name": "Черная нить", "color_hex": "#111111", "path": list(chromosome)})

            layers_snap = list(self.state.layers)
            self.best_chromosomes_by_key["overall"] = layers_snap

            max_steps = max(1, len(chromosome) - 1)
            scale = self.ui_manager.widgets.get("step_scrubber_scale")
            if isinstance(scale, ttk.Scale):
                scale.configure(to=max(10, max_steps))
            self.state.vars["step_scrubber"].set(max_steps)
            self.ui_manager.update_player_step_label()

            self.ui_manager.update_best_fitness_display(fitness)
            total_m = sum(
                self.canvas_manager.calculate_path_length_m(l.get("path", [])) for l in layers_snap
            ) if self.canvas_manager else 0.0
            self.ui_manager.update_thread_length_display(total_m)
            self.request_canvas_update("overall", (layers_snap, config["pins"], config["thread_thickness"]), {"ignore_scrubber": True})

    def _handle_fit_func_maps(self, data: Dict[str, Optional[np.ndarray]]) -> None:
        if data.get("saliency") is not None:
            sal_bgr = cv2.applyColorMap(data["saliency"], cv2.COLORMAP_VIRIDIS)
            sal_rgb = cv2.cvtColor(sal_bgr, cv2.COLOR_BGR2RGB)
            self.render_results_queue.put(("fit_saliency", Image.fromarray(sal_rgb)))

    def _handle_radon_map(self, data: Any) -> None:
        if data is None:
            return
        if isinstance(data, tuple):
            sinogram_data, fbp_recon = data
        else:
            sinogram_data, fbp_recon = data, None
        if sinogram_data is None:
            return
        sino_u8 = np.clip(sinogram_data * 255.0, 0, 255).astype(np.uint8)
        sino_resized = cv2.resize(sino_u8, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        sino_color = cv2.cvtColor(cv2.applyColorMap(sino_resized, cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)

        if fbp_recon is not None and fbp_recon.size > 0:
            fbp_resized = cv2.resize(fbp_recon, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
            fbp_rgb = cv2.cvtColor(fbp_resized, cv2.COLOR_GRAY2RGB)
            combined = np.hstack([sino_color, np.full((IMAGE_SIZE, 6, 3), 24, dtype=np.uint8), fbp_rgb])
            self.render_results_queue.put(("radon_sinogram", Image.fromarray(combined)))
        else:
            self.render_results_queue.put(("radon_sinogram", Image.fromarray(sino_color)))

    def _handle_migration(self, _: Any) -> None:
        self.state.migration_count += 1

    def _handle_done(self, data: str) -> None:
        removed = self.prune_chromosome(silent=True)
        self._finalize_run()
        self.ui_manager.update_layer_selector()
        total_lines = sum(max(0, len(l.get("path", [])) - 1) for l in self.state.layers)
        msg = f"{data} (Нитей: {total_lines}, удалено петель: {removed})"
        self.ui_manager.widgets["status_label"].config(text=msg)
        self.ui_manager.update_player_step_label()

        # Финальное обновление аналитических карт
        config = self.state.get_config()
        if self.state.layers:
            self.request_canvas_update(
                "overall",
                (list(self.state.layers), config["pins"], config["thread_thickness"]),
                {"force_maps": True, "ignore_scrubber": True},
            )

    def _send_status(self, message: str) -> None:
        self.queues["status_queue"].put({"type": "status", "data": message})

    def request_canvas_update(self, key: Any, data: Any, render_params: Dict[str, Any]) -> None:
        if not self.canvas_manager:
            return
        canvas = self.ui_manager.result_canvases.get(key)
        w = canvas.winfo_width() if canvas else 550
        h = canvas.winfo_height() if canvas else 550
        out_size = max(320, min(w, h) - 10) if (w > 20 and h > 20) else 550
        params = dict(render_params)
        params["output_size"] = out_size
        params["line_weight"] = int(self.state.safe_get("line_weight") or 72)
        params["active_layer_idx"] = int(self.state.active_layer_idx)
        params["show_active_chord"] = bool(self.state.safe_get("show_active_chord"))

        if key == "overall" and not params.get("ignore_scrubber"):
            params["max_step"] = int(self.state.safe_get("step_scrubber") or 1)
        else:
            params["max_step"] = None

        with self._render_cond:
            self._pending_renders[key] = (data, params)
            self._render_cond.notify()

    def _update_progress_stats(self) -> None:
        if not self.state.is_running:
            return
        self.ui_manager.update_stats_display(self.state)
        self.stats_update_job = self.root.after(1000, self._update_progress_stats)

    def prune_chromosome(self, silent: bool = False) -> int:
        if not self.state.layers and not self.state.overall_best_chromosome:
            return 0

        total_removed = 0
        if self.state.layers:
            for l_idx, layer in enumerate(self.state.layers):
                pruned, rem = prune_chromosome_loops(layer.get("path", []))
                layer["path"] = pruned
                total_removed += rem
                if l_idx == self.state.active_layer_idx:
                    self.state.overall_best_chromosome = pruned
        elif self.state.overall_best_chromosome:
            pruned, total_removed = prune_chromosome_loops(self.state.overall_best_chromosome)
            self.state.overall_best_chromosome = pruned

        if total_removed > 0:
            active_path = self.state.overall_best_chromosome or []
            max_steps = max(1, len(active_path) - 1)
            scale = self.ui_manager.widgets.get("step_scrubber_scale")
            if isinstance(scale, ttk.Scale):
                scale.configure(to=max(10, max_steps))
            self.state.vars["step_scrubber"].set(max_steps)
            self.ui_manager.update_player_step_label()

            if self.state.layers:
                self.best_chromosomes_by_key["overall"] = list(self.state.layers)
                config = self.state.get_config()
                self.request_canvas_update(
                    "overall",
                    (list(self.state.layers), config["pins"], config["thread_thickness"]),
                    {"force": True},
                )
                if self.canvas_manager:
                    total_m = sum(self.canvas_manager.calculate_path_length_m(l.get("path", [])) for l in self.state.layers)
                    self.ui_manager.update_thread_length_display(total_m)

            if not silent:
                total_lines = sum(max(0, len(l.get("path", [])) - 1) for l in self.state.layers)
                messagebox.showinfo(
                    "Оптимизация пути",
                    f"Удалено холостых возвратных петель: {total_removed}\nИтого четких нитей: {total_lines}",
                )
        elif not silent:
            messagebox.showinfo("Оптимизация пути", "Холостых петель не обнаружено, путь уже оптимален!")
        return total_removed

    def _get_export_cm(self) -> Optional[CanvasManager]:
        if not self.state.layers and not self.state.overall_best_chromosome:
            messagebox.showwarning("Внимание", "Сначала выполните синтез картины.")
            return None
        config = self.state.get_config()
        if self.canvas_manager and self.canvas_manager.num_pins == config["pins"] and self.canvas_manager.shape == config["canvas_shape"]:
            self.canvas_manager.physical_cm = float(config["physical_diameter"])
            return self.canvas_manager
        return CanvasManager(IMAGE_SIZE, config["pins"], config["canvas_shape"], config["physical_diameter"])

    def _get_export_layers(self) -> List[Dict[str, Any]]:
        if self.state.layers:
            return self.state.layers
        return [{"name": "Черная нить", "color_hex": "#111111", "path": self.state.overall_best_chromosome or []}]

    def export_txt(self) -> None:
        cm = self._get_export_cm()
        if cm is None:
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if not filepath:
            return
        export_instructions_txt(filepath, self._get_export_layers(), cm)
        messagebox.showinfo("Экспорт TXT", f"Схема гвоздей успешно сохранена:\n{filepath}")

    def export_csv(self) -> None:
        cm = self._get_export_cm()
        if cm is None:
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not filepath:
            return
        export_csv(filepath, self._get_export_layers(), cm)
        messagebox.showinfo("Экспорт CSV", f"Таблица переходов CSV сохранена:\n{filepath}")

    def export_gcode(self) -> None:
        cm = self._get_export_cm()
        if cm is None:
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".gcode", filetypes=[("CNC G-Code", "*.gcode *.nc")])
        if not filepath:
            return
        export_gcode_cnc(filepath, self._get_export_layers(), cm)
        messagebox.showinfo("Экспорт G-Code", f"ЧПУ-программа G-Code сохранена:\n{filepath}")

    def export_png(self) -> None:
        cm = self._get_export_cm()
        if cm is None:
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png")])
        if not filepath:
            return
        config = self.state.get_config()
        self._send_status("Экспорт Ultra HD PNG (2400x2400px)...")
        self.root.update_idletasks()
        export_png_ultra_hd(
            filepath,
            self._get_export_layers(),
            cm,
            thickness=config["thread_thickness"],
            line_weight_percent=config["line_weight"],
            output_size=2400,
        )
        self._send_status("Экспорт PNG 2400x2400 завершен.")
        messagebox.showinfo("Экспорт PNG", f"Изображение высокого разрешения сохранено в:\n{filepath}")

    def export_svg(self) -> None:
        cm = self._get_export_cm()
        if cm is None:
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG vector", "*.svg")])
        if not filepath:
            return
        config = self.state.get_config()
        export_svg_vector(filepath, self._get_export_layers(), cm, thread_thickness=config["thread_thickness"])
        messagebox.showinfo("Экспорт SVG", f"Векторный файл SVG успешно сохранен:\n{filepath}")

    def export_template(self) -> None:
        config = self.state.get_config()
        cm = self.canvas_manager or CanvasManager(IMAGE_SIZE, config["pins"], config["canvas_shape"], config["physical_diameter"])
        cm.physical_cm = float(config["physical_diameter"])
        filepath = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("Drill Template SVG", "*.svg")])
        if not filepath:
            return
        export_drill_template_svg(filepath, cm)
        messagebox.showinfo("Шаблон 1:1", f"Чертеж сверления гвоздей 1:1 (в мм) сохранен:\n{filepath}")

    def export_html(self) -> None:
        cm = self._get_export_cm()
        if cm is None:
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML5 Web Assistant", "*.html")],
            title="Сохранить автономный веб-ассистент плетения",
        )
        if not filepath:
            return
        try:
            title = "String Art Masterpiece"
            if self.state.original_image_path:
                title = os.path.splitext(os.path.basename(self.state.original_image_path))[0]
            export_html_assistant(filepath, title, cm, self._get_export_layers())
            messagebox.showinfo(
                "Успех",
                f"Автономный веб-ассистент успешно сохранен в:\n{filepath}\n\n"
                "Откройте файл в любом браузере на ПК, планшете или смартфоне!",
            )
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось экспортировать HTML:\n{exc}")


class StringArtApp(tk.Tk):
    """Главное окно String Art Studio Pro."""

    def __init__(self):
        enable_windows_high_dpi()
        super().__init__()
        self.geometry("1540x940")
        self.minsize(1120, 740)

        self._load_window_icon()

        self.state = AppState()
        self.queues = {"status_queue": queue.Queue()}
        self.audio_assistant = AudioAssistant()
        self.ui_manager = UIManager(self, self.state)
        self.coordinator = EvolutionCoordinator(self, self.state, self.ui_manager, self.queues, self.audio_assistant)
        self.ui_manager.coordinator = self.coordinator
        self._auto_play_job: Optional[str] = None

        self.ui_manager.create_widgets()
        self.ui_manager.update_title()
        self.ui_manager.widgets["load_image_button"].config(command=self.load_image)
        self.ui_manager.setup_for_new_run(int(self.state.safe_get("num_islands")))

        self._bind_shortcuts()
        self.after(60, self.coordinator.process_queues)
        self.after(800, self._periodic_ui_update)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _load_window_icon(self) -> None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for cand in [
            os.path.join(base_dir, "StringArt.ico"),
            os.path.join(os.getcwd(), "StringArt.ico"),
        ]:
            if os.path.exists(cand):
                try:
                    self.iconbitmap(cand)
                    break
                except Exception:
                    pass

    def _bind_shortcuts(self) -> None:
        def _is_typing(event) -> bool:
            return isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Text))

        def _on_next(event):
            if not _is_typing(event):
                self.step_player(1)
                return "break"

        def _on_prev(event):
            if not _is_typing(event):
                self.step_player(-1)
                return "break"

        self.bind("<space>", _on_next)
        self.bind("<Right>", _on_next)
        self.bind("<d>", _on_next)
        self.bind("<D>", _on_next)
        self.bind("<Left>", _on_prev)
        self.bind("<a>", _on_prev)
        self.bind("<A>", _on_prev)
        self.bind("<Control-o>", lambda _e: self.load_image())

    def on_closing(self) -> None:
        if self.state.is_running and not messagebox.askokcancel("Выход", "Генерация еще выполняется. Остановить и выйти?"):
            return
        if self._auto_play_job:
            self.after_cancel(self._auto_play_job)
        if self.state.is_running:
            self.coordinator.stop_generation()
        self.coordinator.shutdown()
        self.audio_assistant.shutdown()
        self.destroy()

    def _periodic_ui_update(self) -> None:
        if self.ui_manager.chart_needs_update:
            self.ui_manager.update_chart()
        self.after(800, self._periodic_ui_update)

    def on_preset_selected(self, _event=None) -> None:
        name = self.state.vars["preset"].get()
        preset = PRESETS.get(name)
        if preset is None:
            self.apply_auto_recommendations()
            return
        for k, v in preset.items():
            if k in self.state.vars:
                self.state.vars[k].set(v)
        self.ui_manager.widgets["analysis_label"].config(text=f"Пресет: {name}", foreground="#0969da")
        self.redraw_target_canvas()

    def on_voice_toggle(self) -> None:
        self.audio_assistant.toggle(bool(self.state.safe_get("voice_enabled")))

    def step_player(self, delta: int) -> None:
        if not self.state.layers or self.state.active_layer_idx >= len(self.state.layers):
            return
        path = self.state.layers[self.state.active_layer_idx].get("path", [])
        if len(path) < 2:
            return
        max_s = len(path) - 1
        cur = int(self.state.safe_get("step_scrubber") or 1)
        new_s = max(1, min(max_s, cur + delta))
        if new_s != cur:
            self.state.vars["step_scrubber"].set(new_s)
            self.ui_manager.update_player_step_label()
            if self.state.safe_get("voice_enabled") and delta > 0 and new_s < len(path):
                self.audio_assistant.announce_chord(path[new_s - 1], path[new_s])
            self.ui_manager._on_visual_param_change()

    def toggle_auto_play(self) -> None:
        btn = self.ui_manager.widgets.get("auto_play_button")
        if self._auto_play_job:
            self.after_cancel(self._auto_play_job)
            self._auto_play_job = None
            if btn:
                btn.config(text="▶ Авто")
        else:
            if not self.state.layers:
                return
            if btn:
                btn.config(text="❚❚ Стоп")
            self._auto_play_tick()

    def _auto_play_tick(self) -> None:
        if not self.state.layers or self.state.active_layer_idx >= len(self.state.layers):
            self.toggle_auto_play()
            return
        path = self.state.layers[self.state.active_layer_idx].get("path", [])
        max_s = max(1, len(path) - 1)
        cur = int(self.state.safe_get("step_scrubber") or 1)
        if cur < max_s:
            self.step_player(1)
            self._auto_play_job = self.after(2800, self._auto_play_tick)
        else:
            self.toggle_auto_play()

    def load_image(self) -> None:
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp"), ("All files", "*.*")]
        )
        if not file_path:
            return
        try:
            with Image.open(file_path) as img_raw:
                img_copy = img_raw.copy()
            self.state.original_image = img_copy
            self.state.original_image_path = file_path
            self.ui_manager.update_title()
            self.apply_auto_recommendations()
            self.ui_manager.widgets["status_label"].config(text="Изображение загружено и параметры подобраны.")
        except Exception as exc:
            logging.error("Не удалось загрузить изображение", exc_info=True)
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение:\n{exc}")

    def apply_auto_recommendations(self) -> None:
        if not self.state.original_image:
            return
        shape = str(self.state.safe_get("canvas_shape") or "Круг")
        rec = auto_recommend_parameters(self.state.original_image, shape)
        for k in [
            "pins",
            "lines",
            "line_weight",
            "clahe_clip",
            "gamma",
            "fearless_factor",
            "contour_weight",
            "radon_influence",
            "stages",
        ]:
            if k in self.state.vars:
                self.state.vars[k].set(rec[k])
        self.state.vars["step_scrubber"].set(rec["lines"])
        self.ui_manager.widgets["analysis_label"].config(text=rec["summary"], foreground="#0f5132")
        self.redraw_target_canvas()

    def redraw_target_canvas(self, _event=None) -> None:
        if not self.state.original_image:
            return
        canvas = self.ui_manager.widgets["target_canvas"]
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 20 or h < 20:
            return
        display_size = max(60, min(w, h) - 10)
        bundle = prepare_target_image_bundle(
            self.state.original_image,
            display_size,
            clahe_clip=float(self.state.safe_get("clahe_clip") or 1.3),
            gamma=float(self.state.safe_get("gamma") or 0.82),
            shape=str(self.state.safe_get("canvas_shape") or "Круг"),
            color_mode=str(self.state.safe_get("color_mode") or "Монохром (Черная нить)"),
        )
        self.state.preprocessed_target_np = bundle["gray"]
        self.state.preprocessed_rgb_np = bundle["rgb"]
        is_cmyk = "CMYK" in str(self.state.safe_get("color_mode"))
        img_pil = Image.fromarray(bundle["rgb"] if is_cmyk else bundle["gray"])
        self.state.target_image_tk = ImageTk.PhotoImage(img_pil)
        canvas.delete("all")
        canvas.create_image(w // 2, h // 2, anchor=tk.CENTER, image=self.state.target_image_tk)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(threadName)s - %(levelname)s - %(message)s")
    app = StringArtApp()
    app.mainloop()
