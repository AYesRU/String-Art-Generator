# -*- coding: utf-8 -*-
"""
Графический интерфейс String Art Studio Pro (Clean Light Theme, RU / EN),
интерактивный плеер, шторка сравнения, живая аналитика Matplotlib и переключение языка.
"""

import logging
import os
import queue
import tempfile
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_stringart"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from . import (
    APP_TITLE,
    PRESETS,
    TOOLTIPS,
)
from .audio import SAPI_AVAILABLE, AudioAssistant
from .coordinator import AppState, EvolutionCoordinator
from .engine import (
    CUDA_DEVICE_NAME,
    GPU_ENABLED_FLAG,
    SCIPY_AVAILABLE,
)
from .i18n import (
    LANGUAGE_LABELS,
    get_display_color_modes,
    get_display_metrics,
    get_display_presets,
    get_display_shapes,
    get_tooltip,
    normalize_lang,
    save_language_preference,
    to_canonical_color_mode,
    to_canonical_metric,
    to_canonical_preset,
    to_canonical_shape,
    to_display_color_mode,
    to_display_metric,
    to_display_preset,
    to_display_shape,
    tr,
    translate_layer_name,
)
from .preprocessing import auto_recommend_parameters, prepare_target_image_bundle

__all__ = [
    "enable_windows_high_dpi",
    "AppState",
    "EvolutionCoordinator",
    "ToolTip",
    "UIManager",
    "StringArtApp",
    "main",
]


def enable_windows_high_dpi() -> None:
    """Включает поддержку четкого масштабирования HiDPI на Windows."""
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


class ToolTip:
    """Компактная всплывающая подсказка при наведении на элемент интерфейса."""

    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tipwindow: Optional[tk.Toplevel] = None
        self.widget.bind("<Enter>", self.showtip, add="+")
        self.widget.bind("<Leave>", self.hidetip, add="+")

    def set_text(self, text: str) -> None:
        self.text = text

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
    """Управляет виджетами, светлой темой, графиками Matplotlib, холстами и локализацией RU/EN."""

    def __init__(self, root: "StringArtApp", state: AppState):
        self.root = root
        self.state = state
        self.coordinator: Optional[EvolutionCoordinator] = None
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
        self._i18n_widgets: List[Tuple[tk.Widget, str, str]] = []
        self._tooltips: List[Tuple[ToolTip, str]] = []
        self._last_thread_length_m: Optional[float] = None
        self._last_gpu_flag: bool = GPU_ENABLED_FLAG

    def _t(self, key: str, **kwargs) -> str:
        return tr(key, self.state.lang, **kwargs)

    def _reg_text(self, widget: tk.Widget, key: str, suffix: str = "") -> None:
        self._i18n_widgets.append((widget, key, suffix))

    def _add_tip(self, widget: tk.Widget, tip_key: str) -> None:
        if not tip_key:
            return
        tip = ToolTip(widget, get_tooltip(tip_key, self.state.lang, TOOLTIPS))
        self._tooltips.append((tip, tip_key))

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

        progress_frame = ttk.LabelFrame(main_pane, text=self._t("dashboard_frame"), height=220)
        self._reg_text(progress_frame, "dashboard_frame")
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

        sidebar_nb = ttk.Notebook(container)
        sidebar_nb.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.widgets["sidebar_notebook"] = sidebar_nb

        tab_image = ttk.Frame(sidebar_nb)
        sidebar_nb.add(tab_image, text=self._t("tab_ctrl_image"))

        tab_algo = ttk.Frame(sidebar_nb)
        sidebar_nb.add(tab_algo, text=self._t("tab_ctrl_algo"))

        tab_player = ttk.Frame(sidebar_nb)
        sidebar_nb.add(tab_player, text=self._t("tab_ctrl_player"))

        tab_export = ttk.Frame(sidebar_nb)
        sidebar_nb.add(tab_export, text=self._t("tab_ctrl_export"))

        self._create_image_settings_section(tab_image)
        self._create_geometry_section(tab_image)

        self._create_algorithm_section(tab_algo)
        self._create_fitness_and_radon_section(tab_algo)

        self._create_inspector_section(tab_player)
        self._create_control_and_export_section(tab_export)

    def _create_display_area(self, parent: ttk.PanedWindow) -> None:
        # Right side: vertical frame with display area on top and player bar below
        right_frame = ttk.Frame(parent)
        parent.add(right_frame, weight=1)

        display_area = ttk.PanedWindow(right_frame, orient=tk.HORIZONTAL)
        display_area.pack(fill=tk.BOTH, expand=True)

        target_frame = ttk.LabelFrame(display_area, text=self._t("target_frame"), width=340)
        self._reg_text(target_frame, "target_frame")
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

        self._create_player_bar(right_frame)

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
        elif 1 <= idx <= 4:
            key = ["diff_map", "density_heatmap", "fit_saliency", "radon_sinogram"][idx - 1]
            self.on_result_canvas_resize(None, key)

    def _create_player_bar(self, parent: tk.Widget) -> None:
        """Creates the player control bar below the display area."""
        bar = tk.Frame(parent, bg="#eaeef2")
        bar.pack(fill=tk.X, pady=3, padx=4)
        self.widgets["player_bar"] = bar

        # Active layer combobox
        layer_cb = ttk.Combobox(
            bar,
            textvariable=self.state.vars["active_layer"],
            values=[translate_layer_name("Черная нить", self.state.lang)],
            state="readonly",
            width=20,
        )
        layer_cb.pack(side=tk.LEFT, padx=(0, 2))
        layer_cb.bind("<<ComboboxSelected>>", self._on_active_layer_changed)
        self._add_tip(layer_cb, "active_layer")
        self.widgets["active_layer_combo"] = layer_cb

        # Separator
        tk.Label(bar, text="|", bg="#eaeef2", fg="#aaaaaa").pack(side=tk.LEFT, padx=2)

        # Player step label
        step_lbl = tk.Label(
            bar, text=self._t("player_idle"),
            font=("Segoe UI", 8, "bold"), fg="#0969da", bg="#eaeef2"
        )
        step_lbl.pack(side=tk.LEFT, padx=(0, 4))
        self.widgets["player_step_label"] = step_lbl

        # Prev button
        btn_prev = ttk.Button(bar, text="◀", width=3, command=lambda: self.root.step_player(-1))
        btn_prev.pack(side=tk.LEFT, padx=1)

        # Auto play button
        btn_auto = ttk.Button(bar, text=self._t("btn_auto_play"), command=self.root.toggle_auto_play)
        btn_auto.pack(side=tk.LEFT, padx=1)
        self.widgets["auto_play_button"] = btn_auto

        # Next button
        btn_next = ttk.Button(bar, text="▶", width=3, command=lambda: self.root.step_player(1))
        btn_next.pack(side=tk.LEFT, padx=1)

        # Step scrubber slider
        scrubber_var = self.state.vars["step_scrubber"]
        self._slider_limits["step_scrubber"] = (1.0, 4500.0, True)

        def _scrubber_cmd(v):
            scrubber_var.set(int(round(float(v))))
            self._on_scrubber_slider_move()

        scrubber = ttk.Scale(
            bar, from_=1, to=4500,
            variable=scrubber_var, orient=tk.HORIZONTAL,
            command=_scrubber_cmd,
        )
        scrubber.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.widgets["step_scrubber_scale"] = scrubber
        self._add_tip(scrubber, "step_scrubber")

        # Show active chord checkbutton (✨)
        chk_neon = ttk.Checkbutton(
            bar, text="✨",
            variable=self.state.vars["show_active_chord"],
            command=self._on_visual_param_change,
        )
        chk_neon.pack(side=tk.LEFT, padx=1)
        self._add_tip(chk_neon, "show_active_chord")

        # Voice checkbutton (🔊)
        chk_voice = ttk.Checkbutton(
            bar, text="🔊",
            variable=self.state.vars["voice_enabled"],
            command=self.root.on_voice_toggle,
        )
        if not SAPI_AVAILABLE:
            chk_voice.config(state=tk.DISABLED)
        chk_voice.pack(side=tk.LEFT, padx=1)
        self._add_tip(chk_voice, "voice_enabled")

        # Split compare: mirror label + small scale
        tk.Label(bar, text="🪞", bg="#eaeef2").pack(side=tk.LEFT, padx=(4, 0))
        split_var = self.state.vars["split_compare"]
        self._slider_limits["split_compare"] = (0.0, 100.0, True)

        def _split_cmd(v):
            split_var.set(int(round(float(v))))
            self._on_visual_param_change()

        split_scale = ttk.Scale(
            bar, from_=0, to=100,
            variable=split_var, orient=tk.HORIZONTAL,
            command=_split_cmd,
        )
        split_scale.pack(side=tk.LEFT, padx=2)
        split_scale.configure(length=120)
        self._add_tip(split_scale, "split_compare")

    def _create_section(self, parent: tk.Widget, title_key: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=self._t(title_key), padding=6)
        self._reg_text(frame, title_key)
        frame.pack(fill=tk.X, pady=3, padx=3, anchor="n")
        return frame

    def _create_image_settings_section(self, parent: tk.Widget) -> None:
        frame = self._create_section(parent, "sec_image")

        # Переключатель языка (RU / EN)
        f_lang = ttk.Frame(frame)
        f_lang.pack(fill=tk.X, pady=(0, 2))
        lbl_lang = ttk.Label(f_lang, text=self._t("lbl_language"), width=11)
        lbl_lang.pack(side=tk.LEFT)
        self._reg_text(lbl_lang, "lbl_language")
        lang_cb = ttk.Combobox(
            f_lang,
            textvariable=self.state.vars["language"],
            values=[LANGUAGE_LABELS["ru"], LANGUAGE_LABELS["en"]],
            state="readonly",
        )
        lang_cb.pack(fill=tk.X, expand=True)
        lang_cb.bind("<<ComboboxSelected>>", self._on_language_selected)
        self._add_tip(lang_cb, "language")
        self.widgets["language_combo"] = lang_cb

        btn = ttk.Button(frame, text=self._t("btn_load_image"))
        btn.pack(fill=tk.X, pady=2)
        self._reg_text(btn, "btn_load_image")
        self._add_tip(btn, "load_image")
        self.widgets["load_image_button"] = btn

        proj_row = ttk.Frame(frame)
        proj_row.pack(fill=tk.X, pady=1)
        btn_save_proj = ttk.Button(
            proj_row,
            text=self._t("btn_save_project"),
            command=self.coordinator.save_project,
            state=tk.DISABLED,
        )
        btn_load_proj = ttk.Button(
            proj_row,
            text=self._t("btn_load_project"),
            command=self.coordinator.load_project,
        )
        btn_save_proj.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 1))
        btn_load_proj.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(1, 0))
        self._reg_text(btn_save_proj, "btn_save_project")
        self._reg_text(btn_load_proj, "btn_load_project")
        self._add_tip(btn_save_proj, "save_project")
        self._add_tip(btn_load_proj, "load_project")
        self.widgets["save_project_button"] = btn_save_proj
        self.widgets["load_project_button"] = btn_load_proj

        f_pre = ttk.Frame(frame)
        f_pre.pack(fill=tk.X, pady=2)
        lbl_pre = ttk.Label(f_pre, text=self._t("lbl_preset"), width=9)
        lbl_pre.pack(side=tk.LEFT)
        self._reg_text(lbl_pre, "lbl_preset")
        preset_cb = ttk.Combobox(
            f_pre,
            textvariable=self.state.vars["preset"],
            values=get_display_presets(self.state.lang),
            state="readonly",
        )
        preset_cb.pack(fill=tk.X, expand=True)
        preset_cb.bind("<<ComboboxSelected>>", self.root.on_preset_selected)
        self._add_tip(preset_cb, "preset")
        self.widgets["preset_combo"] = preset_cb

        lbl_info = ttk.Label(
            frame,
            text=self._t("analysis_prompt"),
            font=("Segoe UI", 8),
            foreground="#333333",
            wraplength=245,
        )
        lbl_info.pack(fill=tk.X, pady=2)
        self.widgets["analysis_label"] = lbl_info

        btn_auto = ttk.Button(
            frame,
            text=self._t("btn_auto"),
            command=self.root.apply_auto_recommendations,
        )
        btn_auto.pack(fill=tk.X, pady=2)
        self._reg_text(btn_auto, "btn_auto")

        self._create_double_slider_row(frame, "gamma", "sl_gamma", (0.50, 1.40), command=self.root.redraw_target_canvas)
        self._create_double_slider_row(frame, "clahe_clip", "sl_clahe_clip", (0.0, 3.5), command=self.root.redraw_target_canvas)
        self.widgets["image_settings_frame"] = frame

    def _create_geometry_section(self, parent: tk.Widget) -> None:
        frame = self._create_section(parent, "sec_geometry")
        f_shape = ttk.Frame(frame)
        f_shape.pack(fill=tk.X, pady=2)
        lbl_sh = ttk.Label(f_shape, text=self._t("lbl_canvas_shape"), width=15)
        lbl_sh.pack(side=tk.LEFT)
        self._reg_text(lbl_sh, "lbl_canvas_shape")
        shape_combo = ttk.Combobox(
            f_shape,
            textvariable=self.state.vars["canvas_shape"],
            values=get_display_shapes(self.state.lang),
            state="readonly",
        )
        shape_combo.pack(fill=tk.X, expand=True)
        shape_combo.bind("<<ComboboxSelected>>", self.root.redraw_target_canvas)
        self._add_tip(shape_combo, "canvas_shape")
        self.widgets["shape_combo"] = shape_combo

        f_color = ttk.Frame(frame)
        f_color.pack(fill=tk.X, pady=2)
        lbl_col = ttk.Label(f_color, text=self._t("lbl_color_mode"), width=15)
        lbl_col.pack(side=tk.LEFT)
        self._reg_text(lbl_col, "lbl_color_mode")
        color_combo = ttk.Combobox(
            f_color,
            textvariable=self.state.vars["color_mode"],
            values=get_display_color_modes(self.state.lang),
            state="readonly",
        )
        color_combo.pack(fill=tk.X, expand=True)
        color_combo.bind("<<ComboboxSelected>>", self.root.redraw_target_canvas)
        color_combo.bind("<<ComboboxSelected>>", self._update_adaptive_n_visibility, add="+")
        self._add_tip(color_combo, "color_mode")
        self.widgets["color_combo"] = color_combo

        # Spinbox для адаптивного режима N цветов
        self._adaptive_n_frame = ttk.Frame(frame)
        self._adaptive_n_frame.pack(fill=tk.X, pady=1)
        lbl_n = ttk.Label(self._adaptive_n_frame, text=self._t("sl_adaptive_n_colors"), width=15)
        lbl_n.pack(side=tk.LEFT)
        self._reg_text(lbl_n, "sl_adaptive_n_colors")
        spn_n = ttk.Spinbox(
            self._adaptive_n_frame, from_=2, to=20,
            textvariable=self.state.vars["adaptive_n_colors"], width=5, state="readonly",
        )
        spn_n.pack(side=tk.RIGHT, padx=(2, 0))
        self.widgets["adaptive_n_spinbox"] = spn_n

        # Визуальная палитра нитей (чипы выбранных цветов из каталога)
        self._palette_frame = ttk.Frame(frame)
        self._palette_frame.pack(fill=tk.X, pady=1)

        self._update_adaptive_n_visibility()

        self._create_slider_row(frame, "pins", "sl_pins", (120, 480))
        self._create_double_slider_row(frame, "physical_diameter", "sl_physical_diameter", (20.0, 120.0))
        self._create_slider_row(frame, "lines", "sl_lines", (400, 4500))
        self._create_slider_row(frame, "line_weight", "sl_line_weight", (40, 95), command=self._on_visual_param_change)
        self._create_slider_row(frame, "thread_thickness", "sl_thread_thickness", (1, 3), command=self._on_visual_param_change)
        self._create_slider_row(frame, "fearless_factor", "sl_fearless_factor", (70, 100))
        self.widgets["geometry_frame"] = frame

    def _create_algorithm_section(self, parent: tk.Widget) -> None:
        frame = self._create_section(parent, "sec_algo")
        sliders = [
            ("num_islands", "sl_num_islands", (1, max(2, os.cpu_count() or 4))),
            ("population_size", "sl_population_size", (8, 80)),
            ("stages", "sl_stages", (2, 12)),
            ("mutation_rate", "sl_mutation_rate", (5, 70)),
            ("migration_interval", "sl_migration_interval", (5, 50)),
        ]
        for name, text_key, limits in sliders:
            self._create_slider_row(frame, name, text_key, limits)
        self.widgets["algo_frame"] = frame

    def _create_fitness_and_radon_section(self, parent: tk.Widget) -> None:
        frame = self._create_section(parent, "sec_fitness")
        f_struct = ttk.Frame(frame)
        f_struct.pack(fill=tk.X, pady=2)
        lbl_st = ttk.Label(f_struct, text=self._t("lbl_struct_metric"), width=15)
        lbl_st.pack(side=tk.LEFT)
        self._reg_text(lbl_st, "lbl_struct_metric")
        struct_combo = ttk.Combobox(
            f_struct,
            textvariable=self.state.vars["structural_metric"],
            values=get_display_metrics(self.state.lang, SCIPY_AVAILABLE),
            state="readonly",
        )
        struct_combo.pack(fill=tk.X, expand=True)
        self._add_tip(struct_combo, "structural_metric")
        self.widgets["struct_combo"] = struct_combo
        self._create_slider_row(frame, "contour_weight", "sl_contour_weight", (10, 90))

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=4)
        check_mem = ttk.Checkbutton(
            frame,
            text=self._t("chk_memetic"),
            variable=self.state.vars["memetic_enabled"],
        )
        check_mem.pack(anchor="w")
        self._reg_text(check_mem, "chk_memetic")
        self._add_tip(check_mem, "memetic_enabled")
        self._create_slider_row(frame, "memetic_intensity", "sl_memetic_intensity", (5, 50))
        self._create_slider_row(frame, "memetic_depth", "sl_memetic_depth", (5, 80))

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=4)
        check_radon = ttk.Checkbutton(
            frame,
            text=self._t("chk_radon"),
            variable=self.state.vars["radon_enabled"],
        )
        check_radon.pack(anchor="w")
        self._reg_text(check_radon, "chk_radon")
        self._add_tip(check_radon, "radon_enabled")
        self._create_slider_row(frame, "radon_influence", "sl_radon_influence", (0, 100))
        self.widgets["fitness_frame"] = frame

    def _create_inspector_section(self, parent: tk.Widget) -> None:
        frame = self._create_section(parent, "sec_inspector")
        kb_text = (
            "⌨ Space / D / → = Next step\n"
            "⌨ A / ← = Prev step\n"
            "⌨ Player controls are in the bar below ↓"
        )
        lbl = ttk.Label(
            frame,
            text=kb_text,
            font=("Segoe UI", 8),
            foreground="#57606a",
            justify=tk.LEFT,
        )
        lbl.pack(anchor="w", pady=4, padx=4)
        self.widgets["inspector_frame"] = frame

    def _create_control_and_export_section(self, parent: tk.Widget) -> None:
        frame = self._create_section(parent, "sec_control")
        btn_start = ttk.Button(
            frame,
            text=self._t("btn_start"),
            style="Primary.TButton",
            command=self.coordinator.start_generation,
        )
        btn_start.pack(fill=tk.X, pady=2)
        self._reg_text(btn_start, "btn_start")
        self._add_tip(btn_start, "start")
        self.widgets["start_button"] = btn_start

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=2)
        for name, text_key, command, state in [
            ("polish", "btn_polish", self.coordinator.start_evolution_only, tk.NORMAL),
            ("pause", "btn_pause", self.coordinator.toggle_pause, tk.DISABLED),
            ("stop", "btn_stop", self.coordinator.stop_generation, tk.DISABLED),
        ]:
            btn = ttk.Button(btn_row, text=self._t(text_key), command=command, state=state)
            btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
            if name != "pause":
                self._reg_text(btn, text_key)
            self._add_tip(btn, name)
            self.widgets[f"{name}_button"] = btn

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=4)

        btn_prune = ttk.Button(
            frame,
            text=self._t("btn_prune"),
            command=self.coordinator.prune_chromosome,
            state=tk.DISABLED,
        )
        btn_prune.pack(fill=tk.X, pady=1)
        self._reg_text(btn_prune, "btn_prune")
        self._add_tip(btn_prune, "prune")
        self.widgets["prune_button"] = btn_prune

        # Compact icon-only export buttons row
        icon_row = ttk.Frame(frame)
        icon_row.pack(fill=tk.X, pady=2)
        export_icons = [
            ("export_html", "🌐", self.coordinator.export_html, "export_html"),
            ("export_txt", "📄", self.coordinator.export_txt, "export_txt"),
            ("export_csv", "📊", self.coordinator.export_csv, "export_csv"),
            ("export_png", "🖼", self.coordinator.export_png, "export_png"),
            ("export_svg", "📐", self.coordinator.export_svg, "export_svg"),
            ("export_template", "⭕", self.coordinator.export_template, "export_template"),
            ("export_gcode", "⚙", self.coordinator.export_gcode, "export_gcode"),
        ]
        for name, icon, cmd, tip_key in export_icons:
            btn = ttk.Button(icon_row, text=icon, width=3, command=cmd, state=tk.DISABLED)
            btn.pack(side=tk.LEFT, padx=1)
            self._add_tip(btn, tip_key)
            self.widgets[f"{name}_button"] = btn

    def _create_stats_and_chart_area(self, parent: tk.Widget) -> None:
        bottom_panel = ttk.Frame(parent)
        bottom_panel.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left_panel = ttk.Frame(bottom_panel, width=295)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)

        self.widgets["status_label"] = ttk.Label(
            left_panel, text=self._t("status_waiting"), font=("Segoe UI", 9, "bold"), wraplength=280
        )
        self.widgets["status_label"].pack(anchor="w")
        self.widgets["fitness_label"] = ttk.Label(left_panel, text=self._t("fitness_idle"), foreground="#0969da")
        self.widgets["fitness_label"].pack(anchor="w", pady=(0, 2))

        mode_str = f"PyTorch CUDA ({CUDA_DEVICE_NAME}) + CSR" if GPU_ENABLED_FLAG else "SciPy CSR + NumPy (CPU)"
        self.widgets["compute_mode_label"] = ttk.Label(
            left_panel, text=self._t("engine_label", mode=mode_str), font=("Segoe UI", 8), foreground="#57606a"
        )
        self.widgets["compute_mode_label"].pack(anchor="w", pady=(0, 4))

        for key, text_key in [
            ("time", "stat_time"),
            ("eta", "stat_eta"),
            ("gens", "stat_gens"),
            ("mig", "stat_mig"),
            ("len", "stat_len"),
        ]:
            self.widgets[key + "_label"] = self._create_stat_row(left_panel, text_key)

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
        text_key: str,
        limits: Tuple[int, int],
        command: Optional[Callable] = None,
    ) -> None:
        self._slider_limits[name] = (float(limits[0]), float(limits[1]), True)
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=1)
        lbl = ttk.Label(f, text=f"{self._t(text_key)}:", width=15)
        lbl.pack(side=tk.LEFT)
        self._reg_text(lbl, text_key, suffix=":")
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
        self._add_tip(lbl, name)
        self._add_tip(entry, name)
        self._add_tip(scale, name)

    def _create_double_slider_row(
        self,
        parent: tk.Widget,
        name: str,
        text_key: str,
        limits: Tuple[float, float],
        command: Optional[Callable] = None,
    ) -> None:
        self._slider_limits[name] = (float(limits[0]), float(limits[1]), False)
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=1)
        lbl = ttk.Label(f, text=f"{self._t(text_key)}:", width=15)
        lbl.pack(side=tk.LEFT)
        self._reg_text(lbl, text_key, suffix=":")
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
        self._add_tip(lbl, name)
        self._add_tip(entry, name)
        self._add_tip(scale, name)

    def _create_stat_row(self, parent: tk.Widget, text_key: str) -> ttk.Label:
        row_frame = ttk.Frame(parent)
        row_frame.pack(fill="x")
        lbl = ttk.Label(row_frame, text=self._t(text_key), font=("Segoe UI", 8, "bold"), width=15, anchor="w")
        lbl.pack(side=tk.LEFT)
        self._reg_text(lbl, text_key)
        value_label = ttk.Label(row_frame, text="--", font=("Segoe UI", 8), anchor="w")
        value_label.pack(side=tk.LEFT)
        return value_label

    def _on_language_selected(self, _event=None) -> None:
        sel = str(self.state.vars["language"].get())
        lang = "ru" if sel == LANGUAGE_LABELS["ru"] or sel.lower().startswith("ру") else "en"
        self.root.set_language(lang, persist=True)

    def apply_language(self, lang: str) -> None:
        """Мгновенно переводит весь интерфейс, подсказки, комбобоксы, вкладки и графики без перезапуска."""
        code = normalize_lang(lang)
        self.state.lang = code
        self.state.vars["language"].set(LANGUAGE_LABELS[code])

        # 1) Обновляем все зарегистрированные текстовые виджеты (Label, Button, LabelFrame, Checkbutton)
        for widget, key, suffix in self._i18n_widgets:
            try:
                widget.configure(text=f"{tr(key, code)}{suffix}")
            except tk.TclError:
                pass

        # 2) Обновляем все всплывающие подсказки ToolTip
        for tip, tip_key in self._tooltips:
            tip.set_text(get_tooltip(tip_key, code, TOOLTIPS))

        # 3) Переводим списки и текущие значения всех Combobox (сохраняя каноническое значение)
        if "preset_combo" in self.widgets:
            canon_pre = to_canonical_preset(str(self.state.vars["preset"].get()))
            self.widgets["preset_combo"].configure(values=get_display_presets(code))
            self.state.vars["preset"].set(to_display_preset(canon_pre, code))

        if "shape_combo" in self.widgets:
            canon_sh = to_canonical_shape(str(self.state.vars["canvas_shape"].get()))
            self.widgets["shape_combo"].configure(values=get_display_shapes(code))
            self.state.vars["canvas_shape"].set(to_display_shape(canon_sh, code))

        if "color_combo" in self.widgets:
            canon_col = to_canonical_color_mode(str(self.state.vars["color_mode"].get()))
            self.widgets["color_combo"].configure(values=get_display_color_modes(code))
            self.state.vars["color_mode"].set(to_display_color_mode(canon_col, code))

        if "struct_combo" in self.widgets:
            canon_met = to_canonical_metric(str(self.state.vars["structural_metric"].get()))
            self.widgets["struct_combo"].configure(values=get_display_metrics(code, SCIPY_AVAILABLE))
            self.state.vars["structural_metric"].set(to_display_metric(canon_met, code))

        # 4) Динамические кнопки и лейблы состояния
        if "pause_button" in self.widgets:
            self.widgets["pause_button"].config(
                text=self._t("btn_resume" if self.state.is_paused else "btn_pause")
            )
        if "auto_play_button" in self.widgets:
            self.widgets["auto_play_button"].config(
                text=self._t("btn_auto_stop" if self.root._auto_play_job else "btn_auto_play")
            )
        if not self.state.is_running and not self.state.layers:
            self.widgets["status_label"].config(text=self._t("status_waiting"))
        if self.state.overall_best_fitness > 0:
            self.update_best_fitness_display(self.state.overall_best_fitness)
        else:
            self.widgets["fitness_label"].config(text=self._t("fitness_idle"))

        mode_str = f"PyTorch CUDA ({CUDA_DEVICE_NAME}) + CSR" if self._last_gpu_flag else "SciPy CSR + NumPy (CPU)"
        self.widgets["compute_mode_label"].config(text=self._t("engine_label", mode=mode_str))

        if self._last_thread_length_m is not None:
            self.update_thread_length_display(self._last_thread_length_m)

        if self.state.original_image is None and "analysis_label" in self.widgets:
            self.widgets["analysis_label"].config(text=self._t("analysis_prompt"))

        # 5) Вкладки Notebook и прогресс-бары островов
        notebook = self.widgets.get("results_notebook")
        if isinstance(notebook, ttk.Notebook):
            tabs = notebook.tabs()
            tab_titles = [
                self._t("tab_overall"),
                self._t("tab_diff_map"),
                self._t("tab_density"),
                self._t("tab_saliency"),
                self._t("tab_radon"),
            ]
            for tab_id, title in zip(tabs, tab_titles):
                notebook.tab(tab_id, text=title)

        for i, lbl in self.main_island_labels.items():
            if not self.state.is_running:
                lbl.config(text=self._t("island_done" if self.state.layers else "island_start", i=i + 1))

        # Update sidebar notebook tab titles
        nb = self.widgets.get("sidebar_notebook")
        if isinstance(nb, ttk.Notebook):
            tab_keys = ["tab_ctrl_image", "tab_ctrl_algo", "tab_ctrl_player", "tab_ctrl_export"]
            for tab_id, key in zip(nb.tabs(), tab_keys):
                nb.tab(tab_id, text=tr(key, code))

        self.update_layer_selector()
        self.update_player_step_label()
        self.update_title()
        self.update_chart()

    def update_layer_selector(self) -> None:
        cb = self.widgets.get("active_layer_combo")
        if not isinstance(cb, ttk.Combobox):
            return
        if not self.state.layers:
            raw_name = "Белая нить" if self.state.is_invert_board() else "Черная нить"
            disp_name = translate_layer_name(raw_name, self.state.lang)
            cb["values"] = [disp_name]
            self.state.vars["active_layer"].set(disp_name)
            return

        unit = self._t("unit_lines_short")
        if len(self.state.layers) > 1:
            total_all = sum(max(0, len(l.get("path", [])) - 1) for l in self.state.layers)
            all_label = self._t("all_colors_option", total=total_all, unit=unit)
            labels = [all_label]
            for i, l in enumerate(self.state.layers):
                c_name = translate_layer_name(str(l.get("name", f"Layer {i+1}")), self.state.lang)
                c_hex = l.get("color_hex", "")
                hex_part = f" [{c_hex}]" if c_hex and c_hex not in ("#111111", "#f5f5f5") else ""
                n_lines = max(0, len(l.get("path", [])) - 1)
                labels.append(f"{i+1}. {c_name}{hex_part} ({n_lines} {unit})")
            cb["values"] = labels
            if self.state.active_layer_idx < 0:
                cb_idx = 0
            else:
                cb_idx = min(self.state.active_layer_idx + 1, len(labels) - 1)
            cb.current(cb_idx)
            self.state.vars["active_layer"].set(labels[cb_idx])
        else:
            labels = [
                f"{translate_layer_name(str(l.get('name', f'Layer {i+1}')), self.state.lang)} ({max(0, len(l.get('path', [])) - 1)} {unit})"
                for i, l in enumerate(self.state.layers)
            ]
            cb["values"] = labels
            self.state.active_layer_idx = 0
            cb.current(0)
            self.state.vars["active_layer"].set(labels[0])

    def _update_adaptive_n_visibility(self, *_) -> None:
        mode = to_canonical_color_mode(str(self.state.safe_get("color_mode") or ""))
        f_spin = getattr(self, "_adaptive_n_frame", None)
        f_pal = getattr(self, "_palette_frame", None)
        is_adapt = "Адаптивный" in mode
        if f_spin:
            if is_adapt:
                f_spin.pack(fill=tk.X, pady=1)
            else:
                f_spin.pack_forget()
        if f_pal:
            if is_adapt:
                f_pal.pack(fill=tk.X, pady=2)
            else:
                f_pal.pack_forget()

    def update_palette(self, colors: List[Dict[str, Any]]) -> None:
        """Отображает визуальные цветные чипы выбранных нитей с названием и HEX-кодом."""
        if not hasattr(self, "_palette_frame"):
            return
        for child in self._palette_frame.winfo_children():
            child.destroy()
        if not colors:
            return
        lbl = ttk.Label(self._palette_frame, text=self._t("palette_title"), font=("Segoe UI", 8, "bold"))
        lbl.pack(anchor="w", pady=(2, 1))
        flow = ttk.Frame(self._palette_frame)
        flow.pack(fill=tk.X, expand=True)
        for i, c in enumerate(colors):
            hex_c = c.get("hex", "#333333")
            name = c.get("name", f"Цвет {i+1}")
            rgb = c.get("rgb", [50, 50, 50])
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            fg = "#111111" if lum > 145 else "#ffffff"
            chip = tk.Label(
                flow,
                text=f" {i+1}. {name} {hex_c} ",
                bg=hex_c,
                fg=fg,
                font=("Segoe UI", 8, "bold"),
                relief="ridge",
                bd=1,
                padx=3,
                pady=1,
                cursor="hand2",
            )
            chip.pack(side=tk.LEFT, padx=1, pady=1)
            chip.bind("<Button-1>", lambda _e, idx=i: self._select_layer_by_index(idx))

    def _select_layer_by_index(self, idx: int) -> None:
        cb = self.widgets.get("active_layer_combo")
        if isinstance(cb, ttk.Combobox) and 0 <= idx < len(self.state.layers):
            target_cb_idx = idx + 1 if len(self.state.layers) > 1 else idx
            if target_cb_idx < len(cb["values"]):
                cb.current(target_cb_idx)
                self._on_active_layer_changed()

    def _on_active_layer_changed(self, _event=None) -> None:
        if not self.state.layers:
            return
        cb = self.widgets.get("active_layer_combo")
        cur_idx = cb.current() if isinstance(cb, ttk.Combobox) else -1

        if len(self.state.layers) > 1:
            if cur_idx == 0:
                # Режим «🌈 Все цвета»: отображаем все цвета одновременно!
                self.state.active_layer_idx = -1
                total_all = sum(max(0, len(l.get("path", [])) - 1) for l in self.state.layers)
                scale = self.widgets.get("step_scrubber_scale")
                if isinstance(scale, ttk.Scale):
                    scale.configure(to=max(1, total_all))
                self.state.vars["step_scrubber"].set(max(1, total_all))
                self.update_player_step_label()
                self._on_visual_param_change()
                return
            elif cur_idx > 0:
                layer_idx = cur_idx - 1
            else:
                layer_idx = 0
        else:
            layer_idx = max(0, cur_idx)

        if 0 <= layer_idx < len(self.state.layers):
            self.state.active_layer_idx = layer_idx
            path = self.state.layers[layer_idx].get("path", [])
            max_steps = max(1, len(path) - 1)
            self.state.overall_best_chromosome = path
            scale = self.widgets.get("step_scrubber_scale")
            if isinstance(scale, ttk.Scale):
                scale.configure(to=max(1, max_steps))
            self.state.vars["step_scrubber"].set(max_steps)
            self.update_player_step_label()
            self._on_visual_param_change()

    def _on_scrubber_slider_move(self, *_args) -> None:
        self.update_player_step_label()
        self._on_visual_param_change()

    def update_player_step_label(self) -> None:
        lbl = self.widgets.get("player_step_label")
        if not lbl:
            return
        if not self.state.layers:
            lbl.config(text=self._t("player_idle"))
            return
        if self.state.active_layer_idx < 0:
            total_all = sum(max(0, len(l.get("path", [])) - 1) for l in self.state.layers)
            step = max(1, min(total_all, int(self.state.safe_get("step_scrubber") or total_all)))
            lbl.config(text=self._t("all_colors_step", step=step, total=total_all))
            return
        if self.state.active_layer_idx >= len(self.state.layers):
            lbl.config(text=self._t("player_idle"))
            return
        path = self.state.layers[self.state.active_layer_idx].get("path", [])
        if len(path) < 2:
            lbl.config(text=self._t("player_empty"))
            return
        total = len(path) - 1
        step = max(1, min(total, int(self.state.safe_get("step_scrubber"))))
        lbl.config(text=self._t("player_step", step=step, total=total, p1=path[step - 1], p2=path[step]))

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

        self._create_notebook_tab("overall", self._t("tab_overall"), notebook)

        self._create_analysis_canvas_tab(notebook, "diff_map", self._t("tab_diff_map"))
        self._create_analysis_canvas_tab(notebook, "density_heatmap", self._t("tab_density"))
        self._create_analysis_canvas_tab(notebook, "fit_saliency", self._t("tab_saliency"))
        self._create_analysis_canvas_tab(notebook, "radon_sinogram", self._t("tab_radon"))

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
            label = ttk.Label(frame, text=self._t("island_start", i=i + 1), width=19, font=("Segoe UI", 8))
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
                self.ax_fitness.plot(gens, fitnesses, label=self._t("tab_island", i=i + 1), color=colors[i], linewidth=1.5)
                has_lines = True
        if self.state.omp_baseline_fitness > 0:
            self.ax_fitness.axhline(self.state.omp_baseline_fitness, color="gray", linestyle=":", label=self._t("chart_omp_base"))
            has_lines = True

        self.ax_fitness.set_title(self._t("chart_fit_title"), fontsize=9)
        self.ax_fitness.set_xlabel(self._t("chart_fit_xlabel"), fontsize=8)
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
        self.ax_mut.set_title(self._t("chart_mut_title"), fontsize=9)
        self.ax_mut.tick_params(labelsize=8)
        self.ax_mut.grid(True, axis="x", linestyle="--", alpha=0.6, color="#e1e4e8")

        self.chart_canvas.figure.tight_layout(pad=1.0)
        self.chart_canvas.draw()
        self.chart_needs_update = False

    def _set_widget_tree_state(self, root_widget: Optional[tk.Widget], state: str) -> None:
        if root_widget is None:
            return
        for child in root_widget.winfo_children():
            if child is self.widgets.get("language_combo"):
                continue
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
        if "load_project_button" in self.widgets:
            self.widgets["load_project_button"].config(state=state)
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
            "save_project_button",
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
            self.widgets["pause_button"].config(text=self._t("btn_pause"))
            for i, pb in self.main_island_progress_bars.items():
                pb["value"] = 0
                if i in self.main_island_labels:
                    self.main_island_labels[i].config(text=self._t("island_done", i=i + 1))

    def update_island_progress(self, island_id: int, stage: int, patience: int, limit: int) -> None:
        if island_id in self.main_island_labels:
            pct = int((patience / max(1, limit)) * 100)
            total_st = max(1, int(self.state.safe_get("stages")))
            self.main_island_labels[island_id].config(
                text=self._t("island_prog", i=island_id + 1, st=stage + 1, tot=total_st, pct=pct)
            )
        if island_id in self.main_island_progress_bars:
            pb = self.main_island_progress_bars[island_id]
            pb["maximum"] = max(1, limit)
            pb["value"] = patience

    def update_best_fitness_display(self, fitness: float) -> None:
        n_lines = sum(max(0, len(l.get("path", [])) - 1) for l in self.state.layers)
        if n_lines == 0:
            n_lines = max(0, len(self.state.overall_best_chromosome or []) - 1)
        self.widgets["fitness_label"].config(text=self._t("fitness_val", fitness=fitness, lines=n_lines))

    def update_thread_length_display(self, length_in_meters: float) -> None:
        self._last_thread_length_m = float(length_in_meters)
        dia = float(self.state.safe_get("physical_diameter") or 50.0)
        self.widgets["len_label"].config(text=self._t("len_fmt", m=length_in_meters, dia=dia))

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
            self.widgets["eta_label"].config(text=self._t("eta_calc"))

    def update_canvas_image(self, key: Any, img_pil: Image.Image) -> None:
        self.raw_pil_images[key] = img_pil
        canvas = self.result_canvases.get(key)
        if not canvas:
            return
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 10 or h < 10:
            return

        if key == "radon_sinogram":
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
                    canon_cm = to_canonical_color_mode(str(self.state.safe_get("color_mode")))
                    is_color = "CMYK" in canon_cm or "Адаптивный" in canon_cm
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


class StringArtApp(tk.Tk):
    """Главное окно String Art Studio Pro (автоопределение RU / EN по языку системы)."""

    def __init__(self, lang: Optional[str] = None):
        enable_windows_high_dpi()
        super().__init__()
        self.geometry("1540x940")
        self.minsize(1120, 740)

        self._load_window_icon()

        self.state = AppState(lang=lang)
        self.queues = {"status_queue": queue.Queue()}
        self.audio_assistant = AudioAssistant(lang=self.state.lang)
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

    def _t(self, key: str, **kwargs) -> str:
        return tr(key, self.state.lang, **kwargs)

    def set_language(self, lang: str, persist: bool = True) -> None:
        """Переключает язык интерфейса на лету ('ru' или 'en') без перезапуска."""
        code = normalize_lang(lang)
        self.state.lang = code
        self.audio_assistant.set_language(code)
        if persist:
            save_language_preference(code)
        self.ui_manager.apply_language(code)

    def _load_window_icon(self) -> None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(base_dir, "StringArt.ico"),
            os.path.join(os.getcwd(), "StringArt.ico"),
        ]
        meipass = getattr(__import__("sys"), "_MEIPASS", None)
        if meipass:
            candidates.insert(0, os.path.join(meipass, "StringArt.ico"))
        for cand in candidates:
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
        self.bind("<Control-s>", lambda _e: self.coordinator.save_project())

    def on_closing(self) -> None:
        if self.state.is_running and not messagebox.askokcancel(self._t("dlg_exit_title"), self._t("dlg_exit_msg")):
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
        disp_name = str(self.state.vars["preset"].get())
        canon_name = to_canonical_preset(disp_name)
        preset = PRESETS.get(canon_name)
        if preset is None:
            self.apply_auto_recommendations()
            return
        lang = self.state.lang
        for k, v in preset.items():
            if k in self.state.vars:
                if k == "canvas_shape":
                    v = to_display_shape(str(v), lang)
                elif k == "color_mode":
                    v = to_display_color_mode(str(v), lang)
                self.state.vars[k].set(v)
        self.ui_manager.widgets["analysis_label"].config(
            text=self._t("analysis_preset", name=to_display_preset(canon_name, lang)),
            foreground="#0969da",
        )
        self.redraw_target_canvas()

    def on_voice_toggle(self) -> None:
        self.audio_assistant.toggle(bool(self.state.safe_get("voice_enabled")))

    def step_player(self, delta: int) -> None:
        if not self.state.layers:
            return
        if self.state.active_layer_idx < 0:
            total_s = sum(max(0, len(l.get("path", [])) - 1) for l in self.state.layers)
            cur = int(self.state.safe_get("step_scrubber") or total_s)
            new_s = max(1, min(total_s, cur + delta))
            if new_s != cur:
                self.state.vars["step_scrubber"].set(new_s)
                self.ui_manager.update_player_step_label()
                self.ui_manager._on_visual_param_change()
            return
        if self.state.active_layer_idx >= len(self.state.layers):
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
                btn.config(text=self._t("btn_auto_play"))
        else:
            if not self.state.layers:
                return
            if btn:
                btn.config(text=self._t("btn_auto_stop"))
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
            self.ui_manager.widgets["status_label"].config(text=self._t("msg_loaded_ok"))
        except Exception as exc:
            logging.error("Не удалось загрузить изображение", exc_info=True)
            messagebox.showerror(self._t("dlg_error"), self._t("msg_load_err", err=exc))

    def apply_auto_recommendations(self) -> None:
        if not self.state.original_image:
            return
        shape = to_canonical_shape(str(self.state.safe_get("canvas_shape") or "Круг"))
        rec = auto_recommend_parameters(self.state.original_image, shape, lang=self.state.lang)
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
        canvas = self.ui_manager.widgets["target_canvas"]
        invert_board = self.state.is_invert_board()
        canvas.configure(bg="#121418" if invert_board else "#ffffff")
        if not self.state.original_image:
            return
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 20 or h < 20:
            return
        display_size = max(60, min(w, h) - 10)
        color_mode = to_canonical_color_mode(str(self.state.safe_get("color_mode") or "Монохром (Черная нить)"))
        shape = to_canonical_shape(str(self.state.safe_get("canvas_shape") or "Круг"))
        bundle = prepare_target_image_bundle(
            self.state.original_image,
            display_size,
            clahe_clip=float(self.state.safe_get("clahe_clip") or 1.3),
            gamma=float(self.state.safe_get("gamma") or 0.82),
            shape=shape,
            color_mode=color_mode,
        )
        self.state.preprocessed_target_np = bundle["gray"]
        self.state.preprocessed_rgb_np = bundle["rgb"]
        is_color = "CMYK" in color_mode or "Адаптивный" in color_mode
        if is_color:
            img_pil = Image.fromarray(bundle["rgb"])
        elif bundle.get("invert_board"):
            preview_gray = np.where(
                bundle["mask"] == 255,
                np.clip(18.0 + (bundle["gray"].astype(np.float32) / 255.0) * 237.0, 18.0, 255.0).astype(np.uint8),
                18,
            )
            self.state.preprocessed_target_np = preview_gray
            img_pil = Image.fromarray(preview_gray)
        else:
            img_pil = Image.fromarray(bundle["gray"])
        self.state.target_image_tk = ImageTk.PhotoImage(img_pil)
        canvas.delete("all")
        canvas.create_image(w // 2, h // 2, anchor=tk.CENTER, image=self.state.target_image_tk)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(threadName)s - %(levelname)s - %(message)s")
    app = StringArtApp()
    app.mainloop()
