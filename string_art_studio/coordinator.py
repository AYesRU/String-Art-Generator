# -*- coding: utf-8 -*-
"""
Состояние приложения (AppState) и потокобезопасный координатор синтеза,
рендеринга, сохранения проектов и экспорта (EvolutionCoordinator).
"""

import logging
import os
import queue
import threading
import time
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from . import IMAGE_SIZE
from .audio import AudioAssistant
from .engine import (
    CUDA_DEVICE_NAME,
    SCIPY_AVAILABLE,
    FitnessCalculator,
    GreedyInitializer,
    IslandManagerThread,
    estimate_channel_line_budget,
    optimize_thread_count_delta,
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
    load_project_json,
    save_project_json,
)
from .geometry import CanvasManager
from .i18n import (
    LANGUAGE_LABELS,
    detect_system_language,
    layer_short_code,
    normalize_lang,
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
from .preprocessing import _compute_channel_edges_and_importance, prepare_target_image_bundle
from .radon import RadonGuide
from .color_catalog import COLOR_CATALOG

if TYPE_CHECKING:
    from .ui import StringArtApp, UIManager


class AppState:
    """Хранилище параметров и текущего состояния синтеза/плеера (RU / EN)."""

    def __init__(self, lang: Optional[str] = None):
        self.lang: str = normalize_lang(lang) if lang else detect_system_language()
        default_metric = "Преобразование расстояний" if SCIPY_AVAILABLE else "Контуры"
        self.vars: Dict[str, tk.Variable] = {
            "language": tk.StringVar(value=LANGUAGE_LABELS[self.lang]),
            "preset": tk.StringVar(value=to_display_preset("Авто (по изображению)", self.lang)),
            "clahe_clip": tk.DoubleVar(value=1.3),
            "gamma": tk.DoubleVar(value=0.82),
            "canvas_shape": tk.StringVar(value=to_display_shape("Круг", self.lang)),
            "color_mode": tk.StringVar(value=to_display_color_mode("Монохром (Черная нить)", self.lang)),
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
            "structural_metric": tk.StringVar(value=to_display_metric(default_metric, self.lang)),
            "contour_weight": tk.IntVar(value=55),
            "memetic_enabled": tk.BooleanVar(value=True),
            "memetic_intensity": tk.IntVar(value=20),
            "memetic_depth": tk.IntVar(value=25),
            "radon_enabled": tk.BooleanVar(value=True),
            "radon_influence": tk.IntVar(value=35),
            "step_scrubber": tk.IntVar(value=1400),
            "active_layer": tk.StringVar(value=translate_layer_name("Черная нить", self.lang)),
            "show_active_chord": tk.BooleanVar(value=True),
            "voice_enabled": tk.BooleanVar(value=False),
            "split_compare": tk.IntVar(value=0),
            "adaptive_n_colors": tk.IntVar(value=5),
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
        cfg = {k: self.safe_get(k) for k in self.vars}
        # Канонические значения для движка/экспорта/совместимости .sart.json между RU и EN
        cfg["preset"] = to_canonical_preset(str(cfg.get("preset", "")))
        cfg["canvas_shape"] = to_canonical_shape(str(cfg.get("canvas_shape", "Круг")))
        cfg["color_mode"] = to_canonical_color_mode(str(cfg.get("color_mode", "Монохром (Черная нить)")))
        cfg["structural_metric"] = to_canonical_metric(str(cfg.get("structural_metric", "Контуры")))
        cfg["language"] = self.lang
        return cfg

    def is_invert_board(self) -> bool:
        mode = to_canonical_color_mode(str(self.safe_get("color_mode") or ""))
        return "Белая нить" in mode or "Темная" in mode

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


class EvolutionCoordinator:
    """Координирует подготовку данных, многопоточный синтез, рендеринг, проекты и экспорт."""

    def __init__(
        self,
        root: "StringArtApp",
        state: AppState,
        ui_manager: "UIManager",
        queues: Dict[str, queue.Queue],
        audio_assistant: AudioAssistant,
    ):
        self.root = root
        self.state = state
        self.ui_manager = ui_manager
        self.queues = queues
        self.audio_assistant = audio_assistant
        self.manager_thread: Optional[IslandManagerThread] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._run_id: int = 0
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
        if self.canvas_manager is not None:
            self.canvas_manager.release_gpu_cache()

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
                invert_board = bool(render_params.get("invert_board", False))

                if isinstance(layers_or_chromo, list) and layers_or_chromo and isinstance(layers_or_chromo[0], dict):
                    layers_data = layers_or_chromo
                else:
                    layers_data = [
                        {
                            "name": "Белая нить" if invert_board else "Черная нить",
                            "color_hex": "#f5f5f5" if invert_board else "#111111",
                            "path": list(layers_or_chromo),
                        }
                    ]

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
                    invert_board=invert_board,
                )
                self.render_results_queue.put((key, img_pil))

                # Асинхронно обновляем карты ошибок/плотности без торможения UI-потока
                now = time.perf_counter()
                if (
                    key == "overall"
                    and self.active_fitness_calc is not None
                    and (now - self._last_analytic_map_time > 0.45 or render_params.get("force_maps"))
                ):
                    self._last_analytic_map_time = now
                    _ali = active_layer_idx if active_layer_idx < len(layers_data) else 0
                    path_main = layers_data[_ali].get("path", []) if layers_data else []
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

    def _next_run_id(self) -> int:
        self._run_id += 1
        return self._run_id

    def _t(self, key: str, **kwargs) -> str:
        return tr(key, self.state.lang, **kwargs)

    def start_evolution_only(self) -> None:
        """Запуск параллельной островной эволюции на текущем пути без перезапуска OMP."""
        if self.state.original_image is None:
            messagebox.showwarning(self._t("dlg_warn"), self._t("msg_load_first"))
            return
        if not self.state.overall_best_chromosome or len(self.state.overall_best_chromosome) < 4:
            messagebox.showinfo(self._t("dlg_info"), self._t("msg_no_base_path"))
            self.start_generation()
            return

        config = self.state.get_config()
        run_id = self._next_run_id()
        self._cancel_stats_job()
        self.omp_stop_flag = False
        self.omp_pause_event.set()
        self.state.reset_for_new_run(reset_layers=False)
        self.ui_manager.toggle_ui_state(True)
        self.best_chromosomes_by_key.clear()
        self.ui_manager.setup_for_new_run(config["num_islands"])
        self.stats_update_job = self.root.after(1000, self._update_progress_stats)
        t = threading.Thread(
            target=self._run_islands_only_worker, args=(config, run_id), name="IslandsOnlyWorker", daemon=True
        )
        self._worker_thread = t
        t.start()

    def _run_islands_only_worker(self, config: Dict[str, Any], run_id: int) -> None:
        lang = config.get("language", self.state.lang)
        try:
            self._send_status(tr("st_prep_fc", lang, n=config["num_islands"]), run_id=run_id)
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
            ch_edges = ch_data.get("edges", bundle["edges"])
            is_cmyk = len(bundle["channels"]) > 1
            layer_max_lines = (
                estimate_channel_line_budget(
                    ch_data["darkness"], bundle["mask"], cm, config["lines"], config["line_weight"]
                )
                if is_cmyk
                else int(config["lines"])
            )

            if self.radon_guide is None and config.get("radon_enabled", True) and layer_max_lines > 0:
                self.radon_guide = RadonGuide(
                    bundle["gray"],
                    ch_edges,
                    cm,
                    config,
                    compute_fbp=False,
                    density_override=ch_data["darkness"],
                )

            fc = FitnessCalculator(
                bundle["gray"],
                ch_edges,
                bundle["saliency"],
                ch_data["importance"],
                cm,
                config,
                self.queues,
                target_darkness_override=ch_data["darkness"],
            )
            self.active_fitness_calc = fc
            base_chromosome = list(self.state.overall_best_chromosome or [])
            disp_name = translate_layer_name(ch_data["name"], lang)
            if layer_max_lines == 0:
                self.queues["status_queue"].put({
                    "type": "island_update",
                    "run_id": run_id,
                    "data": {"id": 0, "chromosome": [], "fitness": fc.empty_fitness, "mut_stats": {}, "layer_idx": active_idx},
                })
                self.queues["status_queue"].put({
                    "type": "done",
                    "run_id": run_id,
                    "data": tr("st_layer_no_threads", lang, name=disp_name),
                })
                return

            if len(base_chromosome) < 2:
                base_chromosome, _, _, _ = optimize_thread_count_delta(
                    [], fc, layer_max_lines, radon_guide=self.radon_guide, allow_add=True
                )
                if len(base_chromosome) < 2:
                    self.queues["status_queue"].put({
                        "type": "done",
                        "run_id": run_id,
                        "data": tr("st_layer_opt_empty", lang, name=disp_name),
                    })
                    return

            self.state.omp_baseline_fitness = fc.calculate_single_fitness(base_chromosome)
            self._send_status(
                tr("st_start_islands", lang, n=config["num_islands"], max_l=layer_max_lines),
                run_id=run_id,
            )
            self.manager_thread = IslandManagerThread(
                config,
                self.queues,
                fc,
                base_chromosome,
                self.radon_guide,
                layer_idx=active_idx,
                layer_name=ch_data["name"],
                emit_done=True,
                run_id=run_id,
                max_lines=layer_max_lines,
            )
            self.manager_thread.start()
        except Exception:
            logging.error("Ошибка при запуске эволюционной доводки", exc_info=True)
            self.queues["status_queue"].put({
                "type": "error",
                "run_id": run_id,
                "data": f"{tr('dlg_error', lang)}: {traceback.format_exc()}",
            })

    def _ensure_canvas_manager(self, config: Dict[str, Any]) -> CanvasManager:
        cm = self.canvas_manager
        if (
            cm is None
            or cm.num_pins != config["pins"]
            or cm.shape != config["canvas_shape"]
            or abs(cm.physical_cm - config["physical_diameter"]) > 1e-3
        ):
            if cm is not None:
                cm.release_gpu_cache()
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
            messagebox.showwarning(self._t("dlg_warn"), self._t("msg_load_first"))
            return
        config = self.state.get_config()
        run_id = self._next_run_id()
        self._cancel_stats_job()
        self.omp_stop_flag = False
        self.omp_pause_event.set()
        self.state.reset_for_new_run(reset_layers=True)
        self.ui_manager.toggle_ui_state(True)
        self.best_chromosomes_by_key.clear()
        self.ui_manager.setup_for_new_run(config["num_islands"])
        self.stats_update_job = self.root.after(1000, self._update_progress_stats)
        t = threading.Thread(
            target=self._prepare_and_run_evolution, args=(config, run_id), name="PreparationThread", daemon=True
        )
        self._worker_thread = t
        t.start()

    def _prepare_and_run_evolution(self, config: Dict[str, Any], run_id: int) -> None:
        lang = config.get("language", self.state.lang)
        try:
            self._send_status(tr("st_analyze", lang), run_id=run_id)
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
            channels = bundle["channels"]
            is_cmyk = len(channels) > 1
            is_adaptive = "Адаптивный" in config.get("color_mode", "")
            radon_enabled = bool(config.get("radon_enabled", True))

            radon_guide = None
            self.radon_guide = None
            if radon_enabled:
                self._send_status(tr("st_radon", lang), run_id=run_id)
                primary_density = None if is_cmyk else channels[0]["darkness"]
                radon_guide = RadonGuide(
                    bundle["gray"],
                    bundle["edges"],
                    cm,
                    config,
                    compute_fbp=True,
                    density_override=primary_density,
                )
                self.radon_guide = radon_guide
                if radon_guide.sinogram is not None:
                    self.queues["status_queue"].put({
                        "type": "radon_map",
                        "run_id": run_id,
                        "data": (radon_guide.sinogram, radon_guide.fbp_recon),
                    })

            self.queues["status_queue"].put({
                "type": "fit_func_maps",
                "run_id": run_id,
                "data": {"saliency": bundle["saliency"]},
            })

            accumulated_layers: List[Dict[str, Any]] = []

            if not is_cmyk and not is_adaptive:
                ch = channels[0]
                ch_edges = ch.get("edges", bundle["edges"])
                max_lines = int(config["lines"])
                disp_ch = translate_layer_name(ch["name"], lang)
                fc = FitnessCalculator(
                    bundle["gray"],
                    ch_edges,
                    bundle["saliency"],
                    ch["importance"],
                    cm,
                    config,
                    self.queues,
                    target_darkness_override=ch["darkness"],
                )
                self.active_fitness_calc = fc
                self._send_status(tr("st_omp_start", lang, name=disp_ch, max_l=max_lines), run_id=run_id)
                greedy = GreedyInitializer(
                    cm,
                    fc.target_darkness_flat,
                    fc.importance_flat,
                    config,
                    radon_guide,
                    stop_Check=self._check_omp_stop_or_pause,
                )

                def greedy_cb(status: str, path: List[int]) -> None:
                    layers_snap = [{"name": ch["name"], "color_hex": ch["color_hex"], "path": path, "ref_lines": max_lines}]
                    self.queues["status_queue"].put({
                        "type": "greedy_update",
                        "run_id": run_id,
                        "data": {"text": status, "layers": layers_snap, "active_idx": 0},
                    })

                base_chromosome = greedy.run(max_lines, greedy_cb, layer_label=ch["name"], line_alpha=fc.line_alpha)
                if not base_chromosome or len(base_chromosome) < 2:
                    self.state.layers = [{"name": ch["name"], "color_hex": ch["color_hex"], "path": [], "ref_lines": max_lines}]
                    self.queues["status_queue"].put({
                        "type": "done",
                        "run_id": run_id,
                        "data": tr("st_no_signal", lang),
                    })
                    return

                self.state.layers = [
                    {"name": ch["name"], "color_hex": ch["color_hex"], "path": list(base_chromosome), "ref_lines": max_lines}
                ]
                omp_fit, diff_map, depth_2d = fc.calculate_single_fitness_and_maps(base_chromosome)
                self.state.omp_baseline_fitness = omp_fit
                self.state.layer_best_fitnesses[0] = omp_fit - 1e-6
                self.queues["status_queue"].put({
                    "type": "island_update",
                    "run_id": run_id,
                    "data": {"id": 0, "chromosome": base_chromosome, "fitness": omp_fit, "mut_stats": {}, "layer_idx": 0},
                })
                self._push_analytic_maps(diff_map, depth_2d)

                if self.omp_stop_flag:
                    self.queues["status_queue"].put({"type": "done", "run_id": run_id, "data": tr("st_stopped_omp", lang)})
                    return

                self.manager_thread = IslandManagerThread(
                    config,
                    self.queues,
                    fc,
                    base_chromosome,
                    radon_guide,
                    layer_idx=0,
                    layer_name=ch["name"],
                    emit_done=True,
                    run_id=run_id,
                    max_lines=max_lines,
                )
                self.manager_thread.start()
            elif is_cmyk:
                # 4-слойный CMYK синтез с индивидуальным первоначальным расчетом бюджета нитей на каждый цвет
                max_lines = int(config["lines"])
                ch_budgets = [
                    estimate_channel_line_budget(ch["darkness"], bundle["mask"], cm, max_lines, config["line_weight"])
                    for ch in channels
                ]
                budget_info = ", ".join(f"{layer_short_code(c['name'], lang)}≤{b}" for c, b in zip(channels, ch_budgets))
                logging.info(f"Индивидуальные бюджеты CMYK (макс. {max_lines}): {budget_info}")
                self._send_status(tr("st_cmyk_budgets", lang, info=budget_info), run_id=run_id)

                for c_idx, ch in enumerate(channels):
                    if self.omp_stop_flag:
                        break
                    disp_ch = translate_layer_name(ch["name"], lang)
                    ch_budget = ch_budgets[c_idx]
                    if ch_budget == 0:
                        logging.info(f"CMYK слой {c_idx + 1}/4 ({ch['name']}): цвет отсутствует — 0 нитей (пропуск).")
                        accumulated_layers.append({
                            "name": ch["name"],
                            "color_hex": ch["color_hex"],
                            "path": [],
                            "ref_lines": max_lines,
                        })
                        self.queues["status_queue"].put({
                            "type": "greedy_update",
                            "run_id": run_id,
                            "data": {
                                "text": tr("st_cmyk_absent", lang, idx=c_idx + 1, name=disp_ch),
                                "layers": list(accumulated_layers),
                                "active_idx": c_idx,
                            },
                        })
                        continue

                    ch_edges = ch.get("edges", bundle["edges"])
                    ch_radon = (
                        RadonGuide(
                            bundle["gray"],
                            ch_edges,
                            cm,
                            config,
                            compute_fbp=False,
                            density_override=ch["darkness"],
                        )
                        if radon_enabled
                        else None
                    )
                    fc = FitnessCalculator(
                        bundle["gray"],
                        ch_edges,
                        bundle["saliency"],
                        ch["importance"],
                        cm,
                        config,
                        self.queues,
                        target_darkness_override=ch["darkness"],
                    )
                    self.active_fitness_calc = fc
                    self._send_status(tr("st_cmyk_omp", lang, idx=c_idx + 1, name=disp_ch, bud=ch_budget), run_id=run_id)
                    greedy = GreedyInitializer(
                        cm,
                        fc.target_darkness_flat,
                        fc.importance_flat,
                        config,
                        ch_radon,
                        stop_Check=self._check_omp_stop_or_pause,
                    )

                    def greedy_cb_cmyk(status: str, path: List[int], idx=c_idx, ch_meta=ch, d_name=disp_ch) -> None:
                        snap = list(accumulated_layers)
                        snap.append({
                            "name": ch_meta["name"],
                            "color_hex": ch_meta["color_hex"],
                            "path": path,
                            "ref_lines": max_lines,
                        })
                        self.queues["status_queue"].put({
                            "type": "greedy_update",
                            "run_id": run_id,
                            "data": {"text": tr("st_cmyk_layer_prog", lang, idx=idx + 1, name=d_name, status=status), "layers": snap, "active_idx": idx},
                        })

                    base_path = greedy.run(ch_budget, greedy_cb_cmyk, layer_label=ch["name"], line_alpha=fc.line_alpha)
                    if len(base_path) >= 2 and not self.omp_stop_flag:
                        omp_layer_fit = fc.calculate_single_fitness(base_path)
                        self.state.omp_baseline_fitness = omp_layer_fit
                        self.state.layer_best_fitnesses[c_idx] = omp_layer_fit - 1e-6
                        snap_pre = list(accumulated_layers) + [
                            {"name": ch["name"], "color_hex": ch["color_hex"], "path": list(base_path), "ref_lines": max_lines}
                        ]
                        self.queues["status_queue"].put({
                            "type": "greedy_update",
                            "run_id": run_id,
                            "data": {
                                "text": tr("st_cmyk_polish", lang, idx=c_idx + 1, name=disp_ch, lines=len(base_path) - 1),
                                "layers": snap_pre,
                                "active_idx": c_idx,
                            },
                        })
                        cmyk_stages = max(2, config.get("stages", 5) // 2)
                        mgr = IslandManagerThread(
                            config,
                            self.queues,
                            fc,
                            base_path,
                            ch_radon,
                            layer_idx=c_idx,
                            layer_name=ch["name"],
                            emit_done=False,
                            stages_override=cmyk_stages,
                            patience_scale=0.55,
                            run_id=run_id,
                            max_lines=ch_budget,
                        )
                        self.manager_thread = mgr
                        mgr.start()
                        mgr.join()
                        if mgr.best_result is not None:
                            base_path = list(mgr.best_result.get("chromosome", []))

                    accumulated_layers.append({
                        "name": ch["name"],
                        "color_hex": ch["color_hex"],
                        "path": base_path,
                        "ref_lines": max_lines,
                    })
                    n_done = max(0, len(base_path) - 1)
                    self.queues["status_queue"].put({
                        "type": "greedy_update",
                        "run_id": run_id,
                        "data": {"text": tr("st_cmyk_layer_done", lang, name=disp_ch, lines=n_done), "layers": list(accumulated_layers), "active_idx": c_idx},
                    })

                msg = tr("st_stopped_user", lang) if self.omp_stop_flag else tr("st_cmyk_done", lang)
                self.queues["status_queue"].put({"type": "done", "run_id": run_id, "data": msg})

            elif is_adaptive:
                n_colors = max(2, min(20, int(config.get("adaptive_n_colors", 5))))
                self._send_status(tr("st_adaptive_selecting", lang, n=n_colors, total=len(COLOR_CATALOG)), run_id=run_id)
                rgb_img = self.state.preprocessed_rgb_np if self.state.preprocessed_rgb_np is not None else np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
                mask = bundle.get("mask", np.ones_like(bundle["gray"], dtype=np.uint8) * 255)
                invert_board = bool(bundle.get("invert_board", False))
                selected_colors = self._select_adaptive_colors(rgb_img, n_colors, mask=mask, invert_board=invert_board)
                max_lines = int(config["lines"])

                # Обновляем визуальную палитру нитей в интерфейсе
                self.queues["status_queue"].put({"type": "palette_update", "run_id": run_id, "data": selected_colors})

                # Предобработка исходного изображения под цветоделение (сглаживание шума и гамма)
                rgb_smooth = cv2.bilateralFilter(rgb_img, d=5, sigmaColor=18, sigmaSpace=18)
                gamma_val = float(config.get("gamma", 1.0))
                if abs(gamma_val - 1.0) > 0.03:
                    inv_gamma = 1.0 / max(0.1, gamma_val) if not invert_board else gamma_val
                    table = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)], dtype=np.uint8)
                    rgb_smooth = cv2.LUT(rgb_smooth, table)

                rgb_norm = np.clip(rgb_smooth.astype(np.float32) / 255.0, 0.0, 1.0)
                h, w, _ = rgb_norm.shape

                # Физическое оптическое цветоделение в пространстве CIE Lab
                if not invert_board:
                    gray_norm = 0.299 * rgb_norm[:, :, 0] + 0.587 * rgb_norm[:, :, 1] + 0.114 * rgb_norm[:, :, 2]
                    darkness_total = np.clip(1.0 - gray_norm, 0.0, 1.0)
                    A_cols = [np.clip(1.0 - (np.array(c["rgb"], dtype=np.float32) / 255.0), 0.0, 1.0) for c in selected_colors]
                    contrasts = [float(np.mean(a)) for a in A_cols]
                else:
                    gray_norm = 0.299 * rgb_norm[:, :, 0] + 0.587 * rgb_norm[:, :, 1] + 0.114 * rgb_norm[:, :, 2]
                    darkness_total = np.clip(gray_norm, 0.0, 1.0)
                    A_cols = [np.clip(np.array(c["rgb"], dtype=np.float32) / 255.0, 0.0, 1.0) for c in selected_colors]
                    contrasts = [float(np.mean(a)) for a in A_cols]

                darkness_total = np.where(mask == 255, darkness_total, 0.0)

                # CIE Lab мягкое разбиение: сумма плотностей по слоям в каждом пикселе не превышает исходную темноту
                img_lab = self._rgb_to_lab(rgb_norm)
                cat_rgb = np.array([[c["rgb"][0], c["rgb"][1], c["rgb"][2]] for c in selected_colors], dtype=np.float32) / 255.0
                cat_lab = self._rgb_to_lab(cat_rgb)

                diff_lab = img_lab[:, :, np.newaxis, :] - cat_lab[np.newaxis, np.newaxis, :, :]
                dists_lab = np.sqrt(np.sum(diff_lab ** 2, axis=-1))
                shift_lab = np.min(dists_lab, axis=-1, keepdims=True)
                exp_d = np.exp(-(dists_lab - shift_lab) / 14.0)
                weights = exp_d / (np.sum(exp_d, axis=-1, keepdims=True) + 1e-6)

                channel_densities = []
                channel_masses = []
                for k_i in range(len(selected_colors)):
                    c_density = np.clip(darkness_total * weights[:, :, k_i], 0.0, 1.0).astype(np.float32)
                    c_density[mask == 0] = 0.0
                    c_density[c_density < 0.03] = 0.0
                    channel_densities.append(c_density)
                    m_val = float(np.sum(-np.log(1.0 - np.clip(c_density[mask == 255], 0.0, 0.95)) / 1.6))
                    channel_masses.append(m_val)

                total_mass = sum(channel_masses)

                for c_idx, sc in enumerate(selected_colors):
                    if self.omp_stop_flag:
                        break
                    name_ru = sc["name"]
                    disp_name = translate_layer_name(name_ru, lang)
                    hex_c = sc["hex"]
                    contrast_c = contrasts[c_idx]
                    self._send_status(tr("st_adaptive_layer", lang, idx=c_idx + 1, n=len(selected_colors), name=disp_name), run_id=run_id)

                    adapt_darkness = channel_densities[c_idx]
                    adapt_edges, adapt_imp = _compute_channel_edges_and_importance(adapt_darkness, mask, boost=1.1)

                    if total_mass > 1e-4:
                        prop_budget = int(round(max_lines * (channel_masses[c_idx] / total_mass)))
                    else:
                        prop_budget = max_lines // len(selected_colors)

                    phys_budget = estimate_channel_line_budget(adapt_darkness, mask, cm, max_lines, config["line_weight"])
                    ch_budget = min(prop_budget, phys_budget)

                    if ch_budget < 25:
                        if np.sum(adapt_darkness > 0.06) < 80:
                            logging.info(f"Слой '{disp_name}' не имеет достаточного бюджета ({ch_budget} нитей) — пропуск.")
                            continue
                        ch_budget = 35

                    fc = FitnessCalculator(
                        bundle["gray"], adapt_edges, bundle["saliency"], adapt_imp,
                        cm, config, self.queues, target_darkness_override=adapt_darkness,
                    )
                    self.active_fitness_calc = fc

                    adapt_radon = (
                        RadonGuide(bundle["gray"], adapt_edges, cm, config, compute_fbp=False, density_override=adapt_darkness)
                        if radon_enabled else None
                    )
                    greedy = GreedyInitializer(cm, fc.target_darkness_flat, fc.importance_flat, config, adapt_radon, stop_Check=self._check_omp_stop_or_pause)

                    # Масштабируем оптическую толщину нити пропорционально контрасту цвета нити
                    alpha_layer = fc.line_alpha * max(0.30, contrast_c)
                    layer_display_idx = len(accumulated_layers)
                    def greedy_cb_adapt(status, path, _idx=layer_display_idx, _name=disp_name, _hex=hex_c, _bud=ch_budget):
                        snap = list(accumulated_layers)
                        snap.append({"name": _name, "color_hex": _hex, "path": path, "ref_lines": _bud})
                        self.queues["status_queue"].put({
                            "type": "greedy_update", "run_id": run_id,
                            "data": {"text": f"{_name}: {status}", "layers": snap, "active_idx": _idx},
                        })

                    base_path = greedy.run(ch_budget, greedy_cb_adapt, layer_label=disp_name, line_alpha=alpha_layer)

                    if len(base_path) >= 2 and not self.omp_stop_flag:
                        self.state.omp_baseline_fitness = fc.calculate_single_fitness(base_path)
                        self.state.layer_best_fitnesses[layer_display_idx] = self.state.omp_baseline_fitness - 1e-6
                        adapt_stages = max(2, config.get("stages", 5) // 2)
                        mgr = IslandManagerThread(
                            config, self.queues, fc, base_path, adapt_radon,
                            layer_idx=layer_display_idx, layer_name=disp_name, emit_done=False,
                            stages_override=adapt_stages, patience_scale=0.55,
                            run_id=run_id, max_lines=ch_budget,
                        )
                        self.manager_thread = mgr
                        mgr.start()
                        mgr.join()
                        if mgr.best_result is not None:
                            base_path = list(mgr.best_result.get("chromosome", []))

                        # Оптимизация и устранение избыточных/перетемняющих нитей (Delta pruning)
                        pruned_path, opt_fit, n_rem, _ = optimize_thread_count_delta(
                            base_path, fc, ch_budget, adapt_radon, max_prune_passes=3, allow_add=False,
                        )
                        if len(pruned_path) >= 2:
                            base_path = pruned_path
                            if n_rem > 0:
                                logging.info(f"Слой '{disp_name}': устранено {n_rem} избыточных нитей")

                    if len(base_path) >= 2:
                        n_done = max(0, len(base_path) - 1)
                        if n_done > 0:
                            accumulated_layers.append({"name": disp_name, "color_hex": hex_c, "path": base_path, "ref_lines": ch_budget})
                            self.queues["status_queue"].put({
                                "type": "greedy_update", "run_id": run_id,
                                "data": {"text": f"{disp_name}: готово ({n_done} нитей)", "layers": list(accumulated_layers), "active_idx": len(accumulated_layers) - 1},
                            })
                    else:
                        logging.info(f"Слой '{disp_name}' не получил нитей (0 линий) и был исключен.")

                self.state.layers = list(accumulated_layers)
                active_palette = [{"name": l["name"], "hex": l["color_hex"]} for l in accumulated_layers]
                self.queues["status_queue"].put({"type": "palette_update", "run_id": run_id, "data": active_palette})

                msg = tr("st_stopped_user", lang) if self.omp_stop_flag else tr("st_adaptive_done", lang, n=len(accumulated_layers))
                self.queues["status_queue"].put({"type": "done", "run_id": run_id, "data": msg})

        except Exception:
            logging.error("Ошибка при подготовке эволюции", exc_info=True)
            self.queues["status_queue"].put({
                "type": "error",
                "run_id": run_id,
                "data": f"{tr('dlg_error', lang)}: {traceback.format_exc()}",
            })

    def _check_omp_stop_or_pause(self) -> bool:
        while not self.omp_pause_event.is_set() and not self.omp_stop_flag:
            time.sleep(0.06)
        return self.omp_stop_flag

    def _rgb_to_lab(self, arr: np.ndarray) -> np.ndarray:
        mask_s = arr > 0.04045
        lin = np.where(mask_s, ((arr + 0.055) / 1.055) ** 2.4, arr / 12.92)
        m = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
        shape = lin.shape
        flat = lin.reshape(-1, 3) @ m.T
        flat /= np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
        eps = 0.008856
        kappa = 903.3
        fx = np.where(flat > eps, flat ** (1.0 / 3.0), (kappa * flat + 16.0) / 116.0)
        L = 116.0 * fx[:, 1] - 16.0
        a = 500.0 * (fx[:, 0] - fx[:, 1])
        b = 200.0 * (fx[:, 1] - fx[:, 2])
        lab = np.stack([L, a, b], axis=-1)
        return lab.reshape(shape[:-1] + (3,))

    def _select_adaptive_colors(
        self,
        rgb_image: np.ndarray,
        n: int,
        mask: Optional[np.ndarray] = None,
        invert_board: bool = False,
    ) -> List[Dict[str, Any]]:
        """Выбирает N оптимальных контрастных цветов нитей из COLOR_CATALOG жадным алгоритмом в CIE Lab."""
        img_f32 = rgb_image.astype(np.float32) / 255.0

        cat_rgb = np.array([[c["rgb"][0], c["rgb"][1], c["rgb"][2]] for c in COLOR_CATALOG], dtype=np.float32) / 255.0
        cat_lab = self._rgb_to_lab(cat_rgb)  # (K, 3)

        # Фильтрация заведомо неэффективных нитей под цвет холста:
        # На светлом холсте исключаем белые/пастельные цвета без контраста (L > 82 или слабая оптическая плотность)
        # На темном холсте исключаем черные/околочерные цвета (L < 18)
        if not invert_board:
            valid_indices = [
                i for i in range(len(COLOR_CATALOG))
                if cat_lab[i, 0] <= 82.0
                and (3.0 - np.sum(cat_rgb[i])) >= 0.35
                and not (np.max(cat_rgb[i]) - np.min(cat_rgb[i]) < 0.06 and np.sum(cat_rgb[i]) > 2.2)
            ]
        else:
            valid_indices = [
                i for i in range(len(COLOR_CATALOG))
                if cat_lab[i, 0] >= 18.0
                and np.sum(cat_rgb[i]) >= 0.35
            ]

        if not valid_indices:
            valid_indices = list(range(len(COLOR_CATALOG)))

        cand_lab = cat_lab[valid_indices]
        cand_colors = [COLOR_CATALOG[i] for i in valid_indices]

        # Извлечение только пигментированных пикселей (не фоновой белой бумаги)
        gray = 0.299 * img_f32[:, :, 0] + 0.587 * img_f32[:, :, 1] + 0.114 * img_f32[:, :, 2]
        if not invert_board:
            darkness = 1.0 - gray
        else:
            darkness = gray

        if mask is not None:
            darkness = np.where(mask.reshape(darkness.shape) == 255, darkness, 0.0)

        pigment_mask = darkness > 0.06
        if np.sum(pigment_mask) < 20:
            pigment_mask = darkness > float(np.percentile(darkness, 40))
        if np.sum(pigment_mask) < 10:
            pigment_mask = np.ones_like(darkness, dtype=bool)

        sampled_rgb = img_f32[pigment_mask]
        sampled_w = darkness[pigment_mask]
        sampled_lab = self._rgb_to_lab(sampled_rgb)

        if len(sampled_lab) > 4096:
            step = len(sampled_lab) // 4096
            sampled_lab = sampled_lab[::step]
            sampled_w = sampled_w[::step]

        diff = sampled_lab[:, np.newaxis, :] - cand_lab[np.newaxis, :, :]  # (N, K, 3)
        dists_all = np.sqrt(np.sum(diff ** 2, axis=-1))  # (N, K)

        # 1-й цвет: взвешенный центроид (1-medoid) активных пикселей изображения
        init_losses = np.sum(sampled_w[:, np.newaxis] * dists_all, axis=0)
        first_idx = int(np.argmin(init_losses))
        selected_cand = [first_idx]
        min_dists = dists_all[:, first_idx].copy()

        # 2-й .. N-й цвета: жадный поиск наибольшего прироста аппроксимации с контролем цветоразличимости (ΔE >= 14)
        for _ in range(n - 1):
            gains = np.sum(sampled_w[:, np.newaxis] * np.maximum(0.0, min_dists[:, np.newaxis] - dists_all), axis=0)
            for s in selected_cand:
                gains[s] = -1.0
                dist_s = np.sqrt(np.sum((cand_lab - cand_lab[s]) ** 2, axis=-1))
                gains[dist_s < 14.0] = -1.0

            best_idx = int(np.argmax(gains))
            if gains[best_idx] <= 0:
                rem = [i for i in range(len(cand_colors)) if i not in selected_cand]
                if not rem:
                    break
                cand_dist_to_sel = np.min([np.sqrt(np.sum((cand_lab[rem] - cand_lab[s]) ** 2, axis=-1)) for s in selected_cand], axis=0)
                best_idx = rem[int(np.argmax(cand_dist_to_sel))]

            selected_cand.append(best_idx)
            min_dists = np.minimum(min_dists, dists_all[:, best_idx])

        # Сортировка слоев от светлых тонов к темным (на белом холсте темные нити поверх светлых создают четкие контуры)
        result = [cand_colors[i] for i in selected_cand]
        if not invert_board:
            result.sort(key=lambda c: (0.299 * c["rgb"][0] + 0.587 * c["rgb"][1] + 0.114 * c["rgb"][2]), reverse=True)
        else:
            result.sort(key=lambda c: (0.299 * c["rgb"][0] + 0.587 * c["rgb"][1] + 0.114 * c["rgb"][2]), reverse=False)

        return result

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
        run_id = self._run_id
        self.omp_stop_flag = True
        self.omp_pause_event.set()
        self._cancel_stats_job()
        if self.manager_thread and self.manager_thread.is_alive():
            self.manager_thread.stop()
        self._send_status(self._t("st_stopping"), run_id=run_id)
        deadline = time.monotonic() + 2.5
        self.root.after(120, lambda: self._wait_for_stop(run_id, deadline))

    def _wait_for_stop(self, run_id: int, deadline: float) -> None:
        if self._run_id != run_id or not self.state.is_running:
            return
        mgr_alive = bool(self.manager_thread and self.manager_thread.is_alive())
        wrk_alive = bool(self._worker_thread and self._worker_thread.is_alive())
        if (mgr_alive or wrk_alive) and time.monotonic() < deadline:
            self.root.after(120, lambda: self._wait_for_stop(run_id, deadline))
            return
        self._finalize_run()

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
            self.ui_manager.widgets["pause_button"].config(text=self._t("btn_resume"))
            self._send_status(self._t("st_paused"), run_id=self._run_id)
        else:
            self.omp_pause_event.set()
            if self.manager_thread and self.manager_thread.is_alive():
                self.manager_thread.resume()
            self.ui_manager.widgets["pause_button"].config(text=self._t("btn_pause"))
            self.stats_update_job = self.root.after(1000, self._update_progress_stats)

    def process_queues(self) -> None:
        try:
            while not self.queues["status_queue"].empty():
                msg = self.queues["status_queue"].get_nowait()
                msg_type = msg.get("type")
                msg_run_id = msg.get("run_id")
                if (
                    msg_run_id is not None
                    and msg_run_id != 0
                    and msg_run_id != self._run_id
                    and msg_type not in ("gpu_status", "export_png_done", "export_png_error")
                ):
                    continue
                handler = self._get_message_handler(msg_type)
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
            "export_png_done": self._handle_export_png_done,
            "export_png_error": self._handle_export_png_error,
            "palette_update": self._handle_palette_update,
        }.get(msg_type or "", lambda _d: None)

    def _handle_palette_update(self, data: List[Dict[str, Any]]) -> None:
        self.ui_manager.update_palette(data)

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
        messagebox.showerror(self._t("dlg_error"), data)
        self.ui_manager.widgets["status_label"].config(text=self._t("st_error"))
        self.ui_manager.toggle_ui_state(False)

    def _handle_gpu_status(self, data: bool) -> None:
        self.ui_manager._last_gpu_flag = bool(data)
        mode = f"PyTorch CUDA ({CUDA_DEVICE_NAME}) + CSR" if data else "SciPy CSR + NumPy (CPU)"
        self.ui_manager.widgets["compute_mode_label"].config(text=self._t("engine_label", mode=mode))

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

        if data.get("force_update") or fitness > self.state.layer_best_fitnesses.get(layer_idx, -1.0):
            self.state.layer_best_fitnesses[layer_idx] = fitness
            self.state.overall_best_fitness = fitness
            self.state.overall_best_chromosome = list(chromosome)
            if layer_idx < len(self.state.layers):
                self.state.layers[layer_idx]["path"] = list(chromosome)
            elif layer_idx == len(self.state.layers):
                invert = self.state.is_invert_board()
                self.state.layers.append({
                    "name": "Белая нить" if invert else "Черная нить",
                    "color_hex": "#f5f5f5" if invert else "#111111",
                    "path": list(chromosome),
                })

            layers_snap = list(self.state.layers)
            self.best_chromosomes_by_key["overall"] = layers_snap

            max_steps = max(1, len(chromosome) - 1)
            scale = self.ui_manager.widgets.get("step_scrubber_scale")
            if isinstance(scale, ttk.Scale):
                scale.configure(to=max(10, max_steps))
            self.state.vars["step_scrubber"].set(max_steps)
            self.ui_manager.update_player_step_label()

            self.ui_manager.update_best_fitness_display(fitness)
            total_m = (
                sum(self.canvas_manager.calculate_path_length_m(l.get("path", [])) for l in layers_snap)
                if self.canvas_manager
                else 0.0
            )
            self.ui_manager.update_thread_length_display(total_m)
            self.request_canvas_update(
                "overall", (layers_snap, config["pins"], config["thread_thickness"]), {"ignore_scrubber": True}
            )

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
        # Если активный слой оказался пустым (например, отсутствующий цвет CMYK), переключаемся на последний непустой
        if self.state.layers:
            cur_idx = min(self.state.active_layer_idx, len(self.state.layers) - 1)
            if len(self.state.layers[cur_idx].get("path", [])) < 2:
                non_empty = [i for i, l in enumerate(self.state.layers) if len(l.get("path", [])) >= 2]
                if non_empty:
                    self.state.active_layer_idx = non_empty[-1]
                    active_p = self.state.layers[self.state.active_layer_idx]["path"]
                    self.state.overall_best_chromosome = list(active_p)
                    max_steps = max(1, len(active_p) - 1)
                    scale = self.ui_manager.widgets.get("step_scrubber_scale")
                    if isinstance(scale, ttk.Scale):
                        scale.configure(to=max(10, max_steps))
                    self.state.vars["step_scrubber"].set(max_steps)

        self._finalize_run()
        self.ui_manager.update_layer_selector()
        total_lines = sum(max(0, len(l.get("path", [])) - 1) for l in self.state.layers)
        if len(self.state.layers) > 1:
            per_layer = " ".join(
                f"{layer_short_code(str(l.get('name', 'L')), self.state.lang)}:{max(0, len(l.get('path', [])) - 1)}"
                for l in self.state.layers
            )
            is_adapt = "Адаптивный" in str(self.state.safe_get("color_mode") or "")
            fmt_key = "st_done_adaptive_fmt" if is_adapt else "st_done_cmyk_fmt"
            msg = self._t(fmt_key, msg=data, per_layer=per_layer, total=total_lines)
        else:
            msg = self._t("st_done_mono_fmt", msg=data, total=total_lines, rem=removed)
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

    def _handle_export_png_done(self, filepath: str) -> None:
        if "export_png_button" in self.ui_manager.widgets and not self.state.is_running:
            self.ui_manager.widgets["export_png_button"].config(state=tk.NORMAL)
        self.ui_manager.widgets["status_label"].config(text=self._t("msg_exp_png_done_status"))
        messagebox.showinfo("PNG", self._t("msg_exp_png_ok", path=filepath))

    def _handle_export_png_error(self, err_msg: str) -> None:
        if "export_png_button" in self.ui_manager.widgets and not self.state.is_running:
            self.ui_manager.widgets["export_png_button"].config(state=tk.NORMAL)
        self.ui_manager.widgets["status_label"].config(text=self._t("msg_exp_png_err_status"))
        messagebox.showerror(self._t("dlg_error"), self._t("msg_exp_png_err", err=err_msg))

    def _send_status(self, message: str, run_id: int = 0) -> None:
        self.queues["status_queue"].put({"type": "status", "run_id": run_id, "data": message})

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
        params["invert_board"] = self.state.is_invert_board()

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
        total_added = 0

        if not silent and self.state.original_image is not None and self.state.layers:
            # Ручной запуск: удаляем петли + избыточные перетемняющие нити через точный O(L) дельта-анализ
            config = self.state.get_config()
            cm = self._ensure_canvas_manager(config)
            bundle = prepare_target_image_bundle(
                self.state.original_image,
                IMAGE_SIZE,
                clahe_clip=config["clahe_clip"],
                gamma=config["gamma"],
                shape=config["canvas_shape"],
                color_mode=config["color_mode"],
            )
            channels = bundle["channels"]
            for l_idx, layer in enumerate(self.state.layers):
                path = layer.get("path", [])
                if l_idx < len(channels):
                    ch = channels[l_idx]
                    fc = FitnessCalculator(
                        bundle["gray"],
                        ch.get("edges", bundle["edges"]),
                        bundle["saliency"],
                        ch["importance"],
                        cm,
                        config,
                        self.queues,
                        target_darkness_override=ch["darkness"],
                    )
                    max_l = (
                        estimate_channel_line_budget(ch["darkness"], bundle["mask"], cm, config["lines"], config["line_weight"])
                        if len(channels) > 1
                        else int(config["lines"])
                    )
                    opt_p, _, rem, add = optimize_thread_count_delta(path, fc, max_l, allow_add=False)
                    layer["path"] = opt_p
                    total_removed += rem
                    total_added += add
                else:
                    pruned, rem = prune_chromosome_loops(path)
                    layer["path"] = pruned
                    total_removed += rem
                if l_idx == self.state.active_layer_idx:
                    self.state.overall_best_chromosome = layer["path"]
        elif self.state.layers:
            for l_idx, layer in enumerate(self.state.layers):
                pruned, rem = prune_chromosome_loops(layer.get("path", []))
                layer["path"] = pruned
                total_removed += rem
                if l_idx == self.state.active_layer_idx:
                    self.state.overall_best_chromosome = pruned
        elif self.state.overall_best_chromosome:
            pruned, total_removed = prune_chromosome_loops(self.state.overall_best_chromosome)
            self.state.overall_best_chromosome = pruned

        if total_removed > 0 or total_added > 0:
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
                    self._t("msg_prune_title"),
                    self._t("msg_prune_done", rem=total_removed, total=total_lines),
                )
        elif not silent:
            messagebox.showinfo(self._t("msg_prune_title"), self._t("msg_prune_clean"))
        return total_removed

    def save_project(self) -> None:
        if not self.state.layers and not self.state.overall_best_chromosome:
            messagebox.showwarning(self._t("msg_save_proj_title"), self._t("msg_synth_first"))
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".sart.json",
            filetypes=[("String Art Studio Project", "*.sart.json"), ("JSON files", "*.json")],
            title=self._t("msg_save_proj_dlg"),
        )
        if not filepath:
            return
        try:
            save_project_json(
                filepath,
                config=self.state.get_config(),
                layers=self._get_export_layers(),
                active_layer_idx=self.state.active_layer_idx,
                step_scrubber=int(self.state.safe_get("step_scrubber") or 1),
                image_path=self.state.original_image_path,
            )
            self.ui_manager.widgets["status_label"].config(
                text=self._t("msg_proj_saved_status", name=os.path.basename(filepath))
            )
            messagebox.showinfo(self._t("msg_save_proj_title"), self._t("msg_save_proj_ok", path=filepath))
        except Exception as exc:
            logging.error("Не удалось сохранить проект", exc_info=True)
            messagebox.showerror(self._t("dlg_error"), self._t("msg_save_proj_err", err=exc))

    def load_project(self) -> None:
        if self.state.is_running:
            messagebox.showwarning(self._t("msg_load_proj_title"), self._t("msg_load_proj_stop"))
            return
        filepath = filedialog.askopenfilename(
            filetypes=[("String Art Studio Project", "*.sart.json *.json"), ("All files", "*.*")],
            title=self._t("msg_load_proj_dlg"),
        )
        if not filepath:
            return
        try:
            proj = load_project_json(filepath)
            self.apply_loaded_project(proj, source_path=filepath)
            messagebox.showinfo(self._t("msg_load_proj_title"), self._t("msg_load_proj_ok", path=filepath))
        except Exception as exc:
            logging.error("Не удалось открыть проект", exc_info=True)
            messagebox.showerror(self._t("dlg_error"), self._t("msg_load_proj_err", err=exc))

    def apply_loaded_project(self, proj: Dict[str, Any], source_path: Optional[str] = None) -> None:
        """Восстанавливает состояние приложения, холст и плеер из загруженного словаря проекта."""
        cfg = proj.get("config", {})
        lang = self.state.lang
        for k, v in cfg.items():
            if k in self.state.vars and k != "language":
                try:
                    if k == "canvas_shape":
                        v = to_display_shape(str(v), lang)
                    elif k == "color_mode":
                        v = to_display_color_mode(str(v), lang)
                    elif k == "structural_metric":
                        v = to_display_metric(str(v), lang)
                    elif k == "preset":
                        v = to_display_preset(str(v), lang)
                    self.state.vars[k].set(v)
                except (tk.TclError, ValueError):
                    pass

        self.state.layers = [
            {
                "name": str(l.get("name", f"Слой {i + 1}")),
                "color_hex": str(l.get("color_hex", "#111111")),
                "path": [int(p) for p in l.get("path", [])],
                **({"ref_lines": int(l["ref_lines"])} if "ref_lines" in l else {}),
            }
            for i, l in enumerate(proj["layers"])
        ]
        self.state.active_layer_idx = max(0, min(len(self.state.layers) - 1, int(proj.get("active_layer_idx", 0))))
        active_path = self.state.layers[self.state.active_layer_idx]["path"]
        self.state.overall_best_chromosome = list(active_path)

        img_path = proj.get("image_path") or ""
        if img_path and os.path.exists(img_path):
            try:
                with Image.open(img_path) as raw:
                    self.state.original_image = raw.copy()
                self.state.original_image_path = img_path
                self.root.redraw_target_canvas()
            except Exception:
                pass
        elif source_path:
            self.state.original_image_path = source_path

        config = self.state.get_config()
        cm = CanvasManager(
            IMAGE_SIZE,
            int(config["pins"]),
            shape=str(config["canvas_shape"]),
            physical_cm=float(config["physical_diameter"]),
        )
        if self.canvas_manager is not None:
            self.canvas_manager.release_gpu_cache()
        self.canvas_manager = cm
        self.radon_guide = None
        self.active_fitness_calc = None

        max_steps = max(1, len(active_path) - 1)
        step_val = max(1, min(max_steps, int(proj.get("step_scrubber", max_steps))))
        scale = self.ui_manager.widgets.get("step_scrubber_scale")
        if isinstance(scale, ttk.Scale):
            scale.configure(to=max(10, max_steps))
        self.state.vars["step_scrubber"].set(step_val)

        self.best_chromosomes_by_key["overall"] = list(self.state.layers)
        self.ui_manager.update_title()
        self.ui_manager.update_layer_selector()
        self.ui_manager.update_player_step_label()
        total_m = sum(cm.calculate_path_length_m(l["path"]) for l in self.state.layers)
        self.ui_manager.update_thread_length_display(total_m)
        self.ui_manager.toggle_ui_state(False)
        self.ui_manager.widgets["status_label"].config(
            text=self._t(
                "msg_proj_loaded_status",
                lines=sum(max(0, len(l["path"]) - 1) for l in self.state.layers),
            )
        )
        self.request_canvas_update(
            "overall",
            (list(self.state.layers), config["pins"], config["thread_thickness"]),
            {"force": True},
        )

    def _get_export_cm(self) -> Optional[CanvasManager]:
        if not self.state.layers and not self.state.overall_best_chromosome:
            messagebox.showwarning(self._t("dlg_warn"), self._t("msg_synth_first"))
            return None
        config = self.state.get_config()
        if (
            self.canvas_manager
            and self.canvas_manager.num_pins == config["pins"]
            and self.canvas_manager.shape == config["canvas_shape"]
        ):
            self.canvas_manager.physical_cm = float(config["physical_diameter"])
            return self.canvas_manager
        return CanvasManager(IMAGE_SIZE, config["pins"], config["canvas_shape"], config["physical_diameter"])

    def _get_export_layers(self) -> List[Dict[str, Any]]:
        if self.state.layers:
            return self.state.layers
        invert = self.state.is_invert_board()
        return [
            {
                "name": "Белая нить" if invert else "Черная нить",
                "color_hex": "#f5f5f5" if invert else "#111111",
                "path": self.state.overall_best_chromosome or [],
            }
        ]

    def export_txt(self) -> None:
        cm = self._get_export_cm()
        if cm is None:
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if not filepath:
            return
        export_instructions_txt(filepath, self._get_export_layers(), cm, lang=self.state.lang)
        messagebox.showinfo("TXT", self._t("msg_exp_txt_ok", path=filepath))

    def export_csv(self) -> None:
        cm = self._get_export_cm()
        if cm is None:
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not filepath:
            return
        export_csv(filepath, self._get_export_layers(), cm, lang=self.state.lang)
        messagebox.showinfo("CSV", self._t("msg_exp_csv_ok", path=filepath))

    def export_gcode(self) -> None:
        cm = self._get_export_cm()
        if cm is None:
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".gcode", filetypes=[("CNC G-Code", "*.gcode *.nc")])
        if not filepath:
            return
        export_gcode_cnc(filepath, self._get_export_layers(), cm)
        messagebox.showinfo("G-Code", self._t("msg_exp_gcode_ok", path=filepath))

    def export_png(self) -> None:
        cm = self._get_export_cm()
        if cm is None:
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png")])
        if not filepath:
            return
        config = self.state.get_config()
        layers_copy = [dict(l, path=list(l.get("path", []))) for l in self._get_export_layers()]
        invert_board = self.state.is_invert_board()
        if "export_png_button" in self.ui_manager.widgets:
            self.ui_manager.widgets["export_png_button"].config(state=tk.DISABLED)
        self.ui_manager.widgets["status_label"].config(text=self._t("msg_exp_png_bg"))

        def _worker() -> None:
            try:
                export_png_ultra_hd(
                    filepath,
                    layers_copy,
                    cm,
                    thickness=config["thread_thickness"],
                    line_weight_percent=config["line_weight"],
                    output_size=2400,
                    invert_board=invert_board,
                )
                self.queues["status_queue"].put({"type": "export_png_done", "data": filepath})
            except Exception as exc:
                logging.error("Ошибка фонового экспорта PNG", exc_info=True)
                self.queues["status_queue"].put({"type": "export_png_error", "data": str(exc)})

        threading.Thread(target=_worker, name="PngExportThread", daemon=True).start()

    def export_svg(self) -> None:
        cm = self._get_export_cm()
        if cm is None:
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG vector", "*.svg")])
        if not filepath:
            return
        config = self.state.get_config()
        export_svg_vector(
            filepath,
            self._get_export_layers(),
            cm,
            thread_thickness=config["thread_thickness"],
            invert_board=self.state.is_invert_board(),
        )
        messagebox.showinfo("SVG", self._t("msg_exp_svg_ok", path=filepath))

    def export_template(self) -> None:
        config = self.state.get_config()
        cm = self.canvas_manager or CanvasManager(
            IMAGE_SIZE, config["pins"], config["canvas_shape"], config["physical_diameter"]
        )
        cm.physical_cm = float(config["physical_diameter"])
        filepath = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("Drill Template SVG", "*.svg")])
        if not filepath:
            return
        export_drill_template_svg(filepath, cm, lang=self.state.lang)
        messagebox.showinfo("1:1 SVG", self._t("msg_exp_tpl_ok", path=filepath))

    def export_html(self) -> None:
        cm = self._get_export_cm()
        if cm is None:
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML5 Web Assistant", "*.html")],
            title=self._t("msg_exp_html_dlg"),
        )
        if not filepath:
            return
        try:
            title = "String Art Masterpiece"
            if self.state.original_image_path:
                title = os.path.splitext(os.path.basename(self.state.original_image_path))[0]
            export_html_assistant(
                filepath,
                title,
                cm,
                self._get_export_layers(),
                invert_board=self.state.is_invert_board(),
                lang=self.state.lang,
            )
            messagebox.showinfo(self._t("dlg_success"), self._t("msg_exp_html_ok", path=filepath))
        except Exception as exc:
            messagebox.showerror(self._t("dlg_error"), self._t("msg_exp_html_err", err=exc))
