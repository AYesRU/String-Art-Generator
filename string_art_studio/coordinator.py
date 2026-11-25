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
    JointRGBGreedyInitializer,
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
                if show_active and 0 <= active_layer_idx < len(layers_data):
                    p = layers_data[active_layer_idx].get("path", [])
                    total_l = len(p) - 1
                    step_idx = total_l if max_step is None else min(total_l, max(1, int(max_step)))
                    if 0 < step_idx < len(p) and max_step is not None and max_step < total_l:
                        active_chord = (p[step_idx - 1], p[step_idx])
                elif show_active and active_layer_idx < 0 and max_step is not None:
                    total_all = sum(max(0, len(l.get("path", [])) - 1) for l in layers_data)
                    if 0 < max_step < total_all:
                        rem = max(1, int(max_step))
                        for lyr in layers_data:
                            p = lyr.get("path", [])
                            cnt = max(0, len(p) - 1)
                            if rem <= cnt:
                                if cnt >= 1:
                                    active_chord = (p[rem - 1], p[rem])
                                break
                            rem -= cnt

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
                    and (now - self._last_analytic_map_time > 0.45 or render_params.get("force_maps"))
                ):
                    if 0 <= active_layer_idx < len(layers_data) and self.active_fitness_calc is not None:
                        self._last_analytic_map_time = now
                        path_main = layers_data[active_layer_idx].get("path", [])
                        if len(path_main) >= 2:
                            _, diff_map, depth_2d = self.active_fitness_calc.calculate_single_fitness_and_maps(path_main)
                            self._push_analytic_maps(diff_map, depth_2d)
                    elif active_layer_idx < 0 and layers_data and self.canvas_manager.flat_ready:
                        flat_size = self.canvas_manager.size * self.canvas_manager.size
                        depth_comp = np.zeros(flat_size, dtype=np.float32)
                        for lyr in layers_data:
                            p = lyr.get("path", [])
                            tot_l = max(0, len(p) - 1)
                            if tot_l == 0:
                                continue
                            ref_n = max(400, int(lyr.get("ref_lines") or tot_l))
                            alpha_l = self.canvas_manager.get_line_alpha(ref_n, line_weight)
                            hex_c = str(lyr.get("color_hex", "#111111")).lstrip("#")
                            if len(hex_c) == 6:
                                try:
                                    rgb_c = [int(hex_c[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
                                except ValueError:
                                    rgb_c = [0.0, 0.0, 0.0]
                            else:
                                rgb_c = [0.0, 0.0, 0.0]
                            mean_abs = (1.0 - (sum(rgb_c) / 3.0)) if not invert_board else (sum(rgb_c) / 3.0)
                            depth_comp += self.canvas_manager.compute_optical_depth_1d(p, alpha_l) * max(0.05, mean_abs)
                        depth_2d = depth_comp.reshape((self.canvas_manager.size, self.canvas_manager.size))
                        tgt_np = self.state.preprocessed_target_np
                        if tgt_np is not None and tgt_np.shape == depth_2d.shape:
                            self._last_analytic_map_time = now
                            tgt_d = (
                                (tgt_np.astype(np.float32) / 255.0)
                                if invert_board
                                else ((255.0 - tgt_np.astype(np.float32)) / 255.0)
                            )
                            cur_d = np.clip(1.0 - np.exp(-depth_2d * 1.6), 0.0, 1.0)
                            diff_map = (tgt_d - cur_d).astype(np.float32)
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

    @staticmethod
    def _parse_hex_rgb_0_255(hex_color: str) -> List[int]:
        h = str(hex_color or "#111111").lstrip("#")
        if len(h) == 6:
            try:
                return [int(h[i : i + 2], 16) for i in (0, 2, 4)]
            except ValueError:
                pass
        return [17, 17, 17]

    @staticmethod
    def _preprocess_rgb_for_adaptive(rgb_img: np.ndarray, config: Dict[str, Any], invert_board: bool) -> np.ndarray:
        rgb_smooth = cv2.bilateralFilter(rgb_img, d=3, sigmaColor=10, sigmaSpace=10)
        gamma_val = float(config.get("gamma", 1.0))
        if abs(gamma_val - 1.0) > 0.03:
            inv_gamma = 1.0 / max(0.1, gamma_val) if not invert_board else gamma_val
            table = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)], dtype=np.uint8)
            rgb_smooth = cv2.LUT(rgb_smooth, table)
        return rgb_smooth

    def _compute_adaptive_budgets(
        self,
        channel_densities: List[np.ndarray],
        mask: np.ndarray,
        cm: CanvasManager,
        max_lines: int,
        line_weight: int,
    ) -> Tuple[List[int], List[int]]:
        phys_budgets = [
            estimate_channel_line_budget(d_map, mask, cm, max_lines, line_weight)
            for d_map in channel_densities
        ]
        total_phys = sum(phys_budgets)
        budget_cap = int(round(max_lines * min(2.40, 1.40 + 0.08 * len(channel_densities))))
        if total_phys > budget_cap and total_phys > 0:
            scale_b = budget_cap / float(total_phys)
            ch_budgets = [
                max(30, min(max_lines, int(round(b * scale_b)))) if b >= 15 else 0
                for b in phys_budgets
            ]
        else:
            ch_budgets = list(phys_budgets)
        return phys_budgets, ch_budgets

    @staticmethod
    def _compute_total_activity(rgb_norm: np.ndarray, invert_board: bool) -> np.ndarray:
        gray_norm = 0.299 * rgb_norm[:, :, 0] + 0.587 * rgb_norm[:, :, 1] + 0.114 * rgb_norm[:, :, 2]
        if not invert_board:
            chroma_strength = 1.0 - np.min(rgb_norm, axis=-1)
            return np.maximum(1.0 - gray_norm, 0.85 * chroma_strength)
        chroma_strength = np.max(rgb_norm, axis=-1)
        return np.maximum(gray_norm, 0.85 * chroma_strength)

    @staticmethod
    def _compute_adaptive_importance(
        ch_density: np.ndarray,
        total_activity: np.ndarray,
        mask: np.ndarray,
        num_colors: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Снижает штраф пересечения чужих цветных зон предмета (сохраняя 100% защиту чистого фона)."""
        edg, imp = _compute_channel_edges_and_importance(ch_density, mask, boost=1.1)
        if num_colors >= 3:
            m_ok = mask.reshape(ch_density.shape) == 255
            act_2d = total_activity.reshape(ch_density.shape)
            other_color_px = m_ok & (ch_density <= 0.06) & (act_2d > 0.15)
            imp = np.where(other_color_px, imp * 0.55, imp)
            mask_area = max(1, int(np.sum(m_ok)))
            patch_px = m_ok & (ch_density > 0.06)
            if 0 < int(np.sum(patch_px)) < int(mask_area * 0.05):
                imp = np.where(patch_px, imp * 1.15, imp)
        return edg, imp.astype(np.float32, copy=False)

    def _build_channels_for_existing_layers(
        self, bundle: Dict[str, Any], config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Восстанавливает целевые каналы плотности под текущие слои проекта (Mono / Dark / CMYK / Adaptive)."""
        is_adaptive = "Адаптивный" in str(config.get("color_mode", ""))
        if not is_adaptive or not self.state.layers:
            return bundle["channels"]

        invert_board = bool(bundle.get("invert_board", False))
        mask = bundle.get("mask", np.ones_like(bundle["gray"], dtype=np.uint8) * 255)
        palette = [
            {
                "name": str(l.get("name", f"Цвет {i + 1}")),
                "hex": str(l.get("color_hex", "#111111")),
                "rgb": self._parse_hex_rgb_0_255(str(l.get("color_hex", "#111111"))),
            }
            for i, l in enumerate(self.state.layers)
        ]
        rgb_smooth = self._preprocess_rgb_for_adaptive(bundle["rgb"], config, invert_board)
        rgb_norm = np.clip(rgb_smooth.astype(np.float32) / 255.0, 0.0, 1.0)
        total_activity = self._compute_total_activity(rgb_norm, invert_board)
        densities = self._decompose_adaptive_channels(rgb_norm, palette, mask, invert_board=invert_board)
        res: List[Dict[str, Any]] = []
        for l, d_map in zip(self.state.layers, densities):
            edg, imp = self._compute_adaptive_importance(d_map, total_activity, mask, len(palette))
            res.append({
                "name": str(l.get("name", "Слой")),
                "color_hex": str(l.get("color_hex", "#111111")),
                "darkness": d_map,
                "importance": imp,
                "edges": edg,
            })
        return res

    def start_evolution_only(self) -> None:
        """Запуск параллельной островной эволюции на текущем пути без перезапуска OMP."""
        if self.state.original_image is None:
            messagebox.showwarning(self._t("dlg_warn"), self._t("msg_load_first"))
            return
        has_base = any(len(l.get("path", [])) >= 4 for l in self.state.layers) or bool(
            self.state.overall_best_chromosome and len(self.state.overall_best_chromosome) >= 4
        )
        if not has_base:
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
            channels = self._build_channels_for_existing_layers(bundle, config)
            is_adaptive = "Адаптивный" in str(config.get("color_mode", ""))
            max_lines = int(config["lines"])
            line_weight = int(config.get("line_weight", 72))

            if is_adaptive and len(channels) > 1:
                _, layer_budgets = self._compute_adaptive_budgets(
                    [ch["darkness"] for ch in channels], bundle["mask"], cm, max_lines, line_weight
                )
            elif len(channels) > 1:
                layer_budgets = [
                    estimate_channel_line_budget(ch["darkness"], bundle["mask"], cm, max_lines, line_weight)
                    for ch in channels
                ]
            else:
                layer_budgets = [max_lines]

            radon_enabled = bool(config.get("radon_enabled", True))
            is_multi = len(self.state.layers) > 1 and len(channels) == len(self.state.layers)

            if is_multi and self.state.active_layer_idx < 0:
                adapt_stages = max(2, config.get("stages", 5) // 2)
                for idx in range(len(self.state.layers)):
                    if self.omp_stop_flag:
                        break
                    ch_data = channels[idx]
                    bud = layer_budgets[idx]
                    base_p = list(self.state.layers[idx].get("path", []))
                    if bud <= 0 or len(base_p) < 2:
                        continue
                    ch_edges = ch_data.get("edges", bundle["edges"])
                    ch_radon = (
                        RadonGuide(
                            bundle["gray"], ch_edges, cm, config, compute_fbp=False, density_override=ch_data["darkness"]
                        )
                        if radon_enabled
                        else None
                    )
                    fc = FitnessCalculator(
                        bundle["gray"], ch_edges, bundle["saliency"], ch_data["importance"],
                        cm, config, self.queues, target_darkness_override=ch_data["darkness"],
                    )
                    self.active_fitness_calc = fc
                    self.state.omp_baseline_fitness = fc.calculate_single_fitness(base_p)
                    self.state.layer_best_fitnesses[idx] = self.state.omp_baseline_fitness - 1e-6
                    disp_name = translate_layer_name(ch_data["name"], lang)
                    self._send_status(tr("st_start_islands", lang, n=config["num_islands"], max_l=bud), run_id=run_id)
                    mgr = IslandManagerThread(
                        config, self.queues, fc, base_p, ch_radon,
                        layer_idx=idx, layer_name=disp_name, emit_done=False,
                        stages_override=adapt_stages, patience_scale=0.55,
                        run_id=run_id, max_lines=bud,
                    )
                    self.manager_thread = mgr
                    mgr.start()
                    mgr.join()
                    if mgr.best_result is not None and len(mgr.best_result.get("chromosome", [])) >= 2:
                        self.state.layers[idx]["path"] = list(mgr.best_result["chromosome"])

                done_msg = (
                    tr("st_stopped_user", lang)
                    if self.omp_stop_flag
                    else (
                        tr("st_adaptive_done", lang, n=len(self.state.layers))
                        if is_adaptive
                        else tr("st_cmyk_done", lang)
                    )
                )
                self.queues["status_queue"].put({"type": "done", "run_id": run_id, "data": done_msg})
                return

            active_idx = max(0, min(self.state.active_layer_idx, len(channels) - 1))
            ch_data = channels[active_idx]
            ch_edges = ch_data.get("edges", bundle["edges"])
            layer_max_lines = layer_budgets[active_idx]

            # Для многослойных режимов строим Радон строго под активный канал (а не кеш от другого слоя)
            if len(channels) > 1 and radon_enabled and layer_max_lines > 0:
                self.radon_guide = RadonGuide(
                    bundle["gray"],
                    ch_edges,
                    cm,
                    config,
                    compute_fbp=False,
                    density_override=ch_data["darkness"],
                )
            elif self.radon_guide is None and radon_enabled and layer_max_lines > 0:
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
            if active_idx < len(self.state.layers):
                base_chromosome = list(self.state.layers[active_idx].get("path", []))
            else:
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
                rgb_img = bundle["rgb"] if bundle.get("rgb") is not None else np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
                mask = bundle.get("mask", np.ones_like(bundle["gray"], dtype=np.uint8) * 255)
                invert_board = bool(bundle.get("invert_board", False))
                max_lines = int(config["lines"])

                # Легкое сглаживание с сохранением 2px контуров и темной обводки
                rgb_smooth = self._preprocess_rgb_for_adaptive(rgb_img, config, invert_board)
                rgb_norm = np.clip(rgb_smooth.astype(np.float32) / 255.0, 0.0, 1.0)

                # 1) Подбор до N перцептивно различимых (CIEDE2000) цветов из каталога
                selected_colors = self._select_adaptive_colors(rgb_smooth, n_colors, mask=mask, invert_board=invert_board)
                self.queues["status_queue"].put({"type": "palette_update", "run_id": run_id, "data": selected_colors})

                # 2) Совместный 3-канальный Beer-Lambert RGB OMP по всем цветам одновременно
                def _joint_cb(status_msg: str, layers_snap: List[Dict[str, Any]], active_idx: int) -> None:
                    self.queues["status_queue"].put({
                        "type": "greedy_update",
                        "run_id": run_id,
                        "data": {"text": status_msg, "layers": layers_snap, "active_idx": active_idx},
                    })

                joint_omp = JointRGBGreedyInitializer(
                    cm,
                    rgb_norm,
                    mask,
                    selected_colors,
                    config,
                    invert_board=invert_board,
                    stop_check=self._check_omp_stop_or_pause,
                )
                joint_paths, joint_fit = joint_omp.run(_joint_cb)
                self.state.omp_baseline_fitness = joint_fit
                self.state.overall_best_fitness = joint_fit

                min_layer_threads = max(5, min(15, max_lines // 40))
                raw_layers = []
                for sc, path in zip(selected_colors, joint_paths):
                    cleaned, _ = prune_chromosome_loops(path)
                    raw_layers.append({
                        "name": sc["name"],
                        "color_hex": sc["hex"],
                        "path": cleaned,
                        "ref_lines": max_lines,
                    })

                accumulated_layers = [
                    l for l in raw_layers if max(0, len(l.get("path", [])) - 1) >= min_layer_threads
                ]
                if not accumulated_layers:
                    accumulated_layers = [l for l in raw_layers if len(l.get("path", [])) >= 2][:1]

                self.state.layers = list(accumulated_layers)
                active_palette = [
                    {
                        "name": l["name"],
                        "hex": l["color_hex"],
                        "rgb": self._parse_hex_rgb_0_255(l.get("color_hex", "#333333")),
                    }
                    for l in accumulated_layers
                ]
                self.queues["status_queue"].put({"type": "palette_update", "run_id": run_id, "data": active_palette})
                final_active_idx = -1 if len(accumulated_layers) > 1 else 0
                if accumulated_layers:
                    self.queues["status_queue"].put({
                        "type": "greedy_update",
                        "run_id": run_id,
                        "data": {
                            "text": tr("st_adaptive_done", lang, n=len(accumulated_layers)),
                            "layers": list(accumulated_layers),
                            "active_idx": final_active_idx,
                            "fitness": joint_fit,
                        },
                    })

                if self.omp_stop_flag:
                    msg = tr("st_stopped_user", lang)
                elif len(accumulated_layers) < n_colors:
                    msg = tr("st_adaptive_done_partial", lang, n=len(accumulated_layers), req=n_colors)
                else:
                    msg = tr("st_adaptive_done", lang, n=len(accumulated_layers))
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

    @staticmethod
    def _ciede2000_vec(lab1: np.ndarray, lab2_arr: np.ndarray) -> np.ndarray:
        """Векторизованное перцептивное цветовое расстояние CIEDE2000 от lab1 (3,) до lab2_arr (K, 3)."""
        L1 = float(lab1[0])
        a1 = float(lab1[1])
        b1 = float(lab1[2])
        L2 = lab2_arr[:, 0].astype(np.float64, copy=False)
        a2 = lab2_arr[:, 1].astype(np.float64, copy=False)
        b2 = lab2_arr[:, 2].astype(np.float64, copy=False)

        C1 = (a1 * a1 + b1 * b1) ** 0.5
        C2 = np.sqrt(a2 * a2 + b2 * b2)
        C_bar = 0.5 * (C1 + C2)
        C_bar7 = C_bar ** 7
        G = 0.5 * (1.0 - np.sqrt(C_bar7 / (C_bar7 + 25.0 ** 7)))

        a1p = (1.0 + G) * a1
        a2p = (1.0 + G) * a2
        C1p = np.sqrt(a1p * a1p + b1 * b1)
        C2p = np.sqrt(a2p * a2p + b2 * b2)

        h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
        h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

        dLp = L2 - L1
        dCp = C2p - C1p
        c_prod = C1p * C2p
        dhp = h2p - h1p
        dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
        dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
        dhp = np.where(c_prod == 0.0, 0.0, dhp)
        dHp = 2.0 * np.sqrt(np.maximum(0.0, c_prod)) * np.sin(np.radians(0.5 * dhp))

        L_bar_p = 0.5 * (L1 + L2)
        C_bar_p = 0.5 * (C1p + C2p)
        h_sum = h1p + h2p
        h_diff = np.abs(h1p - h2p)
        h_bar_p = np.where(
            c_prod == 0.0,
            h_sum,
            np.where(
                h_diff <= 180.0,
                0.5 * h_sum,
                np.where(h_sum < 360.0, 0.5 * (h_sum + 360.0), 0.5 * (h_sum - 360.0)),
            ),
        )
        T = (
            1.0
            - 0.17 * np.cos(np.radians(h_bar_p - 30.0))
            + 0.24 * np.cos(np.radians(2.0 * h_bar_p))
            + 0.32 * np.cos(np.radians(3.0 * h_bar_p + 6.0))
            - 0.20 * np.cos(np.radians(4.0 * h_bar_p - 63.0))
        )
        d_theta = 30.0 * np.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
        C_bar_p7 = C_bar_p ** 7
        RC = 2.0 * np.sqrt(C_bar_p7 / (C_bar_p7 + 25.0 ** 7))
        L50 = (L_bar_p - 50.0) ** 2
        SL = 1.0 + (0.015 * L50) / np.sqrt(20.0 + L50)
        SC = 1.0 + 0.045 * C_bar_p
        SH = 1.0 + 0.015 * C_bar_p * T
        RT = -np.sin(np.radians(2.0 * d_theta)) * RC
        de2 = (dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT * (dCp / SC) * (dHp / SH)
        return np.sqrt(np.maximum(0.0, de2)).astype(np.float32)

    def _select_adaptive_colors(
        self,
        rgb_image: np.ndarray,
        n: int,
        mask: Optional[np.ndarray] = None,
        invert_board: bool = False,
    ) -> List[Dict[str, Any]]:
        """Выбирает до N перцептивно различимых (CIEDE2000) контрастных цветов нитей из COLOR_CATALOG."""
        img_f32 = rgb_image.astype(np.float32) / 255.0
        min_de00 = max(10.5, 15.0 - 0.25 * max(0, int(n) - 4))

        cat_rgb = np.array([[c["rgb"][0], c["rgb"][1], c["rgb"][2]] for c in COLOR_CATALOG], dtype=np.float32) / 255.0
        cat_lab = self._rgb_to_lab(cat_rgb)  # (K, 3)
        cat_chroma = np.sqrt(cat_lab[:, 1] ** 2 + cat_lab[:, 2] ** 2)

        # Фильтрация заведомо неэффективных нитей под цвет холста:
        # На светлом холсте исключаем белые/бледные пастельные цвета без контраста (L > 78 или слабая поглощающая способность)
        # На темном холсте исключаем черные/околочерные цвета (L < 20)
        if not invert_board:
            max_abs = 1.0 - np.min(cat_rgb, axis=-1)
            valid_indices = [
                i for i in range(len(COLOR_CATALOG))
                if (cat_lab[i, 0] <= 68.0 or cat_chroma[i] >= 52.0)
                and cat_lab[i, 0] <= 86.0
                and (3.0 - np.sum(cat_rgb[i])) >= 0.65
                and max_abs[i] >= 0.45
                and not (cat_chroma[i] < 22.0 and cat_lab[i, 0] > 38.0)
            ]
        else:
            max_emit = np.max(cat_rgb, axis=-1)
            valid_indices = [
                i for i in range(len(COLOR_CATALOG))
                if cat_lab[i, 0] >= 20.0
                and np.sum(cat_rgb[i]) >= 0.45
                and max_emit[i] >= 0.35
            ]

        if not valid_indices:
            valid_indices = list(range(len(COLOR_CATALOG)))

        cand_lab = cat_lab[valid_indices]
        cand_colors = [COLOR_CATALOG[i] for i in valid_indices]

        # Извлечение всех пигментированных пикселей и восстановление их собственного цвета нити (de-tinted Beer-Lambert ink color)
        gray = 0.299 * img_f32[:, :, 0] + 0.587 * img_f32[:, :, 1] + 0.114 * img_f32[:, :, 2]
        if not invert_board:
            tau_px = -np.log(np.clip(img_f32, 0.04, 1.0)) / 1.6
            tau_max = np.max(tau_px, axis=-1)
            darkness = np.clip(1.0 - np.exp(-1.6 * tau_max), 0.0, 1.0)
            a_unit = tau_px / (tau_max[:, :, np.newaxis] + 1e-4)
            ink_strength = np.clip(tau_max / 0.65, 0.35, 0.95)[:, :, np.newaxis]
            eff_rgb = np.clip(0.68 * (1.0 - a_unit * ink_strength) + 0.32 * img_f32, 0.0, 1.0)
        else:
            luma_light = gray
            chroma_light = np.max(img_f32, axis=-1)
            darkness = np.maximum(luma_light, 0.85 * chroma_light)
            eff_rgb = img_f32

        if mask is not None:
            darkness = np.where(mask.reshape(darkness.shape) == 255, darkness, 0.0)

        pigment_mask = darkness > 0.08
        if np.sum(pigment_mask) < 20:
            pigment_mask = darkness > float(np.percentile(darkness, 40))
        if np.sum(pigment_mask) < 10:
            pigment_mask = np.ones_like(darkness, dtype=bool)

        # Подавление пикселей 1-2 px переходного сглаживания на границах объектов (ореолы смешивания),
        # при этом темная/контрастная обводка (gray < 0.30) всегда сохраняется для точного подбора контурной нити
        pigment_area = int(np.sum(pigment_mask))
        if pigment_area >= 400:
            gx_rgb = cv2.Sobel(img_f32, cv2.CV_32F, 1, 0, ksize=3)
            gy_rgb = cv2.Sobel(img_f32, cv2.CV_32F, 0, 1, ksize=3)
            grad = np.max(np.hypot(gx_rgb, gy_rgb), axis=-1)
            grad_thresh = float(np.percentile(grad[pigment_mask], 80))
            outline_mask = (gray < 0.30) if not invert_board else (gray > 0.70)
            interior_mask = pigment_mask & ((grad <= grad_thresh) | outline_mask)
            if int(np.sum(interior_mask)) >= max(200, int(pigment_area * 0.35)):
                pigment_mask = interior_mask

        sampled_rgb = eff_rgb[pigment_mask]
        sampled_w = darkness[pigment_mask].astype(np.float32)
        sampled_lab = self._rgb_to_lab(sampled_rgb)
        if not invert_board:
            sampled_w = np.where(sampled_lab[:, 0] < 28.0, sampled_w * 2.2, sampled_w)

        if len(sampled_lab) > 4096:
            step = len(sampled_lab) // 4096
            sampled_lab = sampled_lab[::step]
            sampled_w = sampled_w[::step]

        diff = sampled_lab[:, np.newaxis, :] - cand_lab[np.newaxis, :, :]  # (N, K, 3)
        dists_all = np.sqrt(np.sum(diff ** 2, axis=-1))  # (N, K)

        # 1-й цвет: взвешенный центроид (1-medoid) активных пикселей
        init_losses = np.sum(sampled_w[:, np.newaxis] * dists_all, axis=0)
        first_idx = int(np.argmin(init_losses))
        selected_cand = [first_idx]
        min_dists = dists_all[:, first_idx].copy()

        # 2-й .. N-й цвета: жадное покрытие реальных цветовых кластеров с порогом CIEDE2000 >= min_de00
        # Без фантомной подстановки: если все кластеры уже охвачены, останавливаемся естественным образом.
        cand_de_cache: Dict[int, np.ndarray] = {}
        min_win_px = max(10, int(len(sampled_lab) * 0.005))
        for _ in range(max(0, int(n) - 1)):
            unexplained_w = sampled_w * np.clip(min_dists / 25.0, 0.25, 2.5)
            gains = np.sum(unexplained_w[:, np.newaxis] * np.maximum(0.0, min_dists[:, np.newaxis] - dists_all), axis=0)
            for s in selected_cand:
                gains[s] = -1.0
                if s not in cand_de_cache:
                    cand_de_cache[s] = self._ciede2000_vec(cand_lab[s], cand_lab)
                gains[cand_de_cache[s] < min_de00] = -1.0

            best_idx = int(np.argmax(gains))
            if gains[best_idx] <= 0.0:
                break
            won_px = int(np.sum(dists_all[:, best_idx] < (min_dists - 1.5)))
            if won_px < min_win_px and len(selected_cand) >= min(2, int(n)):
                break

            selected_cand.append(best_idx)
            min_dists = np.minimum(min_dists, dists_all[:, best_idx])

        # k-medoids уточнение: каждый выбранный цвет центрируется по своему реальному кластеру
        min_cl_px = max(8, int(len(sampled_lab) * 0.004))
        sel_arr = list(selected_cand)
        for _ in range(3):
            sub_dists = dists_all[:, sel_arr]  # (N, M)
            nearest = np.argmin(sub_dists, axis=1)
            for m_i in range(len(sel_arr)):
                cl_mask = nearest == m_i
                if int(np.sum(cl_mask)) < min_cl_px:
                    continue
                cl_lab = sampled_lab[cl_mask]
                cl_w = sampled_w[cl_mask]
                cl_chroma = np.sqrt(cl_lab[:, 1] ** 2 + cl_lab[:, 2] ** 2)
                vivid_w = cl_w * (1.0 + 0.35 * (cl_chroma / (np.max(cl_chroma) + 1e-4)))
                target_l = float(np.percentile(cl_lab[:, 0], 30)) if not invert_board else float(np.percentile(cl_lab[:, 0], 70))
                target_a = float(np.sum(vivid_w * cl_lab[:, 1]) / (np.sum(vivid_w) + 1e-6)) * 1.12
                target_b = float(np.sum(vivid_w * cl_lab[:, 2]) / (np.sum(vivid_w) + 1e-6)) * 1.12
                target_vec = np.array([target_l, target_a, target_b], dtype=np.float32)

                cl_losses = np.sqrt(np.sum((cand_lab - target_vec[np.newaxis, :]) ** 2, axis=-1))
                for other_i, other_c in enumerate(sel_arr):
                    if other_i != m_i:
                        d_other = self._ciede2000_vec(cand_lab[other_c], cand_lab)
                        cl_losses[d_other < min_de00] = np.inf
                best_medoid = int(np.argmin(cl_losses))
                if np.isfinite(cl_losses[best_medoid]):
                    sel_arr[m_i] = best_medoid

        # Отсев пустых кластеров после уточнения и финальная дедупликация по CIEDE2000
        sub_dists = dists_all[:, sel_arr]
        nearest = np.argmin(sub_dists, axis=1)
        cl_weights = [
            (m_i, int(np.sum(nearest == m_i)), float(np.sum(sampled_w[nearest == m_i])))
            for m_i in range(len(sel_arr))
        ]
        min_keep_px = max(6, int(len(sampled_lab) * 0.004))
        kept_indices = [m_i for m_i, cnt, _ in cl_weights if cnt >= min_keep_px]
        if len(kept_indices) < min(2, len(sel_arr)):
            ranked = sorted(cl_weights, key=lambda x: (x[2], x[1]), reverse=True)
            kept_indices = [x[0] for x in ranked[: min(2, len(sel_arr))]]

        weight_by_mi = {m_i: w_tot for m_i, _, w_tot in cl_weights}
        kept_sorted = sorted(kept_indices, key=lambda m_i: weight_by_mi[m_i], reverse=True)
        final_cand: List[int] = []
        for m_i in kept_sorted:
            c_idx = sel_arr[m_i]
            if any(
                float(self._ciede2000_vec(cand_lab[c_idx], cand_lab[fc_idx : fc_idx + 1])[0]) < min_de00 * 0.95
                for fc_idx in final_cand
            ):
                continue
            final_cand.append(c_idx)
        sel_arr = final_cand

        # Сортировка слоев от светлых тонов к темным (на белом холсте темные нити поверх светлых создают четкие контуры)
        result = [cand_colors[i] for i in sel_arr]
        if not invert_board:
            result.sort(key=lambda c: (0.299 * c["rgb"][0] + 0.587 * c["rgb"][1] + 0.114 * c["rgb"][2]), reverse=True)
        else:
            result.sort(key=lambda c: (0.299 * c["rgb"][0] + 0.587 * c["rgb"][1] + 0.114 * c["rgb"][2]), reverse=False)

        return result

    def _decompose_adaptive_channels(
        self,
        rgb_norm: np.ndarray,
        selected_colors: List[Dict[str, Any]],
        mask: np.ndarray,
        invert_board: bool = False,
    ) -> List[np.ndarray]:
        """
        Разлагает RGB-изображение на K целевых карт плотности слоев [0..0.95] по закону Бера-Ламберта.
        Использует четкую кластерную привязку (Top-2 с малой температурой + остаточный теневой слой),
        чтобы плотность каждого цвета внутри его области не размывалась при росте N, а сумма поглощений
        всех слоев точно соответствовала исходному изображению без перетемнения («каши»).
        """
        h, w, _ = rgb_norm.shape
        k_num = len(selected_colors)
        if k_num == 0:
            return []

        if not invert_board:
            tau_rgb = -np.log(np.clip(rgb_norm, 0.04, 1.0)) / 1.6  # (H, W, 3)
            a_vecs = np.array(
                [np.clip(1.0 - (np.array(c["rgb"], dtype=np.float32) / 255.0), 0.0, 1.0) for c in selected_colors],
                dtype=np.float32,
            )  # (K, 3)
            gray_norm = 0.299 * rgb_norm[:, :, 0] + 0.587 * rgb_norm[:, :, 1] + 0.114 * rgb_norm[:, :, 2]
            chroma_strength = 1.0 - np.min(rgb_norm, axis=-1)
            activity = np.maximum(1.0 - gray_norm, 0.85 * chroma_strength)
        else:
            tau_rgb = -np.log(np.clip(1.0 - rgb_norm, 0.04, 1.0)) / 1.6
            a_vecs = np.array(
                [np.clip(np.array(c["rgb"], dtype=np.float32) / 255.0, 0.0, 1.0) for c in selected_colors],
                dtype=np.float32,
            )
            gray_norm = 0.299 * rgb_norm[:, :, 0] + 0.587 * rgb_norm[:, :, 1] + 0.114 * rgb_norm[:, :, 2]
            chroma_strength = np.max(rgb_norm, axis=-1)
            activity = np.maximum(gray_norm, 0.85 * chroma_strength)

        valid_mask = (mask == 255) & (activity > 0.05)

        if not invert_board:
            tau_max = np.max(tau_rgb, axis=-1, keepdims=True)
            a_unit = tau_rgb / (tau_max + 1e-4)
            ink_strength = np.clip(tau_max / 0.65, 0.35, 0.95)
            eff_rgb = np.clip(0.68 * (1.0 - a_unit * ink_strength) + 0.32 * rgb_norm, 0.0, 1.0)
        else:
            eff_rgb = rgb_norm

        img_lab = self._rgb_to_lab(eff_rgb)  # (H, W, 3)
        cat_rgb = np.array([[c["rgb"][0], c["rgb"][1], c["rgb"][2]] for c in selected_colors], dtype=np.float32) / 255.0
        cat_lab = self._rgb_to_lab(cat_rgb)   # (K, 3)

        # Перцептивное расстояние в CIE Lab с приоритетом цветового тона (a*, b*)
        d_l = (img_lab[:, :, np.newaxis, 0] - cat_lab[np.newaxis, np.newaxis, :, 0]) * 0.65
        d_a = img_lab[:, :, np.newaxis, 1] - cat_lab[np.newaxis, np.newaxis, :, 1]
        d_b = img_lab[:, :, np.newaxis, 2] - cat_lab[np.newaxis, np.newaxis, :, 2]
        dists_lab = np.sqrt(d_l * d_l + d_a * d_a + d_b * d_b)  # (H, W, K)

        # Четкая привязка к Top-2 ближайшим цветам с температурой 4.5:
        # внутри своей области основной цвет получает вес ~0.8..1.0 независимо от общего числа цветов N
        if k_num > 2:
            top2_thresh = np.partition(dists_lab, 1, axis=-1)[:, :, 1:2]
            is_top2 = dists_lab <= (top2_thresh + 1e-4)
        else:
            is_top2 = np.ones_like(dists_lab, dtype=bool)

        min_d = np.min(dists_lab, axis=-1, keepdims=True)
        exp_w = np.where(is_top2, np.exp(-(dists_lab - min_d) / 4.5), 0.0)
        w_sum = np.sum(exp_w, axis=-1, keepdims=True) + 1e-6
        weights = exp_w / w_sum  # (H, W, K)

        channel_densities: List[np.ndarray] = []
        for k_i in range(k_num):
            a_k = a_vecs[k_i]  # (3,)
            a_norm_sq = float(np.dot(a_k, a_k)) + 0.08
            proj_depth = (
                tau_rgb[:, :, 0] * a_k[0]
                + tau_rgb[:, :, 1] * a_k[1]
                + tau_rgb[:, :, 2] * a_k[2]
            ) / a_norm_sq

            w_k = weights[:, :, k_i]
            d_k = np.where(valid_mask & (w_k >= 0.18), w_k * np.maximum(0.0, proj_depth), 0.0)
            dark_k = np.clip(1.0 - np.exp(-1.6 * d_k), 0.0, 0.95).astype(np.float32)
            dark_k[mask == 0] = 0.0
            dark_k[dark_k < 0.06] = 0.0
            channel_densities.append(dark_k)

        return channel_densities

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
        prev_count = len(self.state.layers)
        self.state.layers = layers_snap
        if active_idx < 0 and len(layers_snap) > 1:
            self.state.active_layer_idx = -1
            active_path = layers_snap[-1].get("path", [])
            max_steps = max(1, sum(max(0, len(l.get("path", [])) - 1) for l in layers_snap))
        else:
            self.state.active_layer_idx = max(0, active_idx)
            active_path = (
                layers_snap[self.state.active_layer_idx].get("path", [])
                if self.state.active_layer_idx < len(layers_snap)
                else []
            )
            max_steps = max(1, len(active_path) - 1)
        self.state.overall_best_chromosome = active_path
        self.best_chromosomes_by_key["overall"] = layers_snap
        self.ui_manager.update_layer_selector()
        self.ui_manager.refresh_palette_tab(throttle_s=0.0 if len(layers_snap) != prev_count else 0.8)

        scale = self.ui_manager.widgets.get("step_scrubber_scale")
        if isinstance(scale, ttk.Scale):
            scale.configure(to=max(10, max_steps))
        self.state.vars["step_scrubber"].set(max_steps)
        self.ui_manager.update_player_step_label()

        if "fitness" in data and float(data["fitness"]) > 0:
            self.state.overall_best_fitness = float(data["fitness"])
            self.ui_manager.update_best_fitness_display(self.state.overall_best_fitness)
        if self.canvas_manager and layers_snap:
            total_m = sum(self.canvas_manager.calculate_path_length_m(l.get("path", [])) for l in layers_snap)
            self.ui_manager.update_thread_length_display(total_m)

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
            if 0 <= layer_idx < len(self.state.layers):
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
            if self.state.active_layer_idx < 0:
                if len(self.state.layers) <= 1:
                    self.state.active_layer_idx = 0
                total_all = max(1, sum(max(0, len(l.get("path", [])) - 1) for l in self.state.layers))
                scale = self.ui_manager.widgets.get("step_scrubber_scale")
                if isinstance(scale, ttk.Scale):
                    scale.configure(to=max(10, total_all))
                self.state.vars["step_scrubber"].set(total_all)
            else:
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
        self.ui_manager.refresh_palette_tab()
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
        removed_empty_layers = False

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
            channels = self._build_channels_for_existing_layers(bundle, config)
            is_adaptive = "Адаптивный" in str(config.get("color_mode", ""))
            max_lines = int(config["lines"])
            line_weight = int(config.get("line_weight", 72))
            if is_adaptive and len(channels) > 1:
                _, layer_budgets = self._compute_adaptive_budgets(
                    [ch["darkness"] for ch in channels], bundle["mask"], cm, max_lines, line_weight
                )
            elif len(channels) > 1:
                layer_budgets = [
                    estimate_channel_line_budget(ch["darkness"], bundle["mask"], cm, max_lines, line_weight)
                    for ch in channels
                ]
            else:
                layer_budgets = [max_lines]

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
                    max_l = layer_budgets[l_idx]
                    opt_p, _, rem, add = optimize_thread_count_delta(path, fc, max_l, allow_add=False)
                    layer["path"] = opt_p
                    total_removed += rem
                    total_added += add
                else:
                    pruned, rem = prune_chromosome_loops(path)
                    layer["path"] = pruned
                    total_removed += rem

            if is_adaptive:
                min_prune_threads = max(5, min(15, max_lines // 40))
                kept = [l for l in self.state.layers if max(0, len(l.get("path", [])) - 1) >= min_prune_threads]
                if not kept:
                    kept = [l for l in self.state.layers if len(l.get("path", [])) >= 2]
                if len(kept) != len(self.state.layers):
                    removed_empty_layers = True
                    self.state.layers = kept
                    if self.state.active_layer_idx < 0:
                        if len(kept) <= 1:
                            self.state.active_layer_idx = 0
                    elif self.state.layers:
                        self.state.active_layer_idx = min(self.state.active_layer_idx, len(self.state.layers) - 1)
                    else:
                        self.state.active_layer_idx = 0
                    pal = [
                        {
                            "name": l["name"],
                            "hex": l["color_hex"],
                            "rgb": self._parse_hex_rgb_0_255(l.get("color_hex", "#333333")),
                        }
                        for l in self.state.layers
                    ]
                    self.ui_manager.update_palette(pal)
                self.ui_manager.update_layer_selector()

            if self.state.layers:
                if 0 <= self.state.active_layer_idx < len(self.state.layers):
                    self.state.overall_best_chromosome = list(self.state.layers[self.state.active_layer_idx].get("path", []))
                else:
                    self.state.overall_best_chromosome = list(self.state.layers[-1].get("path", []))
        elif self.state.layers:
            for l_idx, layer in enumerate(self.state.layers):
                pruned, rem = prune_chromosome_loops(layer.get("path", []))
                layer["path"] = pruned
                total_removed += rem
                if l_idx == self.state.active_layer_idx:
                    self.state.overall_best_chromosome = pruned
            if self.state.active_layer_idx < 0 and self.state.layers:
                self.state.overall_best_chromosome = list(self.state.layers[-1].get("path", []))
        elif self.state.overall_best_chromosome:
            pruned, total_removed = prune_chromosome_loops(self.state.overall_best_chromosome)
            self.state.overall_best_chromosome = pruned

        if total_removed > 0 or total_added > 0 or removed_empty_layers:
            if self.state.layers and self.state.active_layer_idx < 0:
                max_steps = max(1, sum(max(0, len(l.get("path", [])) - 1) for l in self.state.layers))
            else:
                active_path = self.state.overall_best_chromosome or []
                max_steps = max(1, len(active_path) - 1)
            scale = self.ui_manager.widgets.get("step_scrubber_scale")
            if isinstance(scale, ttk.Scale):
                scale.configure(to=max(10, max_steps))
            self.state.vars["step_scrubber"].set(max_steps)
            self.ui_manager.update_layer_selector()
            self.ui_manager.refresh_palette_tab()
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
        raw_ali = int(proj.get("active_layer_idx", 0))
        if raw_ali < 0 and len(self.state.layers) > 1:
            self.state.active_layer_idx = -1
            active_path = self.state.layers[-1]["path"]
        else:
            self.state.active_layer_idx = max(0, min(len(self.state.layers) - 1, raw_ali))
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

        if self.state.active_layer_idx < 0:
            max_steps = max(1, sum(max(0, len(l["path"]) - 1) for l in self.state.layers))
        else:
            max_steps = max(1, len(active_path) - 1)
        step_val = max(1, min(max_steps, int(proj.get("step_scrubber", max_steps))))
        scale = self.ui_manager.widgets.get("step_scrubber_scale")
        if isinstance(scale, ttk.Scale):
            scale.configure(to=max(10, max_steps))
        self.state.vars["step_scrubber"].set(step_val)

        self.best_chromosomes_by_key["overall"] = list(self.state.layers)
        self.ui_manager.update_title()
        self.ui_manager._update_adaptive_n_visibility()
        if "Адаптивный" in str(config.get("color_mode", "")):
            pal = [
                {
                    "name": l["name"],
                    "hex": l["color_hex"],
                    "rgb": self._parse_hex_rgb_0_255(l.get("color_hex", "#333333")),
                }
                for l in self.state.layers
            ]
            self.ui_manager.update_palette(pal)
        else:
            self.ui_manager.update_palette([])
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
