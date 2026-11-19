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
from .preprocessing import prepare_target_image_bundle
from .radon import RadonGuide

if TYPE_CHECKING:
    from .ui import StringArtApp, UIManager


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

    def is_invert_board(self) -> bool:
        mode = str(self.safe_get("color_mode") or "")
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

    def _next_run_id(self) -> int:
        self._run_id += 1
        return self._run_id

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
        try:
            self._send_status(f"Подготовка фитнес-калькулятора для {config['num_islands']} островов...", run_id=run_id)
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

            if self.radon_guide is None and config.get("radon_enabled", True):
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
            self.state.omp_baseline_fitness = fc.calculate_single_fitness(base_chromosome)
            self._send_status(f"Запуск {config['num_islands']} параллельных островов эволюции...", run_id=run_id)
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
            )
            self.manager_thread.start()
        except Exception:
            logging.error("Ошибка при запуске эволюционной доводки", exc_info=True)
            self.queues["status_queue"].put({
                "type": "error",
                "run_id": run_id,
                "data": f"Ошибка при доводке: {traceback.format_exc()}",
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
            messagebox.showwarning("Внимание", "Пожалуйста, сначала загрузите изображение.")
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
        try:
            self._send_status("Анализ изображения, геометрии и построение таблиц хорд...", run_id=run_id)
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
            radon_enabled = bool(config.get("radon_enabled", True))

            radon_guide = None
            self.radon_guide = None
            if radon_enabled:
                self._send_status("Вычисление томографической синограммы Радона (Ramp FBP)...", run_id=run_id)
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

            if not is_cmyk:
                ch = channels[0]
                ch_edges = ch.get("edges", bundle["edges"])
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
                self._send_status(f"Этап 1: Высокоточный CSR OMP-синтез ({ch['name']})...", run_id=run_id)
                greedy = GreedyInitializer(
                    cm,
                    fc.target_darkness_flat,
                    fc.importance_flat,
                    config,
                    radon_guide,
                    stop_Check=self._check_omp_stop_or_pause,
                )

                def greedy_cb(status: str, path: List[int]) -> None:
                    layers_snap = [{"name": ch["name"], "color_hex": ch["color_hex"], "path": path}]
                    self.queues["status_queue"].put({
                        "type": "greedy_update",
                        "run_id": run_id,
                        "data": {"text": status, "layers": layers_snap, "active_idx": 0},
                    })

                base_chromosome = greedy.run(config["lines"], greedy_cb, layer_label=ch["name"])
                if not base_chromosome or len(base_chromosome) < 2:
                    self.queues["status_queue"].put({
                        "type": "error",
                        "run_id": run_id,
                        "data": "Не удалось построить начальный путь.",
                    })
                    return

                self.state.layers = [{"name": ch["name"], "color_hex": ch["color_hex"], "path": list(base_chromosome)}]
                omp_fit, diff_map, depth_2d = fc.calculate_single_fitness_and_maps(base_chromosome)
                self.state.omp_baseline_fitness = omp_fit
                self.state.layer_best_fitnesses[0] = omp_fit
                self.queues["status_queue"].put({
                    "type": "island_update",
                    "run_id": run_id,
                    "data": {"id": 0, "chromosome": base_chromosome, "fitness": omp_fit, "mut_stats": {}, "layer_idx": 0},
                })
                self._push_analytic_maps(diff_map, depth_2d)

                if self.omp_stop_flag:
                    self.queues["status_queue"].put({"type": "done", "run_id": run_id, "data": "Остановлено на этапе OMP."})
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
                )
                self.manager_thread.start()
            else:
                # 4-слойный цветной CMYK синтез с индивидуальными контурами и синограммами Радона по каналам
                for c_idx, ch in enumerate(channels):
                    if self.omp_stop_flag:
                        break
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
                    self._send_status(f"CMYK слой {c_idx + 1}/4 ({ch['name']}): CSR OMP-синтез...", run_id=run_id)
                    greedy = GreedyInitializer(
                        cm,
                        fc.target_darkness_flat,
                        fc.importance_flat,
                        config,
                        ch_radon,
                        stop_Check=self._check_omp_stop_or_pause,
                    )

                    def greedy_cb_cmyk(status: str, path: List[int], idx=c_idx, ch_meta=ch) -> None:
                        snap = list(accumulated_layers)
                        snap.append({"name": ch_meta["name"], "color_hex": ch_meta["color_hex"], "path": path})
                        self.queues["status_queue"].put({
                            "type": "greedy_update",
                            "run_id": run_id,
                            "data": {"text": f"Слой {idx + 1}/4 ({ch_meta['name']}): {status}", "layers": snap, "active_idx": idx},
                        })

                    base_path = greedy.run(config["lines"], greedy_cb_cmyk, layer_label=ch["name"])
                    if len(base_path) >= 2 and not self.omp_stop_flag:
                        omp_layer_fit = fc.calculate_single_fitness(base_path)
                        self.state.omp_baseline_fitness = omp_layer_fit
                        self.state.layer_best_fitnesses[c_idx] = omp_layer_fit - 1e-6
                        snap_pre = list(accumulated_layers) + [
                            {"name": ch["name"], "color_hex": ch["color_hex"], "path": list(base_path)}
                        ]
                        self.queues["status_queue"].put({
                            "type": "greedy_update",
                            "run_id": run_id,
                            "data": {
                                "text": f"Слой {c_idx + 1}/4 ({ch['name']}): эволюционная доводка...",
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
                        )
                        self.manager_thread = mgr
                        mgr.start()
                        mgr.join()
                        if mgr.best_result and len(mgr.best_result.get("chromosome", [])) >= 2:
                            base_path = list(mgr.best_result["chromosome"])

                    accumulated_layers.append({"name": ch["name"], "color_hex": ch["color_hex"], "path": base_path})
                    self.queues["status_queue"].put({
                        "type": "greedy_update",
                        "run_id": run_id,
                        "data": {"text": f"Слой {ch['name']} готов", "layers": list(accumulated_layers), "active_idx": c_idx},
                    })

                msg = "Остановлено пользователем." if self.omp_stop_flag else "Цветной CMYK синтез и полировка завершены!"
                self.queues["status_queue"].put({"type": "done", "run_id": run_id, "data": msg})

        except Exception:
            logging.error("Ошибка при подготовке эволюции", exc_info=True)
            self.queues["status_queue"].put({
                "type": "error",
                "run_id": run_id,
                "data": f"Ошибка при подготовке: {traceback.format_exc()}",
            })

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
        run_id = self._run_id
        self.omp_stop_flag = True
        self.omp_pause_event.set()
        self._cancel_stats_job()
        if self.manager_thread and self.manager_thread.is_alive():
            self.manager_thread.stop()
        self._send_status("Остановка потоков...", run_id=run_id)
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
            self.ui_manager.widgets["pause_button"].config(text="► Продолжить")
            self._send_status("Пауза...", run_id=self._run_id)
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

    def _handle_export_png_done(self, filepath: str) -> None:
        if "export_png_button" in self.ui_manager.widgets and not self.state.is_running:
            self.ui_manager.widgets["export_png_button"].config(state=tk.NORMAL)
        self.ui_manager.widgets["status_label"].config(text="Экспорт PNG 2400x2400 завершен.")
        messagebox.showinfo("Экспорт PNG", f"Изображение высокого разрешения сохранено в:\n{filepath}")

    def _handle_export_png_error(self, err_msg: str) -> None:
        if "export_png_button" in self.ui_manager.widgets and not self.state.is_running:
            self.ui_manager.widgets["export_png_button"].config(state=tk.NORMAL)
        self.ui_manager.widgets["status_label"].config(text="Ошибка экспорта PNG.")
        messagebox.showerror("Ошибка экспорта PNG", f"Не удалось сохранить Ultra-HD PNG:\n{err_msg}")

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

    def save_project(self) -> None:
        if not self.state.layers and not self.state.overall_best_chromosome:
            messagebox.showwarning("Сохранение проекта", "Сначала выполните синтез картины.")
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".sart.json",
            filetypes=[("String Art Studio Project", "*.sart.json"), ("JSON files", "*.json")],
            title="Сохранить проект String Art Studio",
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
            self.ui_manager.widgets["status_label"].config(text=f"Проект сохранен: {os.path.basename(filepath)}")
            messagebox.showinfo("Проект сохранен", f"Проект успешно сохранен:\n{filepath}")
        except Exception as exc:
            logging.error("Не удалось сохранить проект", exc_info=True)
            messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить проект:\n{exc}")

    def load_project(self) -> None:
        if self.state.is_running:
            messagebox.showwarning("Загрузка проекта", "Остановите текущую генерацию перед загрузкой проекта.")
            return
        filepath = filedialog.askopenfilename(
            filetypes=[("String Art Studio Project", "*.sart.json *.json"), ("All files", "*.*")],
            title="Открыть проект String Art Studio",
        )
        if not filepath:
            return
        try:
            proj = load_project_json(filepath)
            self.apply_loaded_project(proj, source_path=filepath)
            messagebox.showinfo("Проект загружен", f"Проект успешно открыт:\n{filepath}")
        except Exception as exc:
            logging.error("Не удалось открыть проект", exc_info=True)
            messagebox.showerror("Ошибка загрузки", f"Не удалось открыть проект:\n{exc}")

    def apply_loaded_project(self, proj: Dict[str, Any], source_path: Optional[str] = None) -> None:
        """Восстанавливает состояние приложения, холст и плеер из загруженного словаря проекта."""
        cfg = proj.get("config", {})
        for k, v in cfg.items():
            if k in self.state.vars:
                try:
                    self.state.vars[k].set(v)
                except (tk.TclError, ValueError):
                    pass

        self.state.layers = [
            {
                "name": str(l.get("name", f"Слой {i + 1}")),
                "color_hex": str(l.get("color_hex", "#111111")),
                "path": [int(p) for p in l.get("path", [])],
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
            text=f"Проект загружен ({sum(max(0, len(l['path']) - 1) for l in self.state.layers)} нитей)."
        )
        self.request_canvas_update(
            "overall",
            (list(self.state.layers), config["pins"], config["thread_thickness"]),
            {"force": True},
        )

    def _get_export_cm(self) -> Optional[CanvasManager]:
        if not self.state.layers and not self.state.overall_best_chromosome:
            messagebox.showwarning("Внимание", "Сначала выполните синтез картины.")
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
        layers_copy = [dict(l, path=list(l.get("path", []))) for l in self._get_export_layers()]
        invert_board = self.state.is_invert_board()
        if "export_png_button" in self.ui_manager.widgets:
            self.ui_manager.widgets["export_png_button"].config(state=tk.DISABLED)
        self.ui_manager.widgets["status_label"].config(text="Экспорт Ultra HD PNG (2400x2400px) в фоне...")

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
        messagebox.showinfo("Экспорт SVG", f"Векторный файл SVG успешно сохранен:\n{filepath}")

    def export_template(self) -> None:
        config = self.state.get_config()
        cm = self.canvas_manager or CanvasManager(
            IMAGE_SIZE, config["pins"], config["canvas_shape"], config["physical_diameter"]
        )
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
            export_html_assistant(
                filepath,
                title,
                cm,
                self._get_export_layers(),
                invert_board=self.state.is_invert_board(),
            )
            messagebox.showinfo(
                "Успех",
                f"Автономный веб-ассистент успешно сохранен в:\n{filepath}\n\n"
                "Откройте файл в любом браузере на ПК, планшете или смартфоне!",
            )
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось экспортировать HTML:\n{exc}")
