# -*- coding: utf-8 -*-
"""
Гибридный вычислительный движок String Art Studio Pro:
- Сверхбыстрый OMP-синтез через SciPy CSR + pixel_gain (3500+ нитей/сек на CPU)
- Пакетный расчет фитнеса на GPU NVIDIA CUDA (PyTorch) с авто-откатом на CPU
- Островной генетический алгоритм (Island GA) с 4 графовыми мутаторами
- Векторизованный локальный дельта-спуск (меметическая полировка)
"""

import logging
import queue
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import (
    BASE_MUTATION_DIVISOR,
    PATIENCE_BASE,
    PATIENCE_STAGE_FACTOR,
    STAGNATION_MUTATION_FACTOR,
)
from .geometry import CanvasManager, compute_min_pin_distance
from .radon import RadonGuide

try:
    from scipy.ndimage import distance_transform_edt

    SCIPY_AVAILABLE = True
except ImportError:
    distance_transform_edt = None
    SCIPY_AVAILABLE = False

try:
    import torch

    GPU_ENABLED_FLAG = torch.cuda.is_available()
    CUDA_DEVICE_NAME = torch.cuda.get_device_name(0) if GPU_ENABLED_FLAG else "CPU"
except ImportError:
    torch = None
    GPU_ENABLED_FLAG = False
    CUDA_DEVICE_NAME = "CPU"


# ----------------------------------------------------------------------
# ВЕКТОРИЗОВАННЫЕ ЯДРА OMP, ФИТНЕСА И ДЕЛЬТА-МЕТОДОВ
# ----------------------------------------------------------------------


def compute_pixel_gain_omp(
    residual: np.ndarray,
    target_darkness: np.ndarray,
    importance: np.ndarray,
    line_alpha: float,
    fearless_penalty: float,
) -> np.ndarray:
    """
    Вычисляет поэлементный вклад каждого пикселя при добавлении одной нити
    с защитой светлых зон (white_guard) от 'каши'.
    """
    pos = residual > 0.02
    after = residual - line_alpha
    gain = np.empty_like(residual, dtype=np.float32)

    good_m = pos & (after >= 0.0)
    clip_m = pos & (after < 0.0)
    neg_m = ~pos

    if np.any(good_m):
        gain[good_m] = (residual[good_m] ** 2 - after[good_m] ** 2) * importance[good_m]
    if np.any(clip_m):
        good_part = (residual[clip_m] ** 2) * importance[clip_m]
        bad_part = (after[clip_m] ** 2) * importance[clip_m] * (1.1 + fearless_penalty * 1.6)
        gain[clip_m] = good_part - bad_part
    if np.any(neg_m):
        over = line_alpha - residual[neg_m]
        wg = np.where(target_darkness[neg_m] < 0.08, 2.2, 1.2)
        gain[neg_m] = -(over ** 2) * importance[neg_m] * fearless_penalty * wg

    return gain


def evaluate_chromosome_1d_np(
    chromo_arr: np.ndarray,
    line_indices_flat: np.ndarray,
    line_offsets: np.ndarray,
    line_lengths: np.ndarray,
    num_pins: int,
    flat_size: int,
    line_alpha: float,
    target_darkness: np.ndarray,
    importance: np.ndarray,
    fearless_penalty: float,
) -> float:
    """Быстрая оценка хромосомы на CPU через векторизованный np.bincount."""
    if len(chromo_arr) < 2:
        return 0.0
    p1 = chromo_arr[:-1]
    p2 = chromo_arr[1:]
    valid = (p1 != p2) & (p1 >= 0) & (p1 < num_pins) & (p2 >= 0) & (p2 < num_pins)
    p1 = p1[valid]
    p2 = p2[valid]
    pair_idx = p1 * num_pins + p2
    offs = line_offsets[pair_idx]
    lens = line_lengths[pair_idx]
    has = lens > 0

    if np.any(has):
        slices = [line_indices_flat[int(o) : int(o) + int(l)] for o, l in zip(offs[has], lens[has])]
        cat = np.concatenate(slices)
        depth = np.bincount(cat, minlength=flat_size).astype(np.float32) * float(line_alpha)
    else:
        depth = np.zeros(flat_size, dtype=np.float32)

    cur_dark = 1.0 - np.exp(-depth * 1.6)
    diff = target_darkness - cur_dark
    over_mult = 1.0 + fearless_penalty * 1.8
    err = np.where(cur_dark > target_darkness, (diff * diff) * importance * over_mult, (diff * diff) * importance)
    return 1.0 - (float(np.sum(err)) / float(flat_size))


def compute_delta_hinge_np(
    optical_depth: np.ndarray,
    target_darkness: np.ndarray,
    importance: np.ndarray,
    rem1: np.ndarray,
    rem2: np.ndarray,
    add1: np.ndarray,
    add2: np.ndarray,
    line_alpha: float,
    fearless_penalty: float,
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Векторизованно вычисляет дельту ошибки (Delta L2) при замене A->B->C на A->X->C.
    Не изменяет optical_depth на месте; возвращает (delta_err, affected_indices, new_depths).
    """
    all_rem = np.concatenate((rem1, rem2))
    all_add = np.concatenate((add1, add2))
    affected = np.unique(np.concatenate((all_rem, all_add)))
    if len(affected) == 0:
        return 0.0, None, None

    rem_pos = np.searchsorted(affected, all_rem)
    add_pos = np.searchsorted(affected, all_add)
    net_steps = np.bincount(add_pos, minlength=len(affected)) - np.bincount(rem_pos, minlength=len(affected))
    nonzero = net_steps != 0
    if not np.any(nonzero):
        return 0.0, None, None

    af = affected[nonzero]
    od = optical_depth[af]
    nd = np.maximum(0.0, od + net_steps[nonzero].astype(np.float32) * float(line_alpha))

    o_dark = 1.0 - np.exp(-od * 1.6)
    n_dark = 1.0 - np.exp(-nd * 1.6)
    t = target_darkness[af]
    w = importance[af]
    over_mult = 1.0 + fearless_penalty * 1.8

    diff_o = t - o_dark
    diff_n = t - n_dark
    err_o = (diff_o * diff_o) * w * np.where(o_dark > t, over_mult, 1.0)
    err_n = (diff_n * diff_n) * w * np.where(n_dark > t, over_mult, 1.0)
    delta_err = float(np.sum(err_n - err_o))
    return delta_err, af, nd


def prune_chromosome_loops(chromosome: List[int]) -> Tuple[List[int], int]:
    """Удаляет последовательные дубликаты A->A и каскадные возвратные петли A->B->A за O(N)."""
    if not chromosome:
        return [], 0
    stack: List[int] = []
    for pin in chromosome:
        if stack and stack[-1] == pin:
            continue
        if len(stack) >= 2 and stack[-2] == pin:
            stack.pop()
            continue
        stack.append(int(pin))
    removed = max(0, len(chromosome) - len(stack))
    return stack, removed


# ----------------------------------------------------------------------
# БЭКЕНД GPU CUDA (PYTORCH) ДЛЯ ПАКЕТНОЙ ОЦЕНКИ ХРОМОСОМ
# ----------------------------------------------------------------------


class GPUFitnessBackend:
    """Потокобезопасный пакетный вычислитель фитнеса на CUDA (PyTorch)."""

    def __init__(
        self,
        canvas_manager: CanvasManager,
        target_darkness_flat: np.ndarray,
        importance_flat: np.ndarray,
        line_alpha: float,
        fearless_penalty: float,
    ):
        self.available = False
        self.lock = threading.Lock()
        self.canvas_manager = canvas_manager
        self.line_alpha = float(line_alpha)
        self.over_mult = float(1.0 + fearless_penalty * 1.8)
        self.flat_size = canvas_manager.size * canvas_manager.size

        if not GPU_ENABLED_FLAG or torch is None or canvas_manager.line_matrix is None:
            return
        if canvas_manager.line_matrix.shape[0] == 0:
            return

        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                if hasattr(torch.sparse, "check_sparse_tensor_invariants"):
                    torch.sparse.check_sparse_tensor_invariants(False)

                self.device = torch.device("cuda")
                a_t = canvas_manager.line_matrix.T.tocsr()
                self.n_chords = canvas_manager.line_matrix.shape[0]

                self.a_t_csr = torch.sparse_csr_tensor(
                    torch.from_numpy(a_t.indptr.astype(np.int64)).to(self.device),
                    torch.from_numpy(a_t.indices.astype(np.int64)).to(self.device),
                    torch.from_numpy(a_t.data.astype(np.float32)).to(self.device),
                    size=a_t.shape,
                )
            self.target_gpu = torch.from_numpy(target_darkness_flat.astype(np.float32)).to(self.device).unsqueeze(1)
            self.imp_gpu = torch.from_numpy(importance_flat.astype(np.float32)).to(self.device).unsqueeze(1)
            self.available = True
            logging.info(f"GPU CUDA бэкенд активирован: {CUDA_DEVICE_NAME}")
        except Exception as exc:
            logging.warning(f"Не удалось инициализировать CUDA бэкенд, откат на CPU: {exc}")
            self.available = False

    def evaluate_batch(self, chromosomes: List[List[int]]) -> Optional[List[float]]:
        if not self.available or not chromosomes:
            return None

        batch_lids = [self.canvas_manager.chromosome_to_lids(c) for c in chromosomes]
        b = len(batch_lids)
        if all(len(l) == 0 for l in batch_lids):
            return [0.0] * b

        try:
            with self.lock:
                m_t = torch.zeros((self.n_chords, b), device=self.device, dtype=torch.float32)
                rows_list = []
                cols_list = []
                for col_i, lids in enumerate(batch_lids):
                    if len(lids) > 0:
                        rows_list.append(torch.from_numpy(lids.astype(np.int64)))
                        cols_list.append(torch.full((len(lids),), col_i, dtype=torch.int64))

                if rows_list:
                    r_cat = torch.cat(rows_list).to(self.device)
                    c_cat = torch.cat(cols_list).to(self.device)
                    vals = torch.full((r_cat.numel(),), self.line_alpha, device=self.device, dtype=torch.float32)
                    m_t.index_put_((r_cat, c_cat), vals, accumulate=True)

                depths = torch.sparse.mm(self.a_t_csr, m_t)  # (flat_size, b)
                cur_dark = 1.0 - torch.exp(-depths * 1.6)
                diff = self.target_gpu - cur_dark
                over = torch.where(cur_dark > self.target_gpu, self.over_mult, 1.0)
                err = torch.mean((diff * diff) * self.imp_gpu * over, dim=0)
                scores = (1.0 - err).detach().cpu().tolist()
                return [float(s) for s in scores]
        except Exception as exc:
            logging.warning(f"Ошибка пакетной оценки CUDA ({exc}), переключение на CPU.")
            self.available = False
            return None


# ----------------------------------------------------------------------
# КАЛЬКУЛЯТОР ФИТНЕСА И АНАЛИТИЧЕСКИХ КАРТ
# ----------------------------------------------------------------------


class FitnessCalculator:
    """
    Согласованный вычислитель фитнеса (CPU + CUDA) и аналитических карт расхождений/плотности.
    Учитывает выбранную метрику краев ('Преобразование расстояний' или 'Контуры').
    """

    def __init__(
        self,
        target_2d: np.ndarray,
        target_edges_2d: np.ndarray,
        saliency_uint8: np.ndarray,
        importance_2d: np.ndarray,
        canvas_manager: CanvasManager,
        config: Dict[str, Any],
        queues: Dict[str, queue.Queue],
        target_darkness_override: Optional[np.ndarray] = None,
    ):
        self.canvas_manager = canvas_manager
        self.canvas_manager.build_line_tables()
        self.config = config
        self.queues = queues

        self.target_2d = target_2d
        if target_darkness_override is not None:
            self.target_darkness_flat = target_darkness_override.astype(np.float32).flatten()
        else:
            self.target_darkness_flat = ((255.0 - target_2d.astype(np.float32)) / 255.0).flatten()

        contour_boost = (config.get("contour_weight", 55) / 100.0) * 2.5
        edge_norm = (target_edges_2d.astype(np.float32) / 255.0).flatten()

        self.distance_transform_map: Optional[np.ndarray] = None
        if config.get("structural_metric") == "Преобразование расстояний" and SCIPY_AVAILABLE and distance_transform_edt is not None:
            # Сильные контуры задают минимальную дистанцию; их окрестность получает повышенный вес точности
            edge_binary = np.where(target_edges_2d > 40, 0, 255).astype(np.uint8)
            dt = distance_transform_edt(edge_binary).astype(np.float32)
            max_dt = float(dt.max())
            if max_dt > 0:
                dt /= max_dt
            self.distance_transform_map = dt
            dt_affinity = ((1.0 - dt) ** 1.5).flatten()
            structural_term = 0.55 * edge_norm + 0.45 * dt_affinity
        else:
            structural_term = edge_norm

        self.importance_flat = (importance_2d.flatten() + structural_term * contour_boost).astype(np.float32)

        self.line_alpha = self.canvas_manager.get_line_alpha(config.get("lines", 1400), config.get("line_weight", 72))
        self.fearless_penalty = config.get("fearless_factor", 94) / 100.0

        self.gpu_backend = GPUFitnessBackend(
            self.canvas_manager,
            self.target_darkness_flat,
            self.importance_flat,
            self.line_alpha,
            self.fearless_penalty,
        )
        if "status_queue" in self.queues:
            self.queues["status_queue"].put({"type": "gpu_status", "data": self.gpu_backend.available})

    def calculate_total_error(self, optical_depth: np.ndarray) -> float:
        """Вычисляет суммарную взвешенную ошибку (используется для инкрементального отслеживания)."""
        cur_dark = 1.0 - np.exp(-optical_depth * 1.6)
        diff = self.target_darkness_flat - cur_dark
        over_mult = 1.0 + self.fearless_penalty * 1.8
        err = np.where(
            cur_dark > self.target_darkness_flat,
            (diff * diff) * self.importance_flat * over_mult,
            (diff * diff) * self.importance_flat,
        )
        return float(np.sum(err))

    def calculate_single_fitness(self, chromosome: List[int]) -> float:
        if not chromosome or len(chromosome) < 2:
            return 0.0
        arr = np.asarray(chromosome, dtype=np.int64)
        return float(
            evaluate_chromosome_1d_np(
                arr,
                self.canvas_manager.line_indices_flat,
                self.canvas_manager.line_offsets,
                self.canvas_manager.line_lengths,
                self.canvas_manager.num_pins,
                self.canvas_manager.size * self.canvas_manager.size,
                self.line_alpha,
                self.target_darkness_flat,
                self.importance_flat,
                self.fearless_penalty,
            )
        )

    def calculate_fitness_scores(self, chromosomes: List[List[int]]) -> List[float]:
        if not chromosomes:
            return []
        if self.gpu_backend.available and len(chromosomes) >= 4:
            gpu_res = self.gpu_backend.evaluate_batch(chromosomes)
            if gpu_res is not None:
                return gpu_res
        return [self.calculate_single_fitness(c) for c in chromosomes]

    def calculate_single_fitness_and_maps(self, chromosome: List[int]) -> Tuple[float, np.ndarray, np.ndarray]:
        """Возвращает фитнес, 2D-карту ошибок и 2D-тепловую карту плотности нитей."""
        depth_1d = self.canvas_manager.compute_optical_depth_1d(chromosome, self.line_alpha)
        cur_dark_1d = 1.0 - np.exp(-depth_1d * 1.6)
        diff_1d = self.target_darkness_flat - cur_dark_1d
        over_mult = 1.0 + self.fearless_penalty * 1.8
        err = np.where(
            cur_dark_1d > self.target_darkness_flat,
            (diff_1d * diff_1d) * self.importance_flat * over_mult,
            (diff_1d * diff_1d) * self.importance_flat,
        )
        fitness = 1.0 - float(np.sum(err)) / float(len(self.target_darkness_flat))

        sz = self.canvas_manager.size
        depth_2d = depth_1d.reshape((sz, sz))
        cur_dark_2d = cur_dark_1d.reshape((sz, sz))
        tar_dark_2d = self.target_darkness_flat.reshape((sz, sz))
        # Положительное = перетемнили (красный), Отрицательное = недотемнили (синий)
        diff_map = cur_dark_2d - tar_dark_2d
        return fitness, diff_map, depth_2d


# ----------------------------------------------------------------------
# СВЕРХБЫСТРЫЙ OMP-ИНИЦИАЛИЗАТОР (SCIPY CSR + PIXEL GAIN)
# ----------------------------------------------------------------------


class GreedyInitializer:
    def __init__(
        self,
        canvas_manager: CanvasManager,
        target_darkness_flat: np.ndarray,
        importance_flat: np.ndarray,
        config: Dict[str, Any],
        radon_guide: Optional[RadonGuide] = None,
        stop_Check: Optional[Callable[[], bool]] = None,
    ):
        self.canvas_manager = canvas_manager
        self.canvas_manager.build_line_tables()
        self.target_darkness_flat = target_darkness_flat.astype(np.float32, copy=False)
        self.importance_flat = importance_flat.astype(np.float32, copy=False)
        self.config = config
        self.radon_guide = radon_guide
        self.stop_check = stop_Check
        self.num_pins = canvas_manager.num_pins

    def run(
        self,
        num_lines: int,
        update_callback: Callable[[str, List[int]], None],
        layer_label: str = "черной нитью",
    ) -> List[int]:
        if self.num_pins == 0 or self.canvas_manager.line_matrix is None or self.canvas_manager.line_matrix.shape[0] == 0:
            return []

        A = self.canvas_manager.line_matrix
        n_chords = A.shape[0]
        line_alpha = self.canvas_manager.get_line_alpha(num_lines, self.config.get("line_weight", 72))
        fearless = self.config.get("fearless_factor", 94) / 100.0
        radon_inf = (self.config.get("radon_influence", 35) / 100.0) * 0.22

        # Предвычисляем константные по хордам знаменатель длины и штраф за светлые зоны
        chord_lengths = np.asarray(A.indptr[1:] - A.indptr[:-1], dtype=np.float32)
        chord_lengths_safe = np.maximum(1.0, chord_lengths)
        norm_denom = chord_lengths_safe ** 0.90

        bright_mask = (self.target_darkness_flat < 0.08).astype(np.float32)
        bright_hits = A.dot(bright_mask)
        bright_ratio = bright_hits / chord_lengths_safe
        bright_penalty = np.maximum(0.0, bright_ratio - 0.78) * chord_lengths_safe * (line_alpha * line_alpha) * fearless * 1.2

        radon_vec: Optional[np.ndarray] = None
        if self.radon_guide is not None and len(self.radon_guide.chord_weights) == n_chords and radon_inf > 0:
            radon_vec = self.radon_guide.chord_weights

        start_pin = 0
        if self.radon_guide and self.radon_guide.pin_adj_weights is not None and self.radon_guide.pin_adj_weights.size > 0:
            start_pin = int(np.argmax(np.sum(self.radon_guide.pin_adj_weights, axis=1)))
        elif self.canvas_manager.line_keys:
            best_lid_init = int(np.argmax(A.dot(self.target_darkness_flat)))
            start_pin = self.canvas_manager.line_keys[best_lid_init][0]

        path = [start_pin]
        current_pin = start_pin
        residual = self.target_darkness_flat.copy()
        pixel_gain = compute_pixel_gain_omp(residual, self.target_darkness_flat, self.importance_flat, line_alpha, fearless)

        used_lids = np.zeros(n_chords, dtype=bool)
        recent_pins: List[int] = []
        auto_stop_step = min(400, max(60, int(num_lines * 0.45)))
        neg_streak = 0

        t0 = time.perf_counter()
        last_ui = t0

        for i in range(num_lines):
            if self.stop_check and self.stop_check():
                break

            cand_pins = self.canvas_manager.pin_neighbor_pins[current_pin]
            cand_lids = self.canvas_manager.pin_neighbor_lids[current_pin]
            if len(cand_lids) == 0:
                break

            mask = ~used_lids[cand_lids]
            if recent_pins:
                mask_rec = mask & ~np.isin(cand_pins, recent_pins)
                if np.any(mask_rec):
                    mask = mask_rec

            if not np.any(mask):
                break

            valid_pins = cand_pins[mask]
            valid_lids = cand_lids[mask]

            raw_scores = A[valid_lids].dot(pixel_gain)
            scores = (raw_scores - bright_penalty[valid_lids]) / norm_denom[valid_lids]

            if radon_vec is not None:
                pos_s = scores > 0
                if np.any(pos_s):
                    scores[pos_s] *= 1.0 + radon_inf * radon_vec[valid_lids[pos_s]]

            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            best_next = int(valid_pins[best_idx])
            best_lid = int(valid_lids[best_idx])

            if best_score <= 0.0001:
                neg_streak += 1
                if i >= auto_stop_step and neg_streak >= 14:
                    logging.info(f"OMP Авто-стоп на шаге {i}: изображение сформировано без перетемнения.")
                    break
            else:
                neg_streak = 0

            best_pixels = A.indices[A.indptr[best_lid] : A.indptr[best_lid + 1]]
            if len(best_pixels) > 0:
                residual[best_pixels] -= line_alpha
                pixel_gain[best_pixels] = compute_pixel_gain_omp(
                    residual[best_pixels],
                    self.target_darkness_flat[best_pixels],
                    self.importance_flat[best_pixels],
                    line_alpha,
                    fearless,
                )

            used_lids[best_lid] = True
            path.append(best_next)

            recent_pins.append(current_pin)
            if len(recent_pins) > 6:
                recent_pins.pop(0)
            current_pin = best_next

            now = time.perf_counter()
            if (i < 30 or now - last_ui > 0.06 or i == num_lines - 1) and i > 0:
                prog = int(((i + 1) / max(1, num_lines)) * 100)
                rate = (i + 1) / max(0.001, now - t0)
                update_callback(
                    f"Этап 1: CSR OMP ({layer_label}) {prog}% ({len(path) - 1} нитей, {rate:.0f} лин/с)",
                    list(path),
                )
                last_ui = now

        dt = time.perf_counter() - t0
        logging.info(f"CSR OMP завершен ({layer_label}): {len(path) - 1} нитей за {dt:.2f} сек.")
        update_callback(f"OMP-синтез завершен ({len(path) - 1} нитей)", list(path))
        return path


# ----------------------------------------------------------------------
# ОСТРОВНОЙ ГЕНЕТИЧЕСКИЙ АЛГОРИТМ + ВЕКТОРИЗОВАННАЯ ДОВОДКА
# ----------------------------------------------------------------------


class IslandThread(threading.Thread):
    """
    Поток острова эволюционной доводки: графовые мутации (Smart Hinge, Radon Bridge,
    2-Opt Invert, Jitter Pin) и быстрый векторизованный дельта-поиск.
    """

    def __init__(
        self,
        island_id: int,
        config: Dict[str, Any],
        fitness_calculator: FitnessCalculator,
        queues: Dict[str, queue.Queue],
        base_chromosome: List[int],
        radon_guide: Optional[RadonGuide] = None,
        stages_override: Optional[int] = None,
        patience_scale: float = 1.0,
        layer_idx: int = 0,
    ):
        super().__init__(daemon=True, name=f"Island-{island_id + 1}")
        self.island_id = island_id
        self.config = config
        self.fitness_calculator = fitness_calculator
        self.queues = queues
        self.base_chromosome = list(base_chromosome)
        self.radon_guide = radon_guide
        self.stages_override = stages_override
        self.patience_scale = float(patience_scale)
        self.layer_idx = int(layer_idx)
        self.status_queue = queues["status_queue"]
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.population: List[Dict[str, Any]] = []
        self.best_on_island: Optional[Dict[str, Any]] = None
        self.migrant_queue: queue.Queue = queue.Queue()
        self.min_pin_dist = compute_min_pin_distance(fitness_calculator.canvas_manager.num_pins)

        self.mutation_operators = {
            "Smart Hinge": self._smart_hinge_mutation,
            "Radon Bridge": self._radon_bridge_mutation,
            "2-Opt Invert": self._two_opt_mutation,
            "Jitter Pin": self._jitter_pin_mutation,
        }
        self.mutation_stats = {name: {"success": 1, "attempts": 1} for name in self.mutation_operators.keys()}

    def stop(self) -> None:
        self.stop_event.set()
        self.pause_event.set()

    def pause(self) -> None:
        self.pause_event.clear()

    def resume(self) -> None:
        self.pause_event.set()

    def get_best_individual(self) -> Optional[Dict[str, Any]]:
        return self.best_on_island

    def accept_migrant(self, migrant: Dict[str, Any]) -> None:
        self.migrant_queue.put(migrant)

    def _pin_dist(self, a: int, b: int, num_pins: int) -> int:
        d = abs(a - b)
        return min(d, num_pins - d)

    def _smart_hinge_mutation(self, c: List[int], s: int, e: int) -> List[int]:
        if e - s < 3:
            return c
        idx = random.randint(max(1, s), min(len(c) - 2, e - 1))
        prev_p, next_p = c[idx - 1], c[idx + 1]
        num_pins = self.fitness_calculator.canvas_manager.num_pins
        for _ in range(10):
            cand = random.randint(0, num_pins - 1)
            if self._pin_dist(cand, prev_p, num_pins) >= self.min_pin_dist and self._pin_dist(cand, next_p, num_pins) >= self.min_pin_dist:
                c[idx] = cand
                break
        return c

    def _radon_bridge_mutation(self, c: List[int], s: int, e: int) -> List[int]:
        if e - s < 3 or self.radon_guide is None or not self.radon_guide.top_neighbors_cache:
            return self._smart_hinge_mutation(c, s, e)
        idx = random.randint(max(1, s), min(len(c) - 2, e - 1))
        prev_p, next_p = c[idx - 1], c[idx + 1]
        num_pins = self.fitness_calculator.canvas_manager.num_pins
        candidates = self.radon_guide.top_neighbors_cache[prev_p]
        if len(candidates) > 0:
            best_c = c[idx]
            best_w = -1.0
            for cand in candidates[:12]:
                cand_i = int(cand)
                if self._pin_dist(cand_i, next_p, num_pins) < self.min_pin_dist:
                    continue
                w = float(self.radon_guide.pin_adj_weights[prev_p, cand_i] + self.radon_guide.pin_adj_weights[cand_i, next_p])
                if w > best_w:
                    best_w = w
                    best_c = cand_i
            c[idx] = best_c
        return c

    def _two_opt_mutation(self, c: List[int], s: int, e: int) -> List[int]:
        if e - s < 4:
            return c
        num_pins = self.fitness_calculator.canvas_manager.num_pins
        i1 = random.randint(max(1, s), min(len(c) - 4, e - 3))
        i2 = random.randint(i1 + 2, min(len(c) - 1, e, i1 + 25))
        # Проверяем, не создаст ли инверсия слишком короткие граничные хорды
        if self._pin_dist(c[i1 - 1], c[i2 - 1], num_pins) < self.min_pin_dist:
            return c
        if i2 < len(c) and self._pin_dist(c[i1], c[i2], num_pins) < self.min_pin_dist:
            return c
        c[i1:i2] = reversed(c[i1:i2])
        return c

    def _jitter_pin_mutation(self, c: List[int], s: int, e: int) -> List[int]:
        if e - s < 2:
            return c
        idx = random.randint(max(1, s), min(len(c) - 2, e - 1))
        prev_p, next_p = c[idx - 1], c[idx + 1]
        num_pins = self.fitness_calculator.canvas_manager.num_pins
        for shift in random.sample([-3, -2, -1, 1, 2, 3], 6):
            cand = (c[idx] + shift) % num_pins
            if self._pin_dist(cand, prev_p, num_pins) >= self.min_pin_dist and self._pin_dist(cand, next_p, num_pins) >= self.min_pin_dist:
                c[idx] = cand
                break
        return c

    def _graph_splice_crossover(self, p1: List[int], p2: List[int], start: int, end: int) -> List[int]:
        if len(p1) != len(p2) or end - start < 4:
            return list(p1)
        child = list(p1)
        cut = random.randint(start + 1, min(end - 1, len(p1) - 2))
        child[cut:end] = p2[cut:end]
        return child

    def _fast_delta_coordinate_descent(
        self, chromosome: List[int], start: int, end: int, depth: int
    ) -> Tuple[List[int], float]:
        """
        Сверхбыстрая локальная полировка пути за счет векторизованного расчета дельты
        и инкрементального отслеживания ошибки без пересчета всей хромосомы.
        """
        if len(chromosome) < 4 or end - start < 3:
            return list(chromosome), self.fitness_calculator.calculate_single_fitness(chromosome)

        cm = self.fitness_calculator.canvas_manager
        fc = self.fitness_calculator
        num_pins = cm.num_pins
        line_alpha = fc.line_alpha
        fearless = fc.fearless_penalty
        target_dark = fc.target_darkness_flat
        importance = fc.importance_flat
        flat_size = float(len(target_dark))

        best_c = list(chromosome)
        optical_depth = cm.compute_optical_depth_1d(best_c, line_alpha)
        total_err = fc.calculate_total_error(optical_depth)

        low_idx = max(1, start)
        high_idx = min(len(best_c) - 2, end - 1)
        if low_idx >= high_idx:
            return best_c, 1.0 - (total_err / flat_size)

        for _ in range(depth):
            if self.stop_event.is_set():
                break
            idx = random.randint(low_idx, high_idx)
            prev_p = best_c[idx - 1]
            cur_p = best_c[idx]
            next_p = best_c[idx + 1]

            if self.radon_guide and self.radon_guide.top_neighbors_cache and random.random() < 0.65:
                cands = self.radon_guide.top_neighbors_cache[prev_p]
                cand_p = int(cands[random.randint(0, min(9, len(cands) - 1))]) if len(cands) > 0 else random.randint(0, num_pins - 1)
            else:
                cand_p = (cur_p + random.choice([-4, -3, -2, -1, 1, 2, 3, 4])) % num_pins

            if cand_p == cur_p:
                continue
            if self._pin_dist(cand_p, prev_p, num_pins) < self.min_pin_dist or self._pin_dist(cand_p, next_p, num_pins) < self.min_pin_dist:
                continue

            rem1 = cm.get_line_pixel_indices(prev_p, cur_p)
            rem2 = cm.get_line_pixel_indices(cur_p, next_p)
            add1 = cm.get_line_pixel_indices(prev_p, cand_p)
            add2 = cm.get_line_pixel_indices(cand_p, next_p)

            delta_err, af, nd = compute_delta_hinge_np(
                optical_depth, target_dark, importance, rem1, rem2, add1, add2, line_alpha, fearless
            )
            if delta_err < -1e-7 and af is not None and nd is not None:
                optical_depth[af] = nd
                total_err += delta_err
                best_c[idx] = cand_p

        return best_c, 1.0 - (total_err / flat_size)

    def run(self) -> None:
        try:
            init_fit = self.fitness_calculator.calculate_single_fitness(self.base_chromosome)
            self.best_on_island = {"chromosome": list(self.base_chromosome), "fitness": init_fit}
            pop_size = max(4, self.config.get("population_size", 24))
            self.population = [{"chromosome": list(self.base_chromosome), "fitness": init_fit} for _ in range(pop_size)]

            if self.island_id > 0:
                mutated = [
                    self._smart_hinge_mutation(list(self.base_chromosome), 1, len(self.base_chromosome) - 1)
                    for _ in range(1, pop_size)
                ]
                scores = self.fitness_calculator.calculate_fitness_scores(mutated)
                for k, (ch, sc) in enumerate(zip(mutated, scores), start=1):
                    self.population[k] = {"chromosome": ch, "fitness": sc}
                self.population.sort(key=lambda x: x["fitness"], reverse=True)

            self._send_island_update()

            actual_lines = max(2, len(self.base_chromosome) - 1)
            stages = self.stages_override if self.stages_override is not None else max(1, self.config.get("stages", 5))
            lines_per_stage = max(1, actual_lines // stages)

            for stage in range(stages):
                if self.stop_event.is_set():
                    break
                start_line = stage * lines_per_stage
                end_line = actual_lines if stage == stages - 1 else min((stage + 1) * lines_per_stage, actual_lines)
                patience = 0
                patience_limit = max(6, int(round((PATIENCE_BASE + stage * PATIENCE_STAGE_FACTOR) * self.patience_scale)))
                last_best = self.best_on_island["fitness"]

                while patience < patience_limit:
                    self.pause_event.wait()
                    if self.stop_event.is_set():
                        break

                    self._send_generation_tick(stage, patience, patience_limit)
                    self._process_migrant()

                    stagnation = 1.0 + (patience / max(1, patience_limit)) * STAGNATION_MUTATION_FACTOR
                    mut_rate = min((self.config.get("mutation_rate", 18) / BASE_MUTATION_DIVISOR) * stagnation, 0.85)

                    self.population = self._evolve_population(start_line, end_line, mut_rate)

                    if self.config.get("memetic_enabled", True) and not self.stop_event.is_set():
                        num_mem = max(1, int(len(self.population) * (self.config.get("memetic_intensity", 20) / 100.0)))
                        depth = self.config.get("memetic_depth", 25)
                        for idx_p in range(min(num_mem, len(self.population))):
                            if self.stop_event.is_set():
                                break
                            polished, new_f = self._fast_delta_coordinate_descent(
                                self.population[idx_p]["chromosome"], start_line, end_line, depth
                            )
                            if new_f > self.population[idx_p]["fitness"]:
                                self.population[idx_p]["chromosome"] = polished
                                self.population[idx_p]["fitness"] = new_f
                        self.population.sort(key=lambda x: x["fitness"], reverse=True)

                    if self.population[0]["fitness"] > self.best_on_island["fitness"] + 1e-7:
                        self.best_on_island = {
                            "chromosome": list(self.population[0]["chromosome"]),
                            "fitness": self.population[0]["fitness"],
                        }
                        self._send_island_update()

                    if self.best_on_island["fitness"] > last_best + 1e-6:
                        last_best = self.best_on_island["fitness"]
                        patience = 0
                    else:
                        patience += 1

        except Exception:
            logging.error(f"Ошибка на острове {self.island_id}", exc_info=True)

    def _choose_mutation_operator(self) -> Tuple[str, Callable]:
        names = list(self.mutation_operators.keys())
        weights = [self.mutation_stats[n]["success"] / max(1, self.mutation_stats[n]["attempts"]) for n in names]
        chosen = random.choices(names, weights=weights, k=1)[0]
        return chosen, self.mutation_operators[chosen]

    def _evolve_population(self, start_line: int, end_line: int, mut_rate: float) -> List[Dict[str, Any]]:
        pop_size = max(4, self.config.get("population_size", 24))
        elite = {"chromosome": list(self.population[0]["chromosome"]), "fitness": self.population[0]["fitness"]}
        children: List[List[int]] = []
        mut_infos: List[Optional[Tuple[str, float]]] = []

        while len(children) < pop_size - 1:
            k_tour = min(4, len(self.population))
            p1 = max(random.sample(self.population, k_tour), key=lambda x: x["fitness"])
            p2 = max(random.sample(self.population, k_tour), key=lambda x: x["fitness"])
            child = self._graph_splice_crossover(p1["chromosome"], p2["chromosome"], start_line, end_line)

            mut_info = None
            if random.random() < mut_rate:
                m_name, m_func = self._choose_mutation_operator()
                child = m_func(child, start_line, end_line)
                mut_info = (m_name, p1["fitness"])

            children.append(child)
            mut_infos.append(mut_info)

        scores = self.fitness_calculator.calculate_fitness_scores(children)
        new_pop = [elite]
        for child, f_val, mut_info in zip(children, scores, mut_infos):
            if mut_info is not None:
                m_name, parent_f = mut_info
                self.mutation_stats[m_name]["attempts"] += 1
                if f_val > parent_f:
                    self.mutation_stats[m_name]["success"] += 1
            new_pop.append({"chromosome": child, "fitness": f_val})

        new_pop.sort(key=lambda x: x["fitness"], reverse=True)
        return new_pop

    def _process_migrant(self) -> None:
        try:
            migrant = self.migrant_queue.get_nowait()
            if self.population and migrant["fitness"] > self.population[-1]["fitness"]:
                self.population[-1] = {"chromosome": list(migrant["chromosome"]), "fitness": migrant["fitness"]}
                self.population.sort(key=lambda x: x["fitness"], reverse=True)
        except queue.Empty:
            pass

    def _send_island_update(self) -> None:
        if self.best_on_island:
            self.status_queue.put({
                "type": "island_update",
                "data": {
                    "id": self.island_id,
                    "layer_idx": self.layer_idx,
                    "chromosome": list(self.best_on_island["chromosome"]),
                    "fitness": self.best_on_island["fitness"],
                    "mut_stats": {k: dict(v) for k, v in self.mutation_stats.items()},
                },
            })

    def _send_generation_tick(self, stage: int, patience: int, limit: int) -> None:
        self.status_queue.put({"type": "generation_tick", "data": (self.island_id, stage, patience, limit)})


class IslandManagerThread(threading.Thread):
    def __init__(
        self,
        config: Dict[str, Any],
        queues: Dict[str, queue.Queue],
        fitness_calculator: FitnessCalculator,
        base_chromosome: List[int],
        radon_guide: Optional[RadonGuide] = None,
        layer_idx: int = 0,
        layer_name: str = "Черная нить",
        emit_done: bool = True,
        stages_override: Optional[int] = None,
        patience_scale: float = 1.0,
    ):
        super().__init__(daemon=True, name=f"IslandManager-L{layer_idx}")
        self.config = config
        self.queues = queues
        self.fitness_calculator = fitness_calculator
        self.base_chromosome = base_chromosome
        self.radon_guide = radon_guide
        self.layer_idx = layer_idx
        self.layer_name = layer_name
        self.emit_done = emit_done
        self.stages_override = stages_override
        self.patience_scale = patience_scale
        self.islands: List[IslandThread] = []
        self.stop_event = threading.Event()
        self.status_queue = queues["status_queue"]
        self.best_result: Optional[Dict[str, Any]] = None

    def stop(self) -> None:
        self.stop_event.set()
        for island in self.islands:
            island.stop()

    def pause(self) -> None:
        for island in self.islands:
            island.pause()

    def resume(self) -> None:
        for island in self.islands:
            island.resume()

    def run(self) -> None:
        try:
            num_islands = max(1, self.config.get("num_islands", 3))
            self.status_queue.put({
                "type": "status",
                "data": f"Этап 2: Эволюционная полировка ({self.layer_name}) на {num_islands} островах...",
            })
            self.islands = [
                IslandThread(
                    i,
                    self.config,
                    self.fitness_calculator,
                    self.queues,
                    self.base_chromosome,
                    self.radon_guide,
                    stages_override=self.stages_override,
                    patience_scale=self.patience_scale,
                    layer_idx=self.layer_idx,
                )
                for i in range(num_islands)
            ]
            for island in self.islands:
                island.start()

            tick_count = 0
            mig_interval = max(5, self.config.get("migration_interval", 20))
            while any(island.is_alive() for island in self.islands):
                if self.stop_event.is_set():
                    break
                time.sleep(0.15)
                tick_count += 1
                if tick_count % mig_interval == 0 and len(self.islands) > 1:
                    self.status_queue.put({"type": "migration"})
                    migrants = [isl.get_best_individual() for isl in self.islands]
                    if all(m is not None for m in migrants):
                        for i, isl in enumerate(self.islands):
                            isl.accept_migrant(migrants[(i + 1) % len(self.islands)])

            for island in self.islands:
                island.join(timeout=1.5)

            candidates = [isl.get_best_individual() for isl in self.islands if isl.get_best_individual() is not None]
            if candidates:
                self.best_result = max(candidates, key=lambda x: x["fitness"])

            if self.emit_done:
                msg = "Остановлено пользователем." if self.stop_event.is_set() else "Генерация и полировка успешно завершены!"
                self.status_queue.put({"type": "done", "data": msg})
        except Exception:
            logging.error("Ошибка в IslandManagerThread", exc_info=True)
            self.status_queue.put({"type": "error", "data": "Ошибка в диспетчере островов."})
