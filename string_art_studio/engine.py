# -*- coding: utf-8 -*-
"""
Гибридный вычислительный движок String Art Studio Pro:
- Сверхбыстрый OMP-синтез через преднарезанные подматрицы SciPy CSR + pixel_gain (10,000+ нитей/сек на CPU)
- Пакетный расчет фитнеса на GPU NVIDIA CUDA (PyTorch) с общим кэшем CSR в VRAM и авто-откатом на CPU
- Островной генетический алгоритм (Island GA) с 4 графовыми мутаторами и проверкой границ кроссовера
- Векторизованный O(L) дельта-спуск со scratch-буфером (меметическая полировка)
"""

from collections import deque
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
from .i18n import normalize_lang, to_canonical_metric, translate_layer_name
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


def compute_exact_beer_lambert_gain(
    optical_depth: np.ndarray,
    target_darkness: np.ndarray,
    importance: np.ndarray,
    line_alpha: float,
    fearless_penalty: float,
) -> np.ndarray:
    """
    Вычисляет точное поэлементное уменьшение ошибки Бера-Ламберта (e0 - e1) при добавлении одной нити.
    Для любой хорды с уникальными пикселями сумма gain по хорде равна точному выигрышу фитнеса.
    """
    over_mult = 1.0 + float(fearless_penalty) * 1.8
    c0 = 1.0 - np.exp(-optical_depth * 1.6)
    c1 = 1.0 - np.exp(-(optical_depth + float(line_alpha)) * 1.6)
    d0 = target_darkness - c0
    d1 = target_darkness - c1
    e0 = (d0 * d0) * importance * np.where(c0 > target_darkness, over_mult, 1.0)
    e1 = (d1 * d1) * importance * np.where(c1 > target_darkness, over_mult, 1.0)
    return (e0 - e1).astype(np.float32)


def estimate_channel_line_budget(
    target_darkness: np.ndarray,
    mask: Optional[np.ndarray],
    canvas_manager: CanvasManager,
    max_lines: int,
    line_weight_percent: int = 72,
) -> int:
    """
    Вычисляет индивидуальный физический бюджет нитей для канала (0..max_lines)
    по интегральной оптической глубине Бера-Ламберта внутри маски.
    Отсутствующий в картине цвет (например, Cyan на красном изображении) получает бюджет 0.
    """
    max_l = max(0, int(max_lines))
    if max_l == 0:
        return 0

    d = np.clip(target_darkness.astype(np.float32, copy=False), 0.0, 0.96)
    if mask is not None:
        m = mask.reshape(d.shape) == 255
        d = np.where(m, d, 0.0)

    active_px = int(np.sum(d > 0.06))
    if active_px == 0:
        return 0

    line_alpha = canvas_manager.get_line_alpha(max_l, line_weight_percent)
    if canvas_manager.line_matrix is not None and canvas_manager.line_matrix.shape[0] > 0:
        A = canvas_manager.line_matrix
        avg_chord_px = float(np.mean(A.indptr[1:] - A.indptr[:-1]))
    else:
        avg_chord_px = float(canvas_manager.size) * 0.65

    tau_mass = float(np.sum(-np.log(1.0 - d) / 1.6))
    mass_per_thread = max(1.0, avg_chord_px * line_alpha)
    if tau_mass < mass_per_thread * 0.20:
        return 0

    raw_est = tau_mass / mass_per_thread
    # Запас 1.8x + 35 нитей под геометрические перекрытия хорд; точный Beer-Lambert OMP остановится раньше при насыщении
    return int(np.clip(round(raw_est * 1.80 + 35.0), 10, max_l))


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
    """Быстрая оценка хромосомы на CPU через векторизованный np.bincount (корректна и для пустого пути)."""
    if len(chromo_arr) < 2:
        err_empty = (target_darkness * target_darkness) * importance
        return 1.0 - (float(np.sum(err_empty)) / float(flat_size))
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


def _compute_delta_chords_np(
    optical_depth: np.ndarray,
    target_darkness: np.ndarray,
    importance: np.ndarray,
    rem_chords: Tuple[np.ndarray, ...],
    add_chords: Tuple[np.ndarray, ...],
    line_alpha: float,
    fearless_penalty: float,
    scratch: Optional[np.ndarray] = None,
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Универсальное O(L) ядро точной дельты ошибки Бера-Ламберта при удалении rem_chords и добавлении add_chords.
    Используется для замены (hinge), удаления лишней нити (prune) и вставки новой нити (insert).
    """
    non_empty_rem = [a for a in rem_chords if len(a) > 0]
    non_empty_add = [a for a in add_chords if len(a) > 0]
    if not non_empty_rem and not non_empty_add:
        return 0.0, None, None

    if scratch is not None:
        for arr in non_empty_rem:
            scratch[arr] -= 1
        for arr in non_empty_add:
            scratch[arr] += 1

        af_parts: List[np.ndarray] = []
        step_parts: List[np.ndarray] = []
        # Каждая DDA-хорда имеет строго уникальные индексы; немедленное зануление scratch
        # гарантирует точный учет пересечений и полностью нулевой буфер после вызова.
        for arr in (*non_empty_rem, *non_empty_add):
            vals = scratch[arr]
            nz = vals != 0
            if np.any(nz):
                sub_idx = arr[nz]
                af_parts.append(sub_idx)
                step_parts.append(vals[nz])
                scratch[sub_idx] = 0

        if not af_parts:
            return 0.0, None, None

        af = af_parts[0] if len(af_parts) == 1 else np.concatenate(af_parts)
        net = step_parts[0] if len(step_parts) == 1 else np.concatenate(step_parts)
    else:
        all_rem = np.concatenate(non_empty_rem) if non_empty_rem else np.empty(0, dtype=np.int32)
        all_add = np.concatenate(non_empty_add) if non_empty_add else np.empty(0, dtype=np.int32)
        affected = np.unique(np.concatenate((all_rem, all_add)))
        if len(affected) == 0:
            return 0.0, None, None

        rem_pos = np.searchsorted(affected, all_rem) if len(all_rem) > 0 else np.empty(0, dtype=np.int64)
        add_pos = np.searchsorted(affected, all_add) if len(all_add) > 0 else np.empty(0, dtype=np.int64)
        net_steps = np.bincount(add_pos, minlength=len(affected)) - np.bincount(rem_pos, minlength=len(affected))
        nonzero = net_steps != 0
        if not np.any(nonzero):
            return 0.0, None, None

        af = affected[nonzero]
        net = net_steps[nonzero]

    od = optical_depth[af]
    nd = np.maximum(0.0, od + net.astype(np.float32) * float(line_alpha))

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
    scratch: Optional[np.ndarray] = None,
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Векторизованно вычисляет дельту ошибки (Delta L2) при замене A->B->C на A->X->C.
    При передаче предвыделенного буфера scratch (int8, нули) работает за O(L) без сортировок np.unique.
    """
    return _compute_delta_chords_np(
        optical_depth,
        target_darkness,
        importance,
        (rem1, rem2),
        (add1, add2),
        line_alpha,
        fearless_penalty,
        scratch=scratch,
    )


def compute_delta_remove_pin_np(
    optical_depth: np.ndarray,
    target_darkness: np.ndarray,
    importance: np.ndarray,
    rem1: np.ndarray,
    rem2: np.ndarray,
    add_bridge: np.ndarray,
    line_alpha: float,
    fearless_penalty: float,
    scratch: Optional[np.ndarray] = None,
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Точная O(L) дельта ошибки при удалении лишнего гвоздя B (схлопывание A->B->C в A->C
    или удаление краевой хорды A->B при пустых rem2/add_bridge). Уменьшает число нитей на 1.
    """
    return _compute_delta_chords_np(
        optical_depth,
        target_darkness,
        importance,
        (rem1, rem2),
        (add_bridge,),
        line_alpha,
        fearless_penalty,
        scratch=scratch,
    )


def compute_delta_insert_pin_np(
    optical_depth: np.ndarray,
    target_darkness: np.ndarray,
    importance: np.ndarray,
    rem_bridge: np.ndarray,
    add1: np.ndarray,
    add2: np.ndarray,
    line_alpha: float,
    fearless_penalty: float,
    scratch: Optional[np.ndarray] = None,
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Точная O(L) дельта ошибки при вставке нового гвоздя X (разбиение A->C на A->X->C
    или достраивание краевой хорды Z->X при пустых rem_bridge/add2). Увеличивает число нитей на 1.
    """
    return _compute_delta_chords_np(
        optical_depth,
        target_darkness,
        importance,
        (rem_bridge,),
        (add1, add2),
        line_alpha,
        fearless_penalty,
        scratch=scratch,
    )


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
    """Потокобезопасный пакетный вычислитель фитнеса на CUDA (PyTorch) с общим кэшем матрицы в VRAM."""

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
        self.empty_fitness = 1.0 - float(np.sum((target_darkness_flat * target_darkness_flat) * importance_flat)) / float(
            max(1, self.flat_size)
        )
        self._m_t_buf: Optional[Any] = None
        self._m_t_cols: int = 0

        if not GPU_ENABLED_FLAG or torch is None or canvas_manager.line_matrix is None:
            return
        if canvas_manager.line_matrix.shape[0] == 0:
            return

        try:
            a_t_csr = canvas_manager.get_gpu_transposed_matrix()
            if a_t_csr is None:
                return
            self.device = torch.device("cuda")
            self.n_chords = canvas_manager.line_matrix.shape[0]
            self.a_t_csr = a_t_csr
            self.target_gpu = torch.from_numpy(target_darkness_flat.astype(np.float32)).to(self.device).unsqueeze(1)
            self.imp_gpu = torch.from_numpy(importance_flat.astype(np.float32)).to(self.device).unsqueeze(1)
            self.available = True
            logging.info(f"GPU CUDA бэкенд активирован (общий кэш CSR): {CUDA_DEVICE_NAME}")
        except Exception as exc:
            logging.warning(f"Не удалось инициализировать CUDA бэкенд, откат на CPU: {exc}")
            self.available = False

    def evaluate_batch(self, chromosomes: List[List[int]]) -> Optional[List[float]]:
        if not self.available or not chromosomes:
            return None

        batch_lids = [self.canvas_manager.chromosome_to_lids(c) for c in chromosomes]
        b = len(batch_lids)
        lengths = [len(l) for l in batch_lids]
        total_nnz = sum(lengths)
        if total_nnz == 0:
            return [self.empty_fitness] * b

        rows_np = np.concatenate([l for l in batch_lids if len(l) > 0]).astype(np.int64, copy=False)
        cols_np = np.repeat(np.arange(b, dtype=np.int64), lengths)

        try:
            with self.lock:
                if self._m_t_buf is None or self._m_t_cols != b:
                    self._m_t_buf = torch.zeros((self.n_chords, b), device=self.device, dtype=torch.float32)
                    self._m_t_cols = b
                else:
                    self._m_t_buf.zero_()

                r_cat = torch.from_numpy(rows_np).to(self.device, non_blocking=True)
                c_cat = torch.from_numpy(cols_np).to(self.device, non_blocking=True)
                vals = torch.full((total_nnz,), self.line_alpha, device=self.device, dtype=torch.float32)
                self._m_t_buf.index_put_((r_cat, c_cat), vals, accumulate=True)

                depths = torch.sparse.mm(self.a_t_csr, self._m_t_buf)  # (flat_size, b)
                cur_dark = 1.0 - torch.exp(-depths * 1.6)
                diff = self.target_gpu - cur_dark
                over = torch.where(cur_dark > self.target_gpu, self.over_mult, 1.0)
                err = torch.mean((diff * diff) * self.imp_gpu * over, dim=0)
                scores = (1.0 - err).detach().cpu().tolist()
                return [float(s) for s in scores]
        except Exception as exc:
            logging.warning(f"Ошибка пакетной оценки CUDA ({exc}), переключение на CPU.")
            self.available = False
            self._m_t_buf = None
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
        if (
            to_canonical_metric(str(config.get("structural_metric", ""))) == "Преобразование расстояний"
            and SCIPY_AVAILABLE
            and distance_transform_edt is not None
        ):
            if np.any(target_edges_2d > 40):
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
                self.distance_transform_map = np.ones_like(target_edges_2d, dtype=np.float32)
                structural_term = edge_norm
        else:
            structural_term = edge_norm

        self.importance_flat = (importance_2d.flatten() + structural_term * contour_boost).astype(np.float32)

        self.line_alpha = self.canvas_manager.get_line_alpha(config.get("lines", 1400), config.get("line_weight", 72))
        self.fearless_penalty = config.get("fearless_factor", 94) / 100.0
        self.empty_error = float(np.sum((self.target_darkness_flat * self.target_darkness_flat) * self.importance_flat))
        self.empty_fitness = 1.0 - self.empty_error / float(max(1, len(self.target_darkness_flat)))

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
            return self.empty_fitness
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
        line_alpha: Optional[float] = None,
    ) -> List[int]:
        """
        Точный Beer-Lambert CSR OMP с автоматическим определением необходимого числа нитей:
        - Если цвет/плотность отсутствует (ни одна хорда не уменьшает ошибку), возвращает [] (0 нитей).
        - Останавливается сразу при исчерпании полезного вклада (без принудительного минимума)
          и обрезает путь до префикса с минимальной точной ошибкой Бера-Ламберта.
        """
        max_l = max(0, int(num_lines))
        lang = normalize_lang(self.config.get("language", "ru"))
        disp_label = translate_layer_name(layer_label, lang)
        if (
            max_l == 0
            or self.num_pins == 0
            or self.canvas_manager.line_matrix is None
            or self.canvas_manager.line_matrix.shape[0] == 0
        ):
            msg_zero = (
                f"Layer ({disp_label}): no threads needed (0 threads)"
                if lang == "en"
                else f"Слой ({disp_label}): нити не требуются (0 нитей)"
            )
            update_callback(msg_zero, [])
            return []

        A = self.canvas_manager.line_matrix
        pin_sub_matrices = self.canvas_manager.ensure_pin_sub_matrices()
        n_chords = A.shape[0]
        if line_alpha is None:
            ref_lines = int(self.config.get("lines", max_l))
            alpha = self.canvas_manager.get_line_alpha(max(400, ref_lines), self.config.get("line_weight", 72))
        else:
            alpha = float(line_alpha)
        fearless = self.config.get("fearless_factor", 94) / 100.0
        radon_inf = (self.config.get("radon_influence", 35) / 100.0) * 0.22

        chord_lengths = np.asarray(A.indptr[1:] - A.indptr[:-1], dtype=np.float32)
        chord_lengths_safe = np.maximum(1.0, chord_lengths)
        inv_norm_denom = (1.0 / (chord_lengths_safe ** 0.85)).astype(np.float32)

        radon_vec: Optional[np.ndarray] = None
        if self.radon_guide is not None and len(self.radon_guide.chord_weights) == n_chords and radon_inf > 0:
            radon_vec = (1.0 + radon_inf * self.radon_guide.chord_weights).astype(np.float32)

        # Предвычисляем массивы по каждому гвоздю один раз, убирая fancy-indexing внутри цикла OMP
        pin_neighbor_pins = self.canvas_manager.pin_neighbor_pins
        pin_neighbor_lids = self.canvas_manager.pin_neighbor_lids
        pin_inv_norm = [inv_norm_denom[lids] for lids in pin_neighbor_lids]
        pin_radon_mult = [radon_vec[lids] for lids in pin_neighbor_lids] if radon_vec is not None else None

        optical_depth = np.zeros_like(self.target_darkness_flat, dtype=np.float32)
        pixel_gain = compute_exact_beer_lambert_gain(
            optical_depth, self.target_darkness_flat, self.importance_flat, alpha, fearless
        )
        init_err = float(np.sum((self.target_darkness_flat * self.target_darkness_flat) * self.importance_flat))

        # Проверяем, улучшает ли ХОТЯ БЫ ОДНА хорда точность относительно пустого холста
        init_raw = A.dot(pixel_gain)
        if float(np.max(init_raw)) <= 1e-5:
            logging.info(f"CSR OMP ({layer_label}): цвет/сигнал отсутствует — 0 нитей.")
            msg_empty = (
                f"Layer ({disp_label}): color absent (0 threads)"
                if lang == "en"
                else f"Слой ({disp_label}): цвет отсутствует (0 нитей)"
            )
            update_callback(msg_empty, [])
            return []

        init_scores = init_raw * inv_norm_denom
        if radon_vec is not None:
            pos_init = init_scores > 0.0
            if np.any(pos_init):
                init_scores[pos_init] *= radon_vec[pos_init]
        best_lid_init = int(np.argmax(init_scores))
        start_pin = self.canvas_manager.line_keys[best_lid_init][0]

        path = [start_pin]
        current_pin = start_pin
        current_err = init_err
        best_err = init_err
        best_len = 0

        used_lids = np.zeros(n_chords, dtype=bool)
        recent_pins: deque = deque()
        recent_counts = np.zeros(self.num_pins, dtype=np.int32)
        neg_streak = 0
        a_indices = A.indices
        a_indptr = A.indptr

        t0 = time.perf_counter()
        last_ui = t0

        for i in range(max_l):
            if self.stop_check and self.stop_check():
                break

            cand_pins = pin_neighbor_pins[current_pin]
            cand_lids = pin_neighbor_lids[current_pin]
            if len(cand_lids) == 0:
                break

            mask = ~used_lids[cand_lids]
            if recent_pins:
                mask_rec = mask & (recent_counts[cand_pins] == 0)
                if np.any(mask_rec):
                    mask = mask_rec

            if not np.any(mask):
                break

            # Нулевое копирование CSR: перемножаем преднарезанную подматрицу pin_sub_matrices[current_pin]
            raw_scores = pin_sub_matrices[current_pin].dot(pixel_gain)
            scores = raw_scores * pin_inv_norm[current_pin]

            if pin_radon_mult is not None:
                pos_s = scores > 0.0
                if np.any(pos_s):
                    scores[pos_s] *= pin_radon_mult[current_pin][pos_s]

            scores[~mask] = -np.inf
            best_idx = int(np.argmax(scores))
            best_raw = float(raw_scores[best_idx])
            best_next = int(cand_pins[best_idx])
            best_lid = int(cand_lids[best_idx])

            # Небольшой допуск (до 8 ходов) позволяет перейти между раздельными пятнами цвета;
            # если полезные нити не нашлись — останавливаемся и откатываем холостой хвост.
            if best_raw <= 1e-5:
                neg_streak += 1
                if neg_streak >= 8:
                    logging.info(f"OMP Авто-стоп ({layer_label}) на шаге {i}: оптимум достигнут без перенасыщения.")
                    break
            else:
                neg_streak = 0

            best_pixels = a_indices[a_indptr[best_lid] : a_indptr[best_lid + 1]]
            if len(best_pixels) > 0:
                current_err -= best_raw
                optical_depth[best_pixels] += alpha
                pixel_gain[best_pixels] = compute_exact_beer_lambert_gain(
                    optical_depth[best_pixels],
                    self.target_darkness_flat[best_pixels],
                    self.importance_flat[best_pixels],
                    alpha,
                    fearless,
                )

            used_lids[best_lid] = True
            path.append(best_next)

            if current_err < best_err - 1e-6:
                best_err = current_err
                best_len = len(path)

            recent_pins.append(current_pin)
            recent_counts[current_pin] += 1
            if len(recent_pins) > 6:
                old_pin = recent_pins.popleft()
                recent_counts[old_pin] -= 1
            current_pin = best_next

            now = time.perf_counter()
            if (i < 30 or now - last_ui > 0.06 or i == max_l - 1) and i > 0:
                prog = int(((i + 1) / max(1, max_l)) * 100)
                rate = (i + 1) / max(0.001, now - t0)
                msg_prog = (
                    f"Stage 1: CSR OMP ({disp_label}) {prog}% ({len(path) - 1} threads, {rate:.0f} l/s)"
                    if lang == "en"
                    else f"Этап 1: CSR OMP ({disp_label}) {prog}% ({len(path) - 1} нитей, {rate:.0f} лин/с)"
                )
                update_callback(msg_prog, list(path))
                last_ui = now

        # Откатываем возможные отрицательные переходные хорды в хвосте до глобально лучшего префикса
        if best_len < 2:
            path = []
        else:
            path = path[:best_len]

        dt = time.perf_counter() - t0
        n_res = max(0, len(path) - 1)
        logging.info(f"CSR OMP завершен ({layer_label}): {n_res} нитей (макс. {max_l}) за {dt:.2f} сек.")
        msg_done = (
            f"OMP synthesis complete ({disp_label}: {n_res} threads)"
            if lang == "en"
            else f"OMP-синтез завершен ({disp_label}: {n_res} нитей)"
        )
        update_callback(msg_done, list(path))
        return path


def optimize_thread_count_delta(
    chromosome: List[int],
    fitness_calculator: FitnessCalculator,
    max_lines: int,
    radon_guide: Optional[RadonGuide] = None,
    max_prune_passes: int = 2,
    allow_add: bool = True,
) -> Tuple[List[int], float, int, int]:
    """
    Двунаправленный точный O(L) алгоритм адаптации числа нитей:
    1) Уменьшает число нитей при их избытке (схлопывание A->B->C -> A->C и удаление краевых хорд,
       вплоть до 0 нитей, если пустой слой точнее).
    2) Добавляет недостающие нити при их нехватке (точное Beer-Lambert наращивание строго <= max_lines).
    Возвращает (новый_путь, новый_фитнес, удалено_нитей, добавлено_нитей).
    """
    cm = fitness_calculator.canvas_manager
    cm.build_line_tables()
    num_pins = cm.num_pins
    min_pin_dist = compute_min_pin_distance(num_pins)
    alpha = fitness_calculator.line_alpha
    fearless = fitness_calculator.fearless_penalty
    target_dark = fitness_calculator.target_darkness_flat
    importance = fitness_calculator.importance_flat
    flat_size = float(len(target_dark))
    max_l = max(0, int(max_lines))

    def _pdist(a: int, b: int) -> int:
        d = abs(a - b)
        return min(d, num_pins - d)

    path, loops_removed = prune_chromosome_loops(list(chromosome))
    removed_count = loops_removed
    added_count = 0

    # Жестко ограничиваем сверху, если исходный путь длиннее нового max_lines
    if len(path) - 1 > max_l:
        removed_count += (len(path) - 1) - max_l
        path = path[: max_l + 1] if max_l > 0 else []

    optical_depth = cm.compute_optical_depth_1d(path, alpha)
    total_err = fitness_calculator.calculate_total_error(optical_depth)
    scratch = np.zeros(len(target_dark), dtype=np.int8)
    empty_arr = np.empty(0, dtype=np.int32)

    # ФАЗА 1: Удаление избыточных нитей (уменьшение длины пути при перенасыщении)
    for _ in range(max(1, int(max_prune_passes))):
        pass_removed = 0

        # Удаляем вредные краевые хорды с конца и начала
        while len(path) >= 2:
            rem_tail = cm.get_line_pixel_indices(path[-2], path[-1])
            d_err, af, nd = compute_delta_remove_pin_np(
                optical_depth, target_dark, importance, rem_tail, empty_arr, empty_arr, alpha, fearless, scratch=scratch
            )
            if d_err < -1e-6 and af is not None and nd is not None:
                optical_depth[af] = nd
                total_err += d_err
                path.pop()
                removed_count += 1
                pass_removed += 1
            else:
                break

        while len(path) >= 2:
            rem_head = cm.get_line_pixel_indices(path[0], path[1])
            d_err, af, nd = compute_delta_remove_pin_np(
                optical_depth, target_dark, importance, rem_head, empty_arr, empty_arr, alpha, fearless, scratch=scratch
            )
            if d_err < -1e-6 and af is not None and nd is not None:
                optical_depth[af] = nd
                total_err += d_err
                path.pop(0)
                removed_count += 1
                pass_removed += 1
            else:
                break

        # Сканируем внутренние гвозди: схлопывание A->B->C в A->C
        idx = 1
        while idx < len(path) - 1:
            prev_p, cur_p, next_p = path[idx - 1], path[idx], path[idx + 1]
            if prev_p == next_p:
                rem1 = cm.get_line_pixel_indices(prev_p, cur_p)
                rem2 = cm.get_line_pixel_indices(cur_p, next_p)
                d_err, af, nd = compute_delta_remove_pin_np(
                    optical_depth, target_dark, importance, rem1, rem2, empty_arr, alpha, fearless, scratch=scratch
                )
                if d_err <= 1e-6:
                    if af is not None and nd is not None:
                        optical_depth[af] = nd
                    total_err += d_err
                    del path[idx : idx + 2]
                    removed_count += 2
                    pass_removed += 2
                    continue
            elif _pdist(prev_p, next_p) >= min_pin_dist:
                rem1 = cm.get_line_pixel_indices(prev_p, cur_p)
                rem2 = cm.get_line_pixel_indices(cur_p, next_p)
                bridge = cm.get_line_pixel_indices(prev_p, next_p)
                d_err, af, nd = compute_delta_remove_pin_np(
                    optical_depth, target_dark, importance, rem1, rem2, bridge, alpha, fearless, scratch=scratch
                )
                if d_err < -1e-6 and af is not None and nd is not None:
                    optical_depth[af] = nd
                    total_err += d_err
                    del path[idx]
                    removed_count += 1
                    pass_removed += 1
                    continue
            idx += 1

        if pass_removed == 0:
            break

    # Если пустой слой лучше или равен оставшемуся пути — полностью очищаем слой (0 нитей)
    if len(path) < 2 or total_err >= fitness_calculator.empty_error - 1e-6:
        removed_count += max(0, len(path) - 1)
        path = []
        optical_depth.fill(0.0)
        total_err = fitness_calculator.empty_error

    # ФАЗА 2: Добавление недостающих нитей при дефиците (строго не больше max_lines)
    if allow_add and max_l > 0 and max(0, len(path) - 1) < max_l and cm.line_matrix is not None and cm.line_matrix.shape[0] > 0:
        A = cm.line_matrix
        pin_sub_matrices = cm.ensure_pin_sub_matrices()
        n_chords = A.shape[0]
        radon_inf = (fitness_calculator.config.get("radon_influence", 35) / 100.0) * 0.22
        chord_lengths = np.maximum(1.0, (A.indptr[1:] - A.indptr[:-1]).astype(np.float32))
        inv_norm = (1.0 / (chord_lengths ** 0.85)).astype(np.float32)
        pin_inv_norm = [inv_norm[lids] for lids in cm.pin_neighbor_lids]
        radon_vec = (
            (1.0 + radon_inf * radon_guide.chord_weights).astype(np.float32)
            if (radon_guide is not None and len(radon_guide.chord_weights) == n_chords and radon_inf > 0)
            else None
        )
        pin_radon_mult = [radon_vec[lids] for lids in cm.pin_neighbor_lids] if radon_vec is not None else None

        pixel_gain = compute_exact_beer_lambert_gain(optical_depth, target_dark, importance, alpha, fearless)
        used_lids = np.zeros(n_chords, dtype=bool)
        if len(path) >= 2:
            existing_lids = cm.chromosome_to_lids(path)
            if len(existing_lids) > 0:
                used_lids[existing_lids] = True

        a_indices = A.indices
        a_indptr = A.indptr

        # Если путь был пуст, проверяем, нужна ли хотя бы одна стартовая хорда
        if len(path) < 2:
            init_raw = A.dot(pixel_gain)
            if float(np.max(init_raw)) > 1e-5:
                init_scores = init_raw * inv_norm
                if radon_vec is not None:
                    pos_i = init_scores > 0.0
                    if np.any(pos_i):
                        init_scores[pos_i] *= radon_vec[pos_i]
                best_lid0 = int(np.argmax(init_scores))
                if float(init_raw[best_lid0]) > 1e-5:
                    p_a, p_b = cm.line_keys[best_lid0]
                    pix = a_indices[a_indptr[best_lid0] : a_indptr[best_lid0 + 1]]
                    total_err -= float(init_raw[best_lid0])
                    optical_depth[pix] += alpha
                    pixel_gain[pix] = compute_exact_beer_lambert_gain(
                        optical_depth[pix], target_dark[pix], importance[pix], alpha, fearless
                    )
                    used_lids[best_lid0] = True
                    path = [p_a, p_b]
                    added_count += 1

        if len(path) >= 2:
            recent_pins: deque = deque(path[-6:])
            recent_counts = np.zeros(num_pins, dtype=np.int32)
            for rp in recent_pins:
                if 0 <= rp < num_pins:
                    recent_counts[rp] += 1
            current_pin = path[-1]

            while len(path) - 1 < max_l:
                cand_pins = cm.pin_neighbor_pins[current_pin]
                cand_lids = cm.pin_neighbor_lids[current_pin]
                if len(cand_lids) == 0:
                    break
                mask = ~used_lids[cand_lids]
                if recent_pins:
                    mask_rec = mask & (recent_counts[cand_pins] == 0)
                    if np.any(mask_rec):
                        mask = mask_rec
                if not np.any(mask):
                    break

                raw = pin_sub_matrices[current_pin].dot(pixel_gain)
                scores = raw * pin_inv_norm[current_pin]
                if pin_radon_mult is not None:
                    pos_s = scores > 0.0
                    if np.any(pos_s):
                        scores[pos_s] *= pin_radon_mult[current_pin][pos_s]
                scores[~mask] = -np.inf
                bi = int(np.argmax(scores))
                b_raw = float(raw[bi])
                if b_raw <= 1e-5:
                    break

                b_next = int(cand_pins[bi])
                b_lid = int(cand_lids[bi])
                pix = a_indices[a_indptr[b_lid] : a_indptr[b_lid + 1]]
                total_err -= b_raw
                optical_depth[pix] += alpha
                pixel_gain[pix] = compute_exact_beer_lambert_gain(
                    optical_depth[pix], target_dark[pix], importance[pix], alpha, fearless
                )
                used_lids[b_lid] = True
                path.append(b_next)
                added_count += 1

                recent_pins.append(current_pin)
                recent_counts[current_pin] += 1
                if len(recent_pins) > 6:
                    old_p = recent_pins.popleft()
                    recent_counts[old_p] -= 1
                current_pin = b_next

    return path, 1.0 - (total_err / flat_size), removed_count, added_count


# ----------------------------------------------------------------------
# ОСТРОВНОЙ ГЕНЕТИЧЕСКИЙ АЛГОРИТМ + ВЕКТОРИЗОВАННАЯ ДОВОДКА
# ----------------------------------------------------------------------


class IslandThread(threading.Thread):
    """
    Поток острова эволюционной доводки переменной длины:
    - 6 графовых мутаторов (Smart Hinge, Radon Bridge, 2-Opt Invert, Jitter Pin, Prune Excess, Insert Thread)
    - Кроссовер хромосом переменной длины с проверкой стыков min_pin_dist
    - Быстрый O(L) дельта-поиск (замена, удаление избыточных нитей и вставка недостающих в пределах max_lines).
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
        run_id: int = 0,
        max_lines: Optional[int] = None,
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
        self.run_id = int(run_id)
        self.max_lines = max(0, int(max_lines if max_lines is not None else config.get("lines", 1400)))
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
            "Prune Excess": self._prune_excess_mutation,
            "Insert Thread": self._insert_thread_mutation,
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

    def _safe_interior_range(self, c: List[int], s: int, e: int) -> Optional[Tuple[int, int]]:
        if len(c) < 3:
            return None
        hi = min(len(c) - 2, max(1, e - 1))
        lo = max(1, min(s, hi))
        if lo > hi:
            return None
        return lo, hi

    def _smart_hinge_mutation(self, c: List[int], s: int, e: int) -> List[int]:
        rng = self._safe_interior_range(c, s, e)
        if rng is None:
            return c
        idx = random.randint(rng[0], rng[1])
        prev_p, next_p = c[idx - 1], c[idx + 1]
        num_pins = self.fitness_calculator.canvas_manager.num_pins
        for _ in range(10):
            cand = random.randint(0, num_pins - 1)
            if self._pin_dist(cand, prev_p, num_pins) >= self.min_pin_dist and self._pin_dist(cand, next_p, num_pins) >= self.min_pin_dist:
                c[idx] = cand
                break
        return c

    def _radon_bridge_mutation(self, c: List[int], s: int, e: int) -> List[int]:
        if self.radon_guide is None or not self.radon_guide.top_neighbors_cache:
            return self._smart_hinge_mutation(c, s, e)
        rng = self._safe_interior_range(c, s, e)
        if rng is None:
            return c
        idx = random.randint(rng[0], rng[1])
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
        if len(c) < 5:
            return c
        num_pins = self.fitness_calculator.canvas_manager.num_pins
        hi1 = min(len(c) - 4, max(1, e - 3))
        lo1 = max(1, min(s, hi1))
        if lo1 > hi1:
            return c
        i1 = random.randint(lo1, hi1)
        hi2 = min(len(c) - 1, max(i1 + 2, e), i1 + 25)
        if i1 + 2 > hi2:
            return c
        i2 = random.randint(i1 + 2, hi2)
        # Проверяем, не создаст ли инверсия слишком короткие граничные хорды
        if self._pin_dist(c[i1 - 1], c[i2 - 1], num_pins) < self.min_pin_dist:
            return c
        if i2 < len(c) and self._pin_dist(c[i1], c[i2], num_pins) < self.min_pin_dist:
            return c
        c[i1:i2] = reversed(c[i1:i2])
        return c

    def _jitter_pin_mutation(self, c: List[int], s: int, e: int) -> List[int]:
        rng = self._safe_interior_range(c, s, e)
        if rng is None:
            return c
        idx = random.randint(rng[0], rng[1])
        prev_p, next_p = c[idx - 1], c[idx + 1]
        num_pins = self.fitness_calculator.canvas_manager.num_pins
        for shift in random.sample([-3, -2, -1, 1, 2, 3], 6):
            cand = (c[idx] + shift) % num_pins
            if self._pin_dist(cand, prev_p, num_pins) >= self.min_pin_dist and self._pin_dist(cand, next_p, num_pins) >= self.min_pin_dist:
                c[idx] = cand
                break
        return c

    def _prune_excess_mutation(self, c: List[int], s: int, e: int) -> List[int]:
        """Уменьшает длину хромосомы на 1 нить (удаляет избыточный гвоздь при допустимом стыке A->C)."""
        if len(c) <= 2:
            return [] if len(c) <= 2 else c
        num_pins = self.fitness_calculator.canvas_manager.num_pins
        # Иногда удаляем крайнюю хорду в хвосте
        if random.random() < 0.25:
            c.pop()
            return c
        rng = self._safe_interior_range(c, s, e)
        if rng is None:
            c.pop()
            return c
        for _ in range(8):
            idx = random.randint(rng[0], rng[1])
            if self._pin_dist(c[idx - 1], c[idx + 1], num_pins) >= self.min_pin_dist:
                del c[idx]
                return c
        c.pop()
        return c

    def _insert_thread_mutation(self, c: List[int], s: int, e: int) -> List[int]:
        """Увеличивает длину хромосомы на 1 нить (вставка A->X->C или достройка хвоста), строго <= max_lines."""
        if len(c) - 1 >= self.max_lines or self.max_lines <= 0:
            return self._smart_hinge_mutation(c, s, e)
        num_pins = self.fitness_calculator.canvas_manager.num_pins
        if len(c) < 2:
            a = random.randint(0, num_pins - 1)
            b = (a + max(self.min_pin_dist, num_pins // 3)) % num_pins
            return [a, b]

        # В 30% случаев достраиваем новый шаг в хвосте пути
        if random.random() < 0.30:
            last_p = c[-1]
            cands = (
                self.radon_guide.top_neighbors_cache[last_p]
                if (self.radon_guide and self.radon_guide.top_neighbors_cache)
                else np.empty(0, dtype=np.int32)
            )
            if len(cands) > 0:
                cand = int(cands[random.randint(0, min(11, len(cands) - 1))])
                if self._pin_dist(last_p, cand, num_pins) >= self.min_pin_dist:
                    c.append(cand)
                    return c
            for _ in range(8):
                cand = random.randint(0, num_pins - 1)
                if self._pin_dist(last_p, cand, num_pins) >= self.min_pin_dist:
                    c.append(cand)
                    return c
            return c

        # Разбиваем существующую хорду c[idx] -> c[idx+1] на c[idx] -> cand -> c[idx+1]
        hi = min(len(c) - 2, max(0, e - 1))
        lo = max(0, min(s, hi))
        idx = random.randint(lo, hi)
        a, b = c[idx], c[idx + 1]
        if self.radon_guide and self.radon_guide.top_neighbors_cache:
            cands = self.radon_guide.top_neighbors_cache[a]
            for cand in cands[:12]:
                cand_i = int(cand)
                if self._pin_dist(cand_i, a, num_pins) >= self.min_pin_dist and self._pin_dist(cand_i, b, num_pins) >= self.min_pin_dist:
                    c.insert(idx + 1, cand_i)
                    return c
        for _ in range(10):
            cand = random.randint(0, num_pins - 1)
            if self._pin_dist(cand, a, num_pins) >= self.min_pin_dist and self._pin_dist(cand, b, num_pins) >= self.min_pin_dist:
                c.insert(idx + 1, cand)
                return c
        return c

    def _graph_splice_crossover(self, p1: List[int], p2: List[int], start: int, end: int) -> List[int]:
        """Кроссовер хромосом (включая переменную длину) с проверкой стыков min_pin_dist и лимита max_lines."""
        n1, n2 = len(p1), len(p2)
        if n1 < 4 or n2 < 4 or end - start < 4:
            return list(p1)
        num_pins = self.fitness_calculator.canvas_manager.num_pins
        n_min = min(n1, n2)
        lo = max(1, min(start + 1, n_min - 3))
        hi = min(end - 1, n_min - 2)
        if lo >= hi:
            return list(p1)

        for _ in range(8):
            cut = random.randint(lo, hi)
            if self._pin_dist(p1[cut - 1], p2[cut], num_pins) < self.min_pin_dist:
                continue
            cut2_p1 = min(end, n1)
            cut2_p2 = min(end, n2)
            if cut >= cut2_p2:
                continue
            if cut2_p1 < n1 and self._pin_dist(p2[cut2_p2 - 1], p1[cut2_p1], num_pins) < self.min_pin_dist:
                if cut + 1 < min(cut2_p1, cut2_p2):
                    alt = random.randint(cut + 1, min(cut2_p1, cut2_p2) - 1)
                    if self._pin_dist(p2[alt - 1], p1[alt], num_pins) < self.min_pin_dist:
                        continue
                    cut2_p1 = alt
                    cut2_p2 = alt
                else:
                    continue
            child = p1[:cut] + p2[cut:cut2_p2] + p1[cut2_p1:]
            if len(child) - 1 <= self.max_lines:
                return child

        return list(p1)

    def _fast_delta_coordinate_descent(
        self, chromosome: List[int], start: int, end: int, depth: int
    ) -> Tuple[List[int], float]:
        """
        Сверхбыстрая локальная полировка переменной длины за счет O(L) дельта-буфера (scratch):
        на каждом шаге оценивает замену гвоздя (Radon + локальный сдвиг), удаление лишней нити
        и вставку недостающей нити (строго <= max_lines).
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

        scratch = np.zeros(len(target_dark), dtype=np.int8)
        has_radon = bool(self.radon_guide and self.radon_guide.top_neighbors_cache)
        shifts = (-4, -3, -2, -1, 1, 2, 3, 4)

        for step_i in range(depth):
            if self.stop_event.is_set() or len(best_c) < 3:
                break
            hi_idx = min(len(best_c) - 2, max(1, end - 1))
            lo_idx = max(1, min(start, hi_idx))
            if lo_idx > hi_idx:
                break

            idx = random.randint(lo_idx, hi_idx)
            prev_p = best_c[idx - 1]
            cur_p = best_c[idx]
            next_p = best_c[idx + 1]

            # Формируем до 2 разных кандидатов для замены (1 от Радона + 1 локальный сдвиг)
            cand_list: List[int] = []
            if has_radon:
                cands = self.radon_guide.top_neighbors_cache[prev_p]
                if len(cands) > 0:
                    c_rad = int(cands[random.randint(0, min(11, len(cands) - 1))])
                    if (
                        c_rad != cur_p
                        and self._pin_dist(c_rad, prev_p, num_pins) >= self.min_pin_dist
                        and self._pin_dist(c_rad, next_p, num_pins) >= self.min_pin_dist
                    ):
                        cand_list.append(c_rad)

            c_jit = (cur_p + random.choice(shifts)) % num_pins
            if (
                c_jit != cur_p
                and c_jit not in cand_list
                and self._pin_dist(c_jit, prev_p, num_pins) >= self.min_pin_dist
                and self._pin_dist(c_jit, next_p, num_pins) >= self.min_pin_dist
            ):
                cand_list.append(c_jit)

            rem1 = cm.get_line_pixel_indices(prev_p, cur_p)
            rem2 = cm.get_line_pixel_indices(cur_p, next_p)

            best_delta = -1e-7
            best_move: Optional[Tuple[str, int, Optional[int], np.ndarray, np.ndarray]] = None

            # 1) Замена гвоздя A->B->C -> A->X->C
            for cand_p in cand_list:
                add1 = cm.get_line_pixel_indices(prev_p, cand_p)
                add2 = cm.get_line_pixel_indices(cand_p, next_p)
                delta_err, af, nd = compute_delta_hinge_np(
                    optical_depth,
                    target_dark,
                    importance,
                    rem1,
                    rem2,
                    add1,
                    add2,
                    line_alpha,
                    fearless,
                    scratch=scratch,
                )
                if delta_err < best_delta and af is not None and nd is not None:
                    best_delta = delta_err
                    best_move = ("replace", idx, cand_p, af, nd)

            # 2) Проверка удаления избыточной нити A->B->C -> A->C
            if len(best_c) > 2 and self._pin_dist(prev_p, next_p, num_pins) >= self.min_pin_dist:
                bridge = cm.get_line_pixel_indices(prev_p, next_p)
                d_rem, af_r, nd_r = compute_delta_remove_pin_np(
                    optical_depth,
                    target_dark,
                    importance,
                    rem1,
                    rem2,
                    bridge,
                    line_alpha,
                    fearless,
                    scratch=scratch,
                )
                if d_rem < best_delta and af_r is not None and nd_r is not None:
                    best_delta = d_rem
                    best_move = ("remove", idx, None, af_r, nd_r)

            # 3) Проверка вставки недостающей нити A->B -> A->X->B (строго <= max_lines)
            if (step_i & 1) == 0 and (len(best_c) - 1) < self.max_lines:
                ins_cand: Optional[int] = None
                if has_radon:
                    cands_i = self.radon_guide.top_neighbors_cache[prev_p]
                    if len(cands_i) > 0:
                        ci = int(cands_i[random.randint(0, min(11, len(cands_i) - 1))])
                        if self._pin_dist(ci, prev_p, num_pins) >= self.min_pin_dist and self._pin_dist(ci, cur_p, num_pins) >= self.min_pin_dist:
                            ins_cand = ci
                if ins_cand is None:
                    ci = random.randint(0, num_pins - 1)
                    if self._pin_dist(ci, prev_p, num_pins) >= self.min_pin_dist and self._pin_dist(ci, cur_p, num_pins) >= self.min_pin_dist:
                        ins_cand = ci
                if ins_cand is not None:
                    add_i1 = cm.get_line_pixel_indices(prev_p, ins_cand)
                    add_i2 = cm.get_line_pixel_indices(ins_cand, cur_p)
                    d_ins, af_i, nd_i = compute_delta_insert_pin_np(
                        optical_depth,
                        target_dark,
                        importance,
                        rem1,
                        add_i1,
                        add_i2,
                        line_alpha,
                        fearless,
                        scratch=scratch,
                    )
                    if d_ins < best_delta and af_i is not None and nd_i is not None:
                        best_delta = d_ins
                        best_move = ("insert", idx, ins_cand, af_i, nd_i)

            if best_move is not None:
                op, pos, val, af, nd = best_move
                optical_depth[af] = nd
                total_err += best_delta
                if op == "replace" and val is not None:
                    best_c[pos] = val
                elif op == "remove":
                    del best_c[pos]
                elif op == "insert" and val is not None:
                    best_c.insert(pos, val)

        return best_c, 1.0 - (total_err / flat_size)

    def run(self) -> None:
        try:
            init_fit = self.fitness_calculator.calculate_single_fitness(self.base_chromosome)
            self.best_on_island = {"chromosome": list(self.base_chromosome), "fitness": init_fit}
            pop_size = max(4, self.config.get("population_size", 24))
            self.population = [{"chromosome": list(self.base_chromosome), "fitness": init_fit} for _ in range(pop_size)]

            if self.island_id > 0 and len(self.base_chromosome) >= 3:
                mutated = [
                    self._smart_hinge_mutation(list(self.base_chromosome), 1, len(self.base_chromosome) - 1)
                    for _ in range(1, pop_size)
                ]
                scores = self.fitness_calculator.calculate_fitness_scores(mutated)
                for k, (ch, sc) in enumerate(zip(mutated, scores), start=1):
                    self.population[k] = {"chromosome": ch, "fitness": sc}
                self.population.sort(key=lambda x: x["fitness"], reverse=True)

            self._send_island_update()

            stages = self.stages_override if self.stages_override is not None else max(1, self.config.get("stages", 5))

            for stage in range(stages):
                if self.stop_event.is_set():
                    break
                cur_lines = max(2, len(self.best_on_island["chromosome"]) - 1)
                lines_per_stage = max(1, cur_lines // stages)
                start_line = stage * lines_per_stage
                end_line = cur_lines if stage == stages - 1 else min((stage + 1) * lines_per_stage, cur_lines)
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
                "run_id": self.run_id,
                "data": {
                    "id": self.island_id,
                    "layer_idx": self.layer_idx,
                    "chromosome": list(self.best_on_island["chromosome"]),
                    "fitness": self.best_on_island["fitness"],
                    "mut_stats": {k: dict(v) for k, v in self.mutation_stats.items()},
                },
            })

    def _send_generation_tick(self, stage: int, patience: int, limit: int) -> None:
        self.status_queue.put({
            "type": "generation_tick",
            "run_id": self.run_id,
            "data": (self.island_id, stage, patience, limit),
        })


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
        run_id: int = 0,
        max_lines: Optional[int] = None,
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
        self.run_id = int(run_id)
        self.max_lines = max(0, int(max_lines if max_lines is not None else config.get("lines", 1400)))
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
            lang = normalize_lang(self.config.get("language", "ru"))
            disp_layer = translate_layer_name(self.layer_name, lang)
            st_msg = (
                f"Stage 2: Evolutionary polishing ({disp_layer}, max {self.max_lines}) on {num_islands} islands..."
                if lang == "en"
                else f"Этап 2: Эволюционная полировка ({disp_layer}, макс. {self.max_lines}) на {num_islands} островах..."
            )
            self.status_queue.put({
                "type": "status",
                "run_id": self.run_id,
                "data": st_msg,
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
                    run_id=self.run_id,
                    max_lines=self.max_lines,
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
                    self.status_queue.put({"type": "migration", "run_id": self.run_id})
                    migrants = [isl.get_best_individual() for isl in self.islands]
                    if all(m is not None for m in migrants):
                        for i, isl in enumerate(self.islands):
                            isl.accept_migrant(migrants[(i + 1) % len(self.islands)])

            for island in self.islands:
                island.join(timeout=1.5)

            candidates = [isl.get_best_individual() for isl in self.islands if isl.get_best_individual() is not None]
            if candidates:
                best = max(candidates, key=lambda x: x["fitness"])
                if not self.stop_event.is_set():
                    opt_path, opt_fit, n_rem, n_add = optimize_thread_count_delta(
                        best["chromosome"],
                        self.fitness_calculator,
                        self.max_lines,
                        radon_guide=self.radon_guide,
                        max_prune_passes=2,
                        allow_add=True,
                    )
                    if opt_fit >= best["fitness"] - 1e-7:
                        best = {"chromosome": opt_path, "fitness": opt_fit}
                        if n_rem > 0 or n_add > 0:
                            logging.info(
                                f"Адаптация числа нитей ({self.layer_name}): -{n_rem} лишних, +{n_add} доб. -> {max(0, len(opt_path) - 1)} нитей"
                            )
                            self.status_queue.put({
                                "type": "island_update",
                                "run_id": self.run_id,
                                "data": {
                                    "id": 0,
                                    "layer_idx": self.layer_idx,
                                    "chromosome": list(opt_path),
                                    "fitness": opt_fit,
                                    "mut_stats": {},
                                    "force_update": True,
                                },
                            })
                self.best_result = best

            if self.emit_done:
                if lang == "en":
                    msg = "Stopped by user." if self.stop_event.is_set() else "Generation and polishing complete!"
                else:
                    msg = "Остановлено пользователем." if self.stop_event.is_set() else "Генерация и полировка успешно завершены!"
                self.status_queue.put({"type": "done", "run_id": self.run_id, "data": msg})
        except Exception:
            logging.error("Ошибка в IslandManagerThread", exc_info=True)
            err_msg = (
                "Error in island manager."
                if normalize_lang(self.config.get("language", "ru")) == "en"
                else "Ошибка в диспетчере островов."
            )
            self.status_queue.put({"type": "error", "run_id": self.run_id, "data": err_msg})
