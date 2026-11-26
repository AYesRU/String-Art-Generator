# -*- coding: utf-8 -*-
"""
Геометрия холста, векторизованное построение разреженной таблицы хорд (DDA CSR),
кэширование GPU-матриц и оптический аддитивный рендеринг нитей по закону Бера-Ламберта
с сохранением контраста при любом разрешении (от превью до Ultra-HD 2400px+).
Поддерживает светлую доску (черная нить), темную доску (белая нить) и 4-слойный CMYK.
"""

import logging
import math
import threading
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import scipy.sparse as sp
from PIL import Image

from . import IMAGE_SIZE
from .i18n import to_canonical_shape

# Идеальные краски штатных слоев CMYK (ключи — имена слоев в нижнем регистре, RU и EN)
_CMYK_INK_RGB: Dict[str, Tuple[float, float, float]] = {
    "циан (cyan)": (0.0, 1.0, 1.0), "cyan": (0.0, 1.0, 1.0),
    "маджента (magenta)": (1.0, 0.0, 1.0), "magenta": (1.0, 0.0, 1.0),
    "желтый (yellow)": (1.0, 1.0, 0.0), "yellow": (1.0, 1.0, 0.0),
    "черный (black)": (0.0, 0.0, 0.0), "black": (0.0, 0.0, 0.0),
    "черная нить": (0.0, 0.0, 0.0), "black thread": (0.0, 0.0, 0.0),
}

try:
    import torch

    _TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None
    _TORCH_CUDA_AVAILABLE = False


def compute_min_pin_distance(num_pins: int) -> int:
    """Адаптивный минимальный шаг между гвоздями по периметру."""
    return max(8, min(20, max(1, num_pins // 14)))


class _LazyPinSubMatrix:
    """Легковесная обертка подматрицы CSR для высокого числа гвоздей (>600) без дублирования памяти."""

    def __init__(self, matrix: sp.csr_matrix, row_indices: np.ndarray):
        self._matrix = matrix
        self._row_indices = row_indices

    def dot(self, vec: np.ndarray) -> np.ndarray:
        if len(self._row_indices) == 0:
            return np.empty(0, dtype=np.float32)
        return self._matrix[self._row_indices].dot(vec)


class CanvasManager:
    """
    Менеджер геометрии рамы (6 форм), разреженной матрицы пикселей хорд (CSR),
    GPU-кэша и физически согласованного рендера нитей.
    """

    def __init__(
        self,
        size: int = IMAGE_SIZE,
        num_pins: int = 260,
        shape: str = "Круг",
        physical_cm: float = 50.0,
    ):
        self.size = int(size)
        self.num_pins = int(num_pins)
        self.shape = to_canonical_shape(shape)
        self.physical_cm = float(physical_cm)
        self.pin_coords = self._calculate_pin_coords()

        # Разреженные и плоские структуры хорд
        self.line_matrix: Optional[sp.csr_matrix] = None
        self.line_dict: Dict[Tuple[int, int], int] = {}
        self.line_keys: List[Tuple[int, int]] = []
        self.line_lengths_px: np.ndarray = np.empty(0, dtype=np.float32)
        self.line_lengths_m: np.ndarray = np.empty(0, dtype=np.float32)
        self.pin_candidate_line_ids: List[List[Tuple[int, int]]] = []
        self.pin_neighbor_pins: List[np.ndarray] = []
        self.pin_neighbor_lids: List[np.ndarray] = []
        self.pin_sub_matrices: Optional[List[sp.csr_matrix]] = None

        self.flat_ready: bool = False
        self.line_indices_flat: np.ndarray = np.empty(0, dtype=np.int32)
        self.line_offsets: np.ndarray = np.zeros(self.num_pins * self.num_pins, dtype=np.int64)
        self.line_lengths: np.ndarray = np.zeros(self.num_pins * self.num_pins, dtype=np.int32)
        self.min_pin_distance: int = compute_min_pin_distance(self.num_pins)

        # Кэш общих тензоров на CUDA GPU (переиспользуется между слоями CMYK и доводкой)
        self._gpu_lock = threading.Lock()
        self._gpu_a_t_csr: Optional[Any] = None
        self._gpu_pin_coords: Optional[Any] = None
        self._gpu_render_enabled: bool = _TORCH_CUDA_AVAILABLE

    def mm_per_px(self) -> float:
        """Единый физический масштаб (мм на пиксель) для метража, ЧПУ G-Code и шаблона 1:1."""
        return (self.physical_cm * 10.0) / max(1.0, float(self.size))

    def m_per_px(self) -> float:
        """Единый физический масштаб (метры на пиксель) для расчета длины нити."""
        return self.mm_per_px() / 1000.0

    def _calculate_pin_coords(self) -> np.ndarray:
        coords: List[Tuple[float, float]] = []
        margin = self.size * 0.035
        usable = self.size - 2.0 * margin
        cx = cy = self.size / 2.0

        if self.shape == "Квадрат":
            side = usable
            perim = 4.0 * side
            step = perim / max(1, self.num_pins)
            for i in range(self.num_pins):
                d = (i * step) % perim
                if d <= side:
                    x, y = margin + d, margin
                elif d <= 2.0 * side:
                    x, y = self.size - margin, margin + (d - side)
                elif d <= 3.0 * side:
                    x, y = (self.size - margin) - (d - 2.0 * side), self.size - margin
                else:
                    x, y = margin, (self.size - margin) - (d - 3.0 * side)
                coords.append((x, y))

        elif "Прямоугольник" in self.shape:
            aspect = 4.0 / 3.0 if "4:3" in self.shape else (16.0 / 9.0 if "16:9" in self.shape else 4.0 / 3.0)
            w = usable
            h = usable / aspect
            start_y = (self.size - h) / 2.0
            perim = 2.0 * (w + h)
            step = perim / max(1, self.num_pins)
            for i in range(self.num_pins):
                d = (i * step) % perim
                if d <= w:
                    x, y = margin + d, start_y
                elif d <= w + h:
                    x, y = margin + w, start_y + (d - w)
                elif d <= 2.0 * w + h:
                    x, y = margin + w - (d - w - h), start_y + h
                else:
                    x, y = margin, start_y + h - (d - 2.0 * w - h)
                coords.append((x, y))

        elif self.shape == "Овал":
            rx = usable / 2.0
            ry = rx * 0.78
            for i in range(self.num_pins):
                angle = (i / max(1, self.num_pins)) * 2.0 * math.pi - math.pi / 2.0
                coords.append((cx + rx * math.cos(angle), cy + ry * math.sin(angle)))

        elif self.shape == "Шестиугольник":
            r = usable / 2.0
            vertices = [
                (
                    cx + r * math.cos((k / 6.0) * 2.0 * math.pi - math.pi / 2.0),
                    cy + r * math.sin((k / 6.0) * 2.0 * math.pi - math.pi / 2.0),
                )
                for k in range(6)
            ]
            vertices.append(vertices[0])
            pins_per_side = float(self.num_pins) / 6.0
            for side in range(6):
                p_start = vertices[side]
                p_end = vertices[side + 1]
                count = max(1, int(round((side + 1) * pins_per_side)) - int(round(side * pins_per_side)))
                for j in range(count):
                    t = j / float(count)
                    coords.append((
                        p_start[0] + t * (p_end[0] - p_start[0]),
                        p_start[1] + t * (p_end[1] - p_start[1]),
                    ))
            while len(coords) < self.num_pins:
                coords.append(coords[-1])
            coords = coords[: self.num_pins]

        else:  # 'Круг'
            radius = usable / 2.0
            for i in range(self.num_pins):
                angle = (i / max(1, self.num_pins)) * 2.0 * math.pi - math.pi / 2.0
                coords.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))

        return np.array(coords, dtype=np.float32)

    def get_polygon_mask(self, dilate_px: int = 2) -> np.ndarray:
        """Возвращает бинарную маску внутренней области рамы для любой из 6 форм."""
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        if len(self.pin_coords) >= 3:
            pts = np.round(self.pin_coords).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
            if dilate_px > 0:
                k = dilate_px * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                mask = cv2.dilate(mask, kernel)
        return mask

    def get_content_bbox(self) -> Tuple[int, int, int, int]:
        """Возвращает (x0, y0, x1, y1) ограничивающего прямоугольника области гвоздей."""
        if len(self.pin_coords) == 0:
            return (0, 0, self.size, self.size)
        x0 = max(0, int(math.floor(float(np.min(self.pin_coords[:, 0])))))
        y0 = max(0, int(math.floor(float(np.min(self.pin_coords[:, 1])))))
        x1 = min(self.size, int(math.ceil(float(np.max(self.pin_coords[:, 0])))))
        y1 = min(self.size, int(math.ceil(float(np.max(self.pin_coords[:, 1])))))
        return (x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))

    def build_flat_line_table(self, min_distance: Optional[int] = None) -> None:
        """Псевдоним для совместимости со старым API — строит CSR и плоскую таблицу хорд."""
        self.build_line_tables(min_distance=min_distance)

    def build_line_tables(self, min_distance: Optional[int] = None) -> None:
        """
        Полностью векторизованно строит 8-связную разреженную матрицу хорд SciPy CSR
        методом целочисленного DDA без сортировок np.unique (0 дубликатов пикселей в хорде).
        """
        if self.flat_ready and self.line_matrix is not None:
            return

        t0 = time.perf_counter()
        min_dist = self.min_pin_distance if min_distance is None else int(min_distance)
        self.min_pin_distance = min_dist

        n_pixels = self.size * self.size
        m_per_pixel = self.m_per_px()

        self.line_dict.clear()
        self.line_keys.clear()
        self.pin_candidate_line_ids = [[] for _ in range(self.num_pins)]
        self.pin_sub_matrices = None
        self.line_offsets.fill(0)
        self.line_lengths.fill(0)

        ii, jj = np.triu_indices(self.num_pins, k=1)
        dists = np.minimum(jj - ii, self.num_pins - (jj - ii))
        valid_pairs = dists >= min_dist
        ii = ii[valid_pairs].astype(np.int32)
        jj = jj[valid_pairs].astype(np.int32)

        ix_pins = np.clip(np.round(self.pin_coords[:, 0]).astype(np.int32), 0, self.size - 1)
        iy_pins = np.clip(np.round(self.pin_coords[:, 1]).astype(np.int32), 0, self.size - 1)

        dx_i = ix_pins[jj] - ix_pins[ii]
        dy_i = iy_pins[jj] - iy_pins[ii]
        steps_all = np.maximum(np.abs(dx_i), np.abs(dy_i)) + 1
        keep_len = steps_all >= 2
        ii = ii[keep_len]
        jj = jj[keep_len]
        dx_i = dx_i[keep_len]
        dy_i = dy_i[keep_len]
        steps_all = steps_all[keep_len].astype(np.int32)

        n_chords = len(ii)
        if n_chords > 0:
            # Батчевая растеризация по 12000 хорд для экономного пикового потребления памяти
            idx_chunks: List[np.ndarray] = []
            chunk_sz = 12000
            for start in range(0, n_chords, chunk_sz):
                end = min(n_chords, start + chunk_sz)
                st_c = steps_all[start:end]
                tot_c = int(np.sum(st_c))
                offs_c = np.zeros(len(st_c), dtype=np.int64)
                if len(st_c) > 1:
                    np.cumsum(st_c[:-1], out=offs_c[1:])

                c_ids = np.repeat(np.arange(len(st_c), dtype=np.int32), st_c)
                loc_k = np.arange(tot_c, dtype=np.float32) - np.repeat(offs_c, st_c)
                den = np.repeat(np.maximum(1, st_c - 1), st_c).astype(np.float32)
                t = loc_k / den

                x0_c = ix_pins[ii[start:end]].astype(np.float32)
                y0_c = iy_pins[ii[start:end]].astype(np.float32)
                dx_c = dx_i[start:end].astype(np.float32)
                dy_c = dy_i[start:end].astype(np.float32)

                xs = np.round(x0_c[c_ids] + dx_c[c_ids] * t).astype(np.int32)
                ys = np.round(y0_c[c_ids] + dy_c[c_ids] * t).astype(np.int32)
                idx_chunks.append((ys * self.size + xs).astype(np.int32, copy=False))

            indices = idx_chunks[0] if len(idx_chunks) == 1 else np.concatenate(idx_chunks)
            indptr = np.zeros(n_chords + 1, dtype=np.int64)
            np.cumsum(steps_all.astype(np.int64), out=indptr[1:])
            data = np.ones(len(indices), dtype=np.float32)

            self.line_matrix = sp.csr_matrix((data, indices, indptr), shape=(n_chords, n_pixels), dtype=np.float32)
            self.line_indices_flat = self.line_matrix.indices

            # Векторизованное заполнение плоских таблиц смещений
            ii64 = ii.astype(np.int64)
            jj64 = jj.astype(np.int64)
            idx_a = ii64 * self.num_pins + jj64
            idx_b = jj64 * self.num_pins + ii64
            self.line_offsets[idx_a] = indptr[:-1]
            self.line_offsets[idx_b] = indptr[:-1]
            self.line_lengths[idx_a] = steps_all
            self.line_lengths[idx_b] = steps_all

            p1_xy = self.pin_coords[ii]
            p2_xy = self.pin_coords[jj]
            self.line_lengths_px = np.hypot(p2_xy[:, 0] - p1_xy[:, 0], p2_xy[:, 1] - p1_xy[:, 1]).astype(np.float32)
            self.line_lengths_m = self.line_lengths_px * m_per_pixel

            ii_list = ii.tolist()
            jj_list = jj.tolist()
            self.line_keys = list(zip(ii_list, jj_list))
            line_dict = self.line_dict
            cands = self.pin_candidate_line_ids
            for lid, (a, b) in enumerate(self.line_keys):
                line_dict[(a, b)] = lid
                line_dict[(b, a)] = lid
                cands[a].append((b, lid))
                cands[b].append((a, lid))
        else:
            self.line_matrix = sp.csr_matrix((0, n_pixels), dtype=np.float32)
            self.line_indices_flat = np.empty(0, dtype=np.int32)
            self.line_lengths_px = np.empty(0, dtype=np.float32)
            self.line_lengths_m = np.empty(0, dtype=np.float32)

        self.pin_neighbor_pins = [
            np.asarray([p for p, _ in cands], dtype=np.int32) for cands in self.pin_candidate_line_ids
        ]
        self.pin_neighbor_lids = [
            np.asarray([l for _, l in cands], dtype=np.int32) for cands in self.pin_candidate_line_ids
        ]
        self.flat_ready = True

        dt = time.perf_counter() - t0
        logging.info(f"Таблица хорд DDA CSR построена: {n_chords} хорд за {dt:.3f} сек.")

    def ensure_pin_sub_matrices(self) -> List[Any]:
        """
        Преднарезает подматрицы CSR для каждого гвоздя один раз (при num_pins <= 600),
        а при > 600 гвоздей использует ленивую нарезку без дублирования гигабайтов памяти.
        """
        self.build_line_tables()
        if self.pin_sub_matrices is None and self.line_matrix is not None:
            t0 = time.perf_counter()
            A = self.line_matrix
            if self.num_pins <= 600:
                self.pin_sub_matrices = [
                    A[lids] if len(lids) > 0 else sp.csr_matrix((0, self.size * self.size), dtype=np.float32)
                    for lids in self.pin_neighbor_lids
                ]
            else:
                self.pin_sub_matrices = [_LazyPinSubMatrix(A, lids) for lids in self.pin_neighbor_lids]
            logging.info(f"Преднарезка pin_sub_matrices выполнена за {(time.perf_counter() - t0) * 1000:.1f} мс.")
        return self.pin_sub_matrices or []

    def get_gpu_transposed_matrix(self) -> Optional[Any]:
        """Возвращает кэшированную транспонированную CSR-матрицу на CUDA GPU (1 загрузка на геометрию)."""
        if not _TORCH_CUDA_AVAILABLE or torch is None:
            return None
        self.build_line_tables()
        if self.line_matrix is None or self.line_matrix.shape[0] == 0:
            return None
        with self._gpu_lock:
            if self._gpu_a_t_csr is not None:
                return self._gpu_a_t_csr
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    if hasattr(torch.sparse, "check_sparse_tensor_invariants"):
                        torch.sparse.check_sparse_tensor_invariants(False)
                    device = torch.device("cuda")
                    a_t = self.line_matrix.T.tocsr()
                    self._gpu_a_t_csr = torch.sparse_csr_tensor(
                        torch.from_numpy(a_t.indptr.astype(np.int64)).to(device),
                        torch.from_numpy(a_t.indices.astype(np.int64)).to(device),
                        torch.from_numpy(a_t.data.astype(np.float32)).to(device),
                        size=a_t.shape,
                    )
                return self._gpu_a_t_csr
            except Exception as exc:
                logging.warning(f"Не удалось кэшировать CSR на GPU: {exc}")
                self._gpu_a_t_csr = None
                return None

    def release_gpu_cache(self) -> None:
        """Освобождает кэшированные GPU-тензоры при смене геометрии или завершении работы."""
        with self._gpu_lock:
            self._gpu_a_t_csr = None
            self._gpu_pin_coords = None
            if _TORCH_CUDA_AVAILABLE and torch is not None:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def get_line_pixel_indices(self, p1_idx: int, p2_idx: int) -> np.ndarray:
        """Возвращает массив уникальных плоских индексов пикселей для хорды p1_idx -> p2_idx."""
        if p1_idx == p2_idx:
            return np.empty(0, dtype=np.int32)
        if self.flat_ready:
            pair_idx = p1_idx * self.num_pins + p2_idx
            length = int(self.line_lengths[pair_idx])
            if length > 0:
                off = int(self.line_offsets[pair_idx])
                return self.line_indices_flat[off : off + length]

        # Запасной точный DDA-расчет для коротких хорд вне min_distance
        x1 = int(np.clip(round(float(self.pin_coords[p1_idx, 0])), 0, self.size - 1))
        y1 = int(np.clip(round(float(self.pin_coords[p1_idx, 1])), 0, self.size - 1))
        x2 = int(np.clip(round(float(self.pin_coords[p2_idx, 0])), 0, self.size - 1))
        y2 = int(np.clip(round(float(self.pin_coords[p2_idx, 1])), 0, self.size - 1))
        steps = max(abs(x2 - x1), abs(y2 - y1)) + 1
        if steps < 2:
            return np.empty(0, dtype=np.int32)
        t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        xs = np.round(x1 + (x2 - x1) * t).astype(np.int32)
        ys = np.round(y1 + (y2 - y1) * t).astype(np.int32)
        return (ys * self.size + xs).astype(np.int32, copy=False)

    @staticmethod
    def get_line_alpha(num_lines: int, line_weight_percent: int = 72) -> float:
        """Оптическая плотность одной нити (единая калибровка для оптимизатора и рендера)."""
        base = (line_weight_percent / 100.0) * 0.165
        line_scale = min(1.30, max(0.42, (1200.0 / max(400, int(num_lines))) ** 0.55))
        return float(np.clip(base * line_scale, 0.045, 0.26))

    def chromosome_to_lids(self, chromosome: List[int]) -> np.ndarray:
        """Преобразует путь гвоздей в массив индексов хорд CSR (игнорирует невалидные переходы)."""
        if not chromosome or len(chromosome) < 2 or not self.line_dict:
            return np.empty(0, dtype=np.int32)
        ld = self.line_dict
        lids = [lid for a, b in zip(chromosome[:-1], chromosome[1:]) if (lid := ld.get((a, b))) is not None]
        return np.asarray(lids, dtype=np.int32)

    def compute_optical_depth_1d(self, chromosome: List[int], line_alpha: float) -> np.ndarray:
        """Быстрое вычисление 1D карты оптической глубины через плоскую таблицу и bincount."""
        flat_size = self.size * self.size
        if not chromosome or len(chromosome) < 2:
            return np.zeros(flat_size, dtype=np.float32)

        if self.flat_ready:
            arr = np.asarray(chromosome, dtype=np.int64)
            p1 = arr[:-1]
            p2 = arr[1:]
            valid = (p1 != p2) & (p1 >= 0) & (p1 < self.num_pins) & (p2 >= 0) & (p2 < self.num_pins)
            p1 = p1[valid]
            p2 = p2[valid]
            pair_idx = p1 * self.num_pins + p2
            offs = self.line_offsets[pair_idx]
            lens = self.line_lengths[pair_idx]
            has = lens > 0
            if np.any(has):
                flat = self.line_indices_flat
                slices = [flat[int(o) : int(o) + int(l)] for o, l in zip(offs[has], lens[has])]
                cat = np.concatenate(slices)
                return np.bincount(cat, minlength=flat_size).astype(np.float32) * float(line_alpha)
            return np.zeros(flat_size, dtype=np.float32)

        depth = np.zeros(flat_size, dtype=np.float32)
        for i in range(len(chromosome) - 1):
            pts = self.get_line_pixel_indices(chromosome[i], chromosome[i + 1])
            if len(pts) > 0:
                depth[pts] += float(line_alpha)
        return depth

    def calculate_path_length_m(self, chromosome: Optional[List[int]]) -> float:
        """Точная длина пути нити в метрах с учетом физического размера холста."""
        if not chromosome or len(chromosome) < 2:
            return 0.0
        idx = np.asarray(chromosome, dtype=np.int64)
        valid = (idx >= 0) & (idx < self.num_pins)
        if not np.all(valid):
            idx = idx[valid]
            if len(idx) < 2:
                return 0.0
        diffs = self.pin_coords[idx[1:]] - self.pin_coords[idx[:-1]]
        total_px = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
        return total_px * self.m_per_px()

    def _accumulate_path_depth_gpu(
        self,
        p1_idx: np.ndarray,
        p2_idx: np.ndarray,
        out_dim: int,
        alpha_500: float,
    ) -> Optional[np.ndarray]:
        """Ускоренный аддитивный билинейный рендер на PyTorch CUDA (~3x быстрее CPU)."""
        if not self._gpu_render_enabled or torch is None:
            return None
        try:
            with self._gpu_lock:
                dev = torch.device("cuda")
                if self._gpu_pin_coords is None:
                    self._gpu_pin_coords = torch.from_numpy(self.pin_coords).to(dev)

                scale = out_dim / float(self.size)
                coords_scaled = self._gpu_pin_coords * scale
                alpha_out = alpha_500 * scale
                depth = torch.zeros(out_dim * out_dim, device=dev, dtype=torch.float32)

                chunk_size = 1200
                for start in range(0, len(p1_idx), chunk_size):
                    i1 = torch.from_numpy(p1_idx[start : start + chunk_size]).to(dev)
                    i2 = torch.from_numpy(p2_idx[start : start + chunk_size]).to(dev)
                    p1s = coords_scaled[i1]
                    p2s = coords_scaled[i2]
                    dx = p2s[:, 0] - p1s[:, 0]
                    dy = p2s[:, 1] - p1s[:, 1]
                    lens = torch.hypot(dx, dy)
                    keep = lens > 0.5
                    if not bool(torch.any(keep).item()):
                        continue
                    p1s, dx, dy, lens = p1s[keep], dx[keep], dy[keep], lens[keep]

                    n_samples = torch.clamp(torch.ceil(lens * 1.25).to(torch.int64), min=2)
                    tot = int(n_samples.sum().item())
                    seg_ids = torch.repeat_interleave(torch.arange(len(lens), device=dev), n_samples)
                    offs = torch.cumsum(n_samples, dim=0) - n_samples
                    local_idx = torch.arange(tot, device=dev, dtype=torch.float32) - torch.repeat_interleave(offs, n_samples).to(torch.float32)
                    denom = torch.repeat_interleave(torch.clamp(n_samples - 1, min=1), n_samples).to(torch.float32)
                    t = local_idx / denom

                    xs = p1s[seg_ids, 0] + dx[seg_ids] * t
                    ys = p1s[seg_ids, 1] + dy[seg_ids] * t
                    step_w = (lens / n_samples.to(torch.float32))[seg_ids] * alpha_out

                    x0 = torch.floor(xs).to(torch.int64)
                    y0 = torch.floor(ys).to(torch.int64)
                    fx = xs - x0.to(torch.float32)
                    fy = ys - y0.to(torch.float32)

                    for ox, oy, wxy in (
                        (0, 0, (1.0 - fx) * (1.0 - fy)),
                        (1, 0, fx * (1.0 - fy)),
                        (0, 1, (1.0 - fx) * fy),
                        (1, 1, fx * fy),
                    ):
                        xx = x0 + ox
                        yy = y0 + oy
                        m = (xx >= 0) & (xx < out_dim) & (yy >= 0) & (yy < out_dim)
                        if bool(torch.any(m).item()):
                            depth.index_add_(0, yy[m] * out_dim + xx[m], wxy[m] * step_w[m])

                return depth.view(out_dim, out_dim).detach().cpu().numpy()
        except Exception as exc:
            logging.warning(f"Откат рендера с GPU на CPU: {exc}")
            self._gpu_render_enabled = False
            return None

    def _accumulate_path_depth(
        self,
        path: List[int],
        draw_count: int,
        out_dim: int,
        thickness: int,
        alpha_500: float,
    ) -> np.ndarray:
        """
        Векторизованно накапливает истинную аддитивную оптическую глубину
        с субпиксельным билинейным антиалиасингом (GPU CUDA с откатом на CPU).
        """
        if draw_count <= 0 or len(path) < 2:
            return np.zeros((out_dim, out_dim), dtype=np.float32)

        arr = np.asarray(path[: draw_count + 1], dtype=np.int64)
        valid_mask = (
            (arr[:-1] != arr[1:])
            & (arr[:-1] >= 0)
            & (arr[:-1] < self.num_pins)
            & (arr[1:] >= 0)
            & (arr[1:] < self.num_pins)
        )
        if not np.any(valid_mask):
            return np.zeros((out_dim, out_dim), dtype=np.float32)

        p1_idx = arr[:-1][valid_mask]
        p2_idx = arr[1:][valid_mask]

        depth_2d = None
        if self._gpu_render_enabled and len(p1_idx) >= 120:
            depth_2d = self._accumulate_path_depth_gpu(p1_idx, p2_idx, out_dim, alpha_500)

        if depth_2d is None:
            depth = np.zeros(out_dim * out_dim, dtype=np.float32)
            scale = out_dim / float(self.size)
            coords_scaled = self.pin_coords * scale
            alpha_out = alpha_500 * scale

            p1_all = coords_scaled[p1_idx]
            p2_all = coords_scaled[p2_idx]

            chunk_size = 800
            for start in range(0, len(p1_all), chunk_size):
                p1s = p1_all[start : start + chunk_size]
                p2s = p2_all[start : start + chunk_size]
                dx = p2s[:, 0] - p1s[:, 0]
                dy = p2s[:, 1] - p1s[:, 1]
                lens = np.hypot(dx, dy)
                keep = lens > 0.5
                if not np.any(keep):
                    continue
                p1s, dx, dy, lens = p1s[keep], dx[keep], dy[keep], lens[keep]

                n_samples = np.maximum(2, np.ceil(lens * 1.25).astype(np.int32))
                total_samples = int(np.sum(n_samples))
                seg_ids = np.repeat(np.arange(len(lens), dtype=np.int32), n_samples)
                offsets = np.zeros(len(lens), dtype=np.int64)
                if len(lens) > 1:
                    np.cumsum(n_samples[:-1], out=offsets[1:])
                local_idx = np.arange(total_samples, dtype=np.float32) - np.repeat(offsets, n_samples)
                denom = np.repeat(np.maximum(1, n_samples - 1), n_samples).astype(np.float32)
                t = local_idx / denom

                xs = p1s[seg_ids, 0] + dx[seg_ids] * t
                ys = p1s[seg_ids, 1] + dy[seg_ids] * t
                step_w = (lens / n_samples)[seg_ids] * alpha_out

                x0 = np.floor(xs).astype(np.int32)
                y0 = np.floor(ys).astype(np.int32)
                fx = xs - x0
                fy = ys - y0

                for ox, oy, wxy in (
                    (0, 0, (1.0 - fx) * (1.0 - fy)),
                    (1, 0, fx * (1.0 - fy)),
                    (0, 1, (1.0 - fx) * fy),
                    (1, 1, fx * fy),
                ):
                    xx = x0 + ox
                    yy = y0 + oy
                    m = (xx >= 0) & (xx < out_dim) & (yy >= 0) & (yy < out_dim)
                    if np.any(m):
                        idx = yy[m] * out_dim + xx[m]
                        depth += np.bincount(
                            idx, weights=wxy[m] * step_w[m], minlength=out_dim * out_dim
                        ).astype(np.float32)
            depth_2d = depth.reshape(out_dim, out_dim)

        # Масштабируем профиль толщины нити пропорционально разрешению (сохраняя интеграл)
        ref_scale = out_dim / float(IMAGE_SIZE)
        sigma = max(0.0, 0.38 * ref_scale * float(thickness) - 0.38)
        if sigma > 0.15:
            depth_2d = cv2.GaussianBlur(depth_2d, (0, 0), sigmaX=sigma, sigmaY=sigma)
        return depth_2d

    def render_layers_to_numpy(
        self,
        layers: List[Dict[str, Any]],
        thickness: int = 1,
        line_weight_percent: int = 72,
        max_step: Optional[int] = None,
        output_size: Optional[int] = None,
        active_layer_idx: int = 0,
        active_chord: Optional[Tuple[int, int]] = None,
        invert_board: bool = False,
    ) -> np.ndarray:
        """
        Рендерит один (Монохром / Белая нить на темном) или несколько слоев (CMYK) в RGB uint8
        с истинным законом Бера-Ламберта, опциональной неоновой подсветкой активного шага и гвоздями.
        """
        out_dim = int(output_size) if output_size is not None else self.size
        bg_val = 18 if invert_board else 255
        if not layers or all(len(l.get("path", [])) < 2 for l in layers):
            return np.full((out_dim, out_dim, 3), bg_val, dtype=np.uint8)

        is_color = len(layers) > 1

        if not is_color:
            path = layers[0].get("path", [])
            total_lines = max(0, len(path) - 1)
            draw_count = total_lines if max_step is None else min(total_lines, max(0, int(max_step)))
            ref_n = max(400, int(layers[0].get("ref_lines") or total_lines))
            alpha_500 = self.get_line_alpha(ref_n, line_weight_percent)
            depth_2d = self._accumulate_path_depth(path, draw_count, out_dim, thickness, alpha_500)
            if invert_board:
                intensity = np.clip(1.0 - np.exp(-depth_2d * 1.6), 0.0, 1.0)
                gray_u8 = np.clip(18.0 + intensity * 237.0, 0.0, 255.0).astype(np.uint8)
            else:
                transmittance = np.clip(np.exp(-depth_2d * 1.6), 0.0, 1.0)
                gray_u8 = (transmittance * 255.0).astype(np.uint8)
            result_rgb = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB)
        else:
            # Многоканальный цветной рендеринг: честная субтрактивная модель поглощения (Beer-Lambert)
            # Корректно отображает как стандартный CMYK, так и любые произвольные цвета нитей из каталога!
            d_r = np.zeros((out_dim, out_dim), dtype=np.float32)
            d_g = np.zeros((out_dim, out_dim), dtype=np.float32)
            d_b = np.zeros((out_dim, out_dim), dtype=np.float32)

            rem_global = None if max_step is None else max(0, int(max_step))
            for l_idx, layer in enumerate(layers):
                path = layer.get("path", [])
                total_lines = max(0, len(path) - 1)
                # active_layer_idx < 0 означает режим «Все цвета / Все слои» с общим скроллингом шагов
                if active_layer_idx < 0:
                    if rem_global is None:
                        draw_count = total_lines
                    else:
                        draw_count = min(total_lines, rem_global)
                        rem_global -= draw_count
                elif max_step is None:
                    draw_count = total_lines
                elif l_idx < active_layer_idx:
                    draw_count = total_lines
                elif l_idx == active_layer_idx:
                    draw_count = min(total_lines, max(0, int(max_step)))
                else:
                    draw_count = 0

                if draw_count <= 0:
                    continue

                ref_n = max(400, int(layer.get("ref_lines") or total_lines))
                alpha_500 = self.get_line_alpha(ref_n, line_weight_percent)
                d_layer = self._accumulate_path_depth(path, draw_count, out_dim, thickness, alpha_500)

                # Определение RGB цвета нити в диапазоне [0.0, 1.0]
                hex_color = str(layer.get("color_hex", "#111111")).lstrip("#")
                name = str(layer.get("name", "")).strip().lower()
                # Идеальные краски подставляются ТОЛЬКО для штатных слоев CMYK (точное совпадение имени).
                # Цвета каталога («Темный желтовато-коричневый», «Коричневато-черный» …) рисуются строго по HEX.
                cmyk_ink = _CMYK_INK_RGB.get(name)
                if cmyk_ink is not None:
                    r_c, g_c, b_c = cmyk_ink
                elif len(hex_color) == 6:
                    try:
                        r_c = int(hex_color[0:2], 16) / 255.0
                        g_c = int(hex_color[2:4], 16) / 255.0
                        b_c = int(hex_color[4:6], 16) / 255.0
                    except ValueError:
                        r_c, g_c, b_c = 0.0, 0.0, 0.0
                else:
                    r_c, g_c, b_c = 0.0, 0.0, 0.0

                # Светлая доска: нить поглощает дополнительный цвет (1 - r_c).
                # Темная доска (invert_board=True): нить отражает свой собственный цвет (r_c, g_c, b_c).
                if invert_board:
                    d_r += d_layer * r_c
                    d_g += d_layer * g_c
                    d_b += d_layer * b_c
                else:
                    d_r += d_layer * (1.0 - r_c)
                    d_g += d_layer * (1.0 - g_c)
                    d_b += d_layer * (1.0 - b_c)

            if invert_board:
                r_u8 = np.clip(18.0 + (1.0 - np.exp(-d_r * 1.6)) * 237.0, 0.0, 255.0).astype(np.uint8)
                g_u8 = np.clip(18.0 + (1.0 - np.exp(-d_g * 1.6)) * 237.0, 0.0, 255.0).astype(np.uint8)
                b_u8 = np.clip(18.0 + (1.0 - np.exp(-d_b * 1.6)) * 237.0, 0.0, 255.0).astype(np.uint8)
            else:
                r_u8 = np.clip(np.exp(-d_r * 1.6) * 255.0, 0.0, 255.0).astype(np.uint8)
                g_u8 = np.clip(np.exp(-d_g * 1.6) * 255.0, 0.0, 255.0).astype(np.uint8)
                b_u8 = np.clip(np.exp(-d_b * 1.6) * 255.0, 0.0, 255.0).astype(np.uint8)
            result_rgb = cv2.merge([r_u8, g_u8, b_u8])

        pin_scale = out_dim / float(self.size)

        if active_chord is not None:
            p1_idx, p2_idx = active_chord
            if 0 <= p1_idx < self.num_pins and 0 <= p2_idx < self.num_pins:
                p1_scr = (
                    int(round(float(self.pin_coords[p1_idx, 0]) * pin_scale)),
                    int(round(float(self.pin_coords[p1_idx, 1]) * pin_scale)),
                )
                p2_scr = (
                    int(round(float(self.pin_coords[p2_idx, 0]) * pin_scale)),
                    int(round(float(self.pin_coords[p2_idx, 1]) * pin_scale)),
                )
                glow_w = max(2, int(round(out_dim / 220)))
                cv2.line(result_rgb, p1_scr, p2_scr, (0, 210, 255), thickness=glow_w + 1, lineType=cv2.LINE_AA)
                cv2.line(result_rgb, p1_scr, p2_scr, (9, 105, 218), thickness=max(1, glow_w - 1), lineType=cv2.LINE_AA)
                cv2.circle(result_rgb, p1_scr, max(3, int(round(out_dim / 140))), (220, 38, 38), -1, lineType=cv2.LINE_AA)
                cv2.circle(result_rgb, p2_scr, max(4, int(round(out_dim / 120))), (0, 220, 180), -1, lineType=cv2.LINE_AA)

        pin_color = (185, 190, 200) if invert_board else (60, 60, 65)
        pin_r = max(1, int(round(out_dim / 340)))
        for px, py in self.pin_coords:
            cx_i = int(round(float(px) * pin_scale))
            cy_i = int(round(float(py) * pin_scale))
            cv2.circle(result_rgb, (cx_i, cy_i), pin_r, pin_color, -1, lineType=cv2.LINE_AA)

        return result_rgb

    def render_layers_to_pil(
        self,
        layers: List[Dict[str, Any]],
        thickness: int = 1,
        line_weight_percent: int = 72,
        max_step: Optional[int] = None,
        output_size: Optional[int] = None,
        active_layer_idx: int = 0,
        active_chord: Optional[Tuple[int, int]] = None,
        invert_board: bool = False,
    ) -> Image.Image:
        rgb = self.render_layers_to_numpy(
            layers,
            thickness=thickness,
            line_weight_percent=line_weight_percent,
            max_step=max_step,
            output_size=output_size,
            active_layer_idx=active_layer_idx,
            active_chord=active_chord,
            invert_board=invert_board,
        )
        return Image.fromarray(rgb)

    def render_chromosome_to_numpy(
        self,
        chromosome: List[int],
        thickness: int = 1,
        line_weight_percent: int = 72,
        max_step: Optional[int] = None,
        output_size: Optional[int] = None,
        active_chord: Optional[Tuple[int, int]] = None,
        invert_board: bool = False,
    ) -> np.ndarray:
        """Рендер одной хромосомы (обратная совместимость)."""
        rgb = self.render_layers_to_numpy(
            [{"name": "Белая нить" if invert_board else "Черная нить", "color_hex": "#ffffff" if invert_board else "#111111", "path": chromosome}],
            thickness=thickness,
            line_weight_percent=line_weight_percent,
            max_step=max_step,
            output_size=output_size,
            active_chord=active_chord,
            invert_board=invert_board,
        )
        if active_chord is not None:
            return rgb
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def render_chromosome_to_pil(
        self,
        chromosome: List[int],
        thickness: int = 1,
        line_weight_percent: int = 72,
        max_step: Optional[int] = None,
        output_size: Optional[int] = None,
        active_chord: Optional[Tuple[int, int]] = None,
        invert_board: bool = False,
    ) -> Image.Image:
        return self.render_layers_to_pil(
            [{"name": "Белая нить" if invert_board else "Черная нить", "color_hex": "#ffffff" if invert_board else "#111111", "path": chromosome}],
            thickness=thickness,
            line_weight_percent=line_weight_percent,
            max_step=max_step,
            output_size=output_size,
            active_chord=active_chord,
            invert_board=invert_board,
        )
