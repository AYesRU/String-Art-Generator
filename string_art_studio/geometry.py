# -*- coding: utf-8 -*-
"""
Геометрия холста, построение разреженной таблицы хорд (CSR) и оптический
аддитивный рендеринг нитей по закону Бера-Ламберта с сохранением контраста
при любом разрешении (от превью до Ultra-HD 2400px+).
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import scipy.sparse as sp
from PIL import Image

from . import IMAGE_SIZE, MIN_PIN_DISTANCE


def compute_min_pin_distance(num_pins: int) -> int:
    """Адаптивный минимальный шаг между гвоздями по периметру."""
    return max(8, min(20, max(1, num_pins // 14)))


class CanvasManager:
    """
    Менеджер геометрии рамы (6 форм), разреженной матрицы пикселей хорд (CSR)
    и физически согласованного рендера нитей.
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
        self.shape = shape
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

        self.flat_ready: bool = False
        self.line_indices_flat: np.ndarray = np.empty(0, dtype=np.int32)
        self.line_offsets: np.ndarray = np.zeros(self.num_pins * self.num_pins, dtype=np.int64)
        self.line_lengths: np.ndarray = np.zeros(self.num_pins * self.num_pins, dtype=np.int32)
        self.min_pin_distance: int = compute_min_pin_distance(self.num_pins)

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
        Векторизованно строит разреженную матрицу хорд SciPy CSR и плоский индекс
        пикселей без дублирования данных в памяти.
        """
        if self.flat_ready and self.line_matrix is not None:
            return

        t0 = time.perf_counter()
        min_dist = self.min_pin_distance if min_distance is None else int(min_distance)
        self.min_pin_distance = min_dist

        n_pixels = self.size * self.size
        m_per_pixel = (self.physical_cm / 100.0) / max(1.0, float(self.size * 0.93))

        self.line_dict.clear()
        self.line_keys.clear()
        self.pin_candidate_line_ids = [[] for _ in range(self.num_pins)]
        self.line_offsets.fill(0)
        self.line_lengths.fill(0)

        cols_chunks: List[np.ndarray] = []
        lengths_px_list: List[float] = []
        chord_pixel_counts: List[int] = []
        lid = 0

        for i in range(self.num_pins):
            js = np.arange(i + 1, self.num_pins)
            if len(js) == 0:
                continue
            dists = np.minimum(js - i, self.num_pins - (js - i))
            js = js[dists >= min_dist]
            if len(js) == 0:
                continue

            p1 = self.pin_coords[i]
            p2s = self.pin_coords[js]
            chord_lens = np.hypot(p2s[:, 0] - p1[0], p2s[:, 1] - p1[1])
            max_samples = max(2, int(math.ceil(float(np.max(chord_lens)) * 1.25)) + 1)

            ts = np.linspace(0.0, 1.0, max_samples, dtype=np.float32)[None, :]
            xs = np.round(p1[0] + (p2s[:, 0:1] - p1[0]) * ts).astype(np.int32)
            ys = np.round(p1[1] + (p2s[:, 1:2] - p1[1]) * ts).astype(np.int32)
            valid = (xs >= 0) & (xs < self.size) & (ys >= 0) & (ys < self.size)
            flat_all = ys * self.size + xs

            for k, j_raw in enumerate(js):
                j = int(j_raw)
                chord_len = float(chord_lens[k])
                if chord_len < 1.0:
                    continue
                pts = np.unique(flat_all[k][valid[k]])
                if len(pts) == 0:
                    continue

                cols_chunks.append(pts.astype(np.int32, copy=False))
                chord_pixel_counts.append(len(pts))
                lengths_px_list.append(chord_len)

                self.line_keys.append((i, j))
                self.line_dict[(i, j)] = lid
                self.line_dict[(j, i)] = lid
                self.pin_candidate_line_ids[i].append((j, lid))
                self.pin_candidate_line_ids[j].append((i, lid))
                lid += 1

        if lid > 0:
            indices = np.concatenate(cols_chunks).astype(np.int32, copy=False)
            counts_arr = np.asarray(chord_pixel_counts, dtype=np.int64)
            indptr = np.zeros(lid + 1, dtype=np.int64)
            np.cumsum(counts_arr, out=indptr[1:])
            data = np.ones(len(indices), dtype=np.float32)

            self.line_matrix = sp.csr_matrix((data, indices, indptr), shape=(lid, n_pixels), dtype=np.float32)
            self.line_indices_flat = self.line_matrix.indices

            for k_idx, (p1, p2) in enumerate(self.line_keys):
                off = indptr[k_idx]
                cnt = counts_arr[k_idx]
                idx_a = p1 * self.num_pins + p2
                idx_b = p2 * self.num_pins + p1
                self.line_offsets[idx_a] = off
                self.line_offsets[idx_b] = off
                self.line_lengths[idx_a] = cnt
                self.line_lengths[idx_b] = cnt
        else:
            self.line_matrix = sp.csr_matrix((0, n_pixels), dtype=np.float32)
            self.line_indices_flat = np.empty(0, dtype=np.int32)

        self.line_lengths_px = np.asarray(lengths_px_list, dtype=np.float32)
        self.line_lengths_m = self.line_lengths_px * m_per_pixel

        self.pin_neighbor_pins = [
            np.asarray([p for p, _ in cands], dtype=np.int32) for cands in self.pin_candidate_line_ids
        ]
        self.pin_neighbor_lids = [
            np.asarray([l for _, l in cands], dtype=np.int32) for cands in self.pin_candidate_line_ids
        ]
        self.flat_ready = True

        dt = time.perf_counter() - t0
        logging.info(f"Таблица хорд CSR построена: {lid} хорд за {dt:.2f} сек.")

    def get_line_pixel_indices(self, p1_idx: int, p2_idx: int) -> np.ndarray:
        """Возвращает массив плоских индексов пикселей для хорды между гвоздями p1_idx и p2_idx."""
        if p1_idx == p2_idx:
            return np.empty(0, dtype=np.int32)
        if self.flat_ready:
            pair_idx = p1_idx * self.num_pins + p2_idx
            length = int(self.line_lengths[pair_idx])
            if length > 0:
                off = int(self.line_offsets[pair_idx])
                return self.line_indices_flat[off : off + length]

        # Запасной прямой расчет для редких/коротких хорд вне min_distance
        p1 = self.pin_coords[p1_idx]
        p2 = self.pin_coords[p2_idx]
        length = int(math.ceil(float(np.hypot(p2[0] - p1[0], p2[1] - p1[1])) * 1.25))
        if length < 1:
            return np.empty(0, dtype=np.int32)
        ts = np.linspace(0.0, 1.0, max(2, length), dtype=np.float32)
        xs = np.round(p1[0] + (p2[0] - p1[0]) * ts).astype(np.int32)
        ys = np.round(p1[1] + (p2[1] - p1[1]) * ts).astype(np.int32)
        valid = (xs >= 0) & (xs < self.size) & (ys >= 0) & (ys < self.size)
        return np.unique(ys[valid] * self.size + xs[valid]).astype(np.int32, copy=False)

    @staticmethod
    def get_line_alpha(num_lines: int, line_weight_percent: int = 72) -> float:
        """Оптическая плотность одной нити для 1D-оптимизатора (масштабируется под число нитей)."""
        base = (line_weight_percent / 100.0) * 0.22
        line_scale = min(1.25, max(0.75, 1400.0 / max(400, int(num_lines))))
        return float(np.clip(base * line_scale, 0.08, 0.36))

    def chromosome_to_lids(self, chromosome: List[int]) -> np.ndarray:
        """Преобразует путь гвоздей в массив индексов хорд CSR (игнорирует невалидные переходы)."""
        if not chromosome or len(chromosome) < 2 or not self.line_dict:
            return np.empty(0, dtype=np.int32)
        lids = []
        ld = self.line_dict
        for i in range(len(chromosome) - 1):
            lid = ld.get((chromosome[i], chromosome[i + 1]))
            if lid is not None:
                lids.append(lid)
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
            valid = p1 != p2
            p1 = p1[valid]
            p2 = p2[valid]
            pair_idx = p1 * self.num_pins + p2
            offs = self.line_offsets[pair_idx]
            lens = self.line_lengths[pair_idx]
            has = lens > 0
            if np.any(has):
                slices = [self.line_indices_flat[o : o + l] for o, l in zip(offs[has], lens[has])]
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
        m_per_px = (self.physical_cm / 100.0) / max(1.0, float(self.size * 0.93))
        return total_px * m_per_px

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
        с субпиксельным билинейным антиалиасингом и сохранением тона при любом out_dim.
        """
        depth = np.zeros(out_dim * out_dim, dtype=np.float32)
        if draw_count <= 0 or len(path) < 2:
            return depth.reshape(out_dim, out_dim)

        scale = out_dim / float(self.size)
        coords_scaled = self.pin_coords * scale
        alpha_out = alpha_500 * scale

        arr = np.asarray(path[: draw_count + 1], dtype=np.int64)
        valid_mask = (arr[:-1] != arr[1:]) & (arr[:-1] >= 0) & (arr[:-1] < self.num_pins) & (arr[1:] >= 0) & (arr[1:] < self.num_pins)
        if not np.any(valid_mask):
            return depth.reshape(out_dim, out_dim)

        p1_all = coords_scaled[arr[:-1][valid_mask]]
        p2_all = coords_scaled[arr[1:][valid_mask]]

        # Обрабатываем порциями по 800 сегментов для жесткого контроля потребления RAM
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
    ) -> np.ndarray:
        """
        Рендерит один (Монохром) или несколько слоев (CMYK) в RGB uint8 массив
        с истинным законом Бера-Ламберта, опциональной неоновой подсветкой активного
        шага и точками гвоздей по периметру.
        """
        out_dim = int(output_size) if output_size is not None else self.size
        if not layers or sum(len(l.get("path", [])) for l in layers) < 2:
            return np.full((out_dim, out_dim, 3), 255, dtype=np.uint8)

        alpha_500 = float(np.clip((line_weight_percent / 100.0) * 0.22, 0.08, 0.36))
        is_color = len(layers) > 1

        if not is_color:
            path = layers[0].get("path", [])
            total_lines = max(0, len(path) - 1)
            draw_count = total_lines if max_step is None else min(total_lines, max(0, int(max_step)))
            depth_2d = self._accumulate_path_depth(path, draw_count, out_dim, thickness, alpha_500)
            transmittance = np.clip(np.exp(-depth_2d * 1.6), 0.0, 1.0)
            gray_u8 = (transmittance * 255.0).astype(np.uint8)
            result_rgb = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB)
        else:
            c_depth = np.zeros((out_dim, out_dim), dtype=np.float32)
            m_depth = np.zeros((out_dim, out_dim), dtype=np.float32)
            y_depth = np.zeros((out_dim, out_dim), dtype=np.float32)
            k_depth = np.zeros((out_dim, out_dim), dtype=np.float32)

            for l_idx, layer in enumerate(layers):
                path = layer.get("path", [])
                total_lines = max(0, len(path) - 1)
                if max_step is None:
                    draw_count = total_lines
                elif l_idx < active_layer_idx:
                    draw_count = total_lines
                elif l_idx == active_layer_idx:
                    draw_count = min(total_lines, max(0, int(max_step)))
                else:
                    draw_count = 0

                if draw_count <= 0:
                    continue

                d_layer = self._accumulate_path_depth(path, draw_count, out_dim, thickness, alpha_500)
                name = layer.get("name", "").lower()
                if "cyan" in name or "циан" in name or "голуб" in name:
                    c_depth += d_layer
                elif "magenta" in name or "маджент" in name or "пурпур" in name:
                    m_depth += d_layer
                elif "yellow" in name or "желт" in name:
                    y_depth += d_layer
                else:
                    k_depth += d_layer

            tc = np.exp(-c_depth * 1.6)
            tm = np.exp(-m_depth * 1.6)
            ty = np.exp(-y_depth * 1.6)
            tk = np.exp(-k_depth * 1.6)

            r = np.clip(tc * tk * 255.0, 0.0, 255.0).astype(np.uint8)
            g = np.clip(tm * tk * 255.0, 0.0, 255.0).astype(np.uint8)
            b = np.clip(ty * tk * 255.0, 0.0, 255.0).astype(np.uint8)
            result_rgb = cv2.merge([r, g, b])

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

        pin_r = max(1, int(round(out_dim / 340)))
        for px, py in self.pin_coords:
            cx_i = int(round(float(px) * pin_scale))
            cy_i = int(round(float(py) * pin_scale))
            cv2.circle(result_rgb, (cx_i, cy_i), pin_r, (60, 60, 65), -1, lineType=cv2.LINE_AA)

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
    ) -> Image.Image:
        rgb = self.render_layers_to_numpy(
            layers,
            thickness=thickness,
            line_weight_percent=line_weight_percent,
            max_step=max_step,
            output_size=output_size,
            active_layer_idx=active_layer_idx,
            active_chord=active_chord,
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
    ) -> np.ndarray:
        """Рендер одной хромосомы в оттенках серого (обратная совместимость)."""
        rgb = self.render_layers_to_numpy(
            [{"name": "Черная нить", "color_hex": "#111111", "path": chromosome}],
            thickness=thickness,
            line_weight_percent=line_weight_percent,
            max_step=max_step,
            output_size=output_size,
            active_chord=active_chord,
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
    ) -> Image.Image:
        return self.render_layers_to_pil(
            [{"name": "Черная нить", "color_hex": "#111111", "path": chromosome}],
            thickness=thickness,
            line_weight_percent=line_weight_percent,
            max_step=max_step,
            output_size=output_size,
            active_chord=active_chord,
        )
