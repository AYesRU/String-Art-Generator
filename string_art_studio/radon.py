# -*- coding: utf-8 -*-
"""
Томографический гид на базе преобразования Радона с частотным фильтром Рама-Лака
(Ramp Filter + окно Хэмминга) и фильтрованной обратной проекцией (FBP).
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .geometry import CanvasManager, compute_min_pin_distance

try:
    from skimage.transform import iradon, radon

    SKIMAGE_AVAILABLE = True
except ImportError:
    radon = None
    iradon = None
    SKIMAGE_AVAILABLE = False


class RadonGuide:
    """
    Вычисляет фильтрованную синограмму Радона и FBP-реконструкцию, а также
    веса хорд и топ-соседей для каждого гвоздя.
    """

    def __init__(
        self,
        target_image_np: np.ndarray,
        edges_np: np.ndarray,
        canvas_manager: CanvasManager,
        config: Optional[Dict[str, Any]] = None,
        compute_fbp: bool = True,
        density_override: Optional[np.ndarray] = None,
    ):
        self.config = config or {"radon_enabled": True}
        self.canvas_manager = canvas_manager
        self.sinogram: Optional[np.ndarray] = None
        self.fbp_recon: Optional[np.ndarray] = None
        self.line_weights: Dict[Tuple[int, int], float] = {}
        self.chord_weights: np.ndarray = np.empty(0, dtype=np.float32)
        self.pin_adj_weights = np.zeros((canvas_manager.num_pins, canvas_manager.num_pins), dtype=np.float32)
        self.top_neighbors_cache: List[np.ndarray] = []

        if self.config.get("radon_enabled", True):
            self._compute_filtered_sinogram(
                target_image_np,
                edges_np,
                compute_fbp=compute_fbp,
                density_override=density_override,
            )

    @staticmethod
    def _apply_ramp_filter(sinogram: np.ndarray) -> np.ndarray:
        """Применяет фильтр Рама-Лака с окном Хэмминга в частотной области."""
        n_rho, _ = sinogram.shape
        padded_len = max(16, int(2 ** np.ceil(np.log2(max(2, n_rho * 2)))))
        freqs = np.fft.rfftfreq(padded_len).reshape(-1, 1)
        max_f = float(freqs.max()) + 1e-8
        ramp = np.abs(freqs) * (0.54 + 0.46 * np.cos(np.pi * freqs / max_f))
        sino_padded = np.pad(sinogram, ((0, padded_len - n_rho), (0, 0)), mode="constant")
        sino_fft = np.fft.rfft(sino_padded, axis=0)
        filtered = np.fft.irfft(sino_fft * ramp, n=padded_len, axis=0)[:n_rho, :]
        return np.maximum(0.0, filtered).astype(np.float32)

    @staticmethod
    def _fallback_radon_cv2(img_2d: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Запасной расчет синограммы через OpenCV при отсутствии scikit-image."""
        h, w = img_2d.shape
        diag = int(np.ceil(np.hypot(h, w)))
        pad_y = max(0, (diag - h) // 2)
        pad_x = max(0, (diag - w) // 2)
        padded = np.pad(img_2d, ((pad_y, diag - h - pad_y), (pad_x, diag - w - pad_x)), mode="constant")
        n = padded.shape[0]
        cx = cy = n / 2.0
        sino = np.zeros((n, len(theta)), dtype=np.float32)
        for col, ang in enumerate(theta):
            M = cv2.getRotationMatrix2D((cx, cy), -float(ang), 1.0)
            rot = cv2.warpAffine(padded, M, (n, n))
            sino[:, col] = np.sum(rot, axis=0)
        return sino

    @staticmethod
    def _fallback_fbp_cv2(filtered_sino: np.ndarray, theta: np.ndarray, out_size: int) -> np.ndarray:
        n_rho = filtered_sino.shape[0]
        fbp = np.zeros((n_rho, n_rho), dtype=np.float32)
        cx = cy = n_rho / 2.0
        for col, ang in enumerate(theta):
            strip = np.tile(filtered_sino[:, col], (n_rho, 1))
            M = cv2.getRotationMatrix2D((cx, cy), float(ang), 1.0)
            fbp += cv2.warpAffine(strip, M, (n_rho, n_rho))
        if n_rho != out_size:
            start = max(0, (n_rho - out_size) // 2)
            fbp = fbp[start : start + out_size, start : start + out_size]
        return fbp

    def _compute_filtered_sinogram(
        self,
        target_image: np.ndarray,
        edges_np: np.ndarray,
        compute_fbp: bool = True,
        density_override: Optional[np.ndarray] = None,
    ) -> None:
        t0 = time.perf_counter()
        if density_override is not None:
            darkness = np.clip(density_override.astype(np.float32, copy=False), 0.0, 1.0)
        else:
            darkness = (255.0 - target_image.astype(np.float32)) / 255.0
        edge_norm = edges_np.astype(np.float32) / 255.0
        hybrid_density = 0.65 * darkness + 0.35 * edge_norm

        small_size = 250
        small_density = cv2.resize(hybrid_density, (small_size, small_size), interpolation=cv2.INTER_AREA)
        theta = np.linspace(0.0, 180.0, 180, endpoint=False)
        use_circle = self.canvas_manager.shape == "Круг"

        if SKIMAGE_AVAILABLE and radon is not None:
            if use_circle:
                yy, xx = np.ogrid[:small_size, :small_size]
                c = small_size // 2
                r2 = max(1, small_size // 2 - 1) ** 2
                small_density = np.where((xx - c) ** 2 + (yy - c) ** 2 <= r2, small_density, 0.0)
            raw_sino = radon(small_density, theta=theta, circle=use_circle)
        else:
            raw_sino = self._fallback_radon_cv2(small_density, theta)

        filtered_sino = self._apply_ramp_filter(raw_sino)
        f_min = float(filtered_sino.min())
        f_max = float(filtered_sino.max())
        if f_max > f_min:
            self.sinogram = ((filtered_sino - f_min) / (f_max - f_min)).astype(np.float32)
        else:
            self.sinogram = filtered_sino.astype(np.float32)

        # Реконструкция FBP для визуальной вкладки аналитики (можно пропускать для доп. слоев CMYK)
        if compute_fbp:
            try:
                if SKIMAGE_AVAILABLE and iradon is not None:
                    fbp = iradon(raw_sino, theta=theta, filter_name="ramp", circle=use_circle, output_size=small_size)
                else:
                    fbp = self._fallback_fbp_cv2(filtered_sino, theta, small_size)
                fbp = np.maximum(0.0, fbp)
                self.fbp_recon = cv2.normalize(fbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            except Exception:
                self.fbp_recon = np.zeros((small_size, small_size), dtype=np.uint8)

        # Векторизованное проецирование всех допустимых хорд в координаты (rho, theta)
        num_pins = self.canvas_manager.num_pins
        coords = self.canvas_manager.pin_coords
        center = self.canvas_manager.size / 2.0
        scale_rho = small_size / float(self.canvas_manager.size)
        min_dist = compute_min_pin_distance(num_pins)
        n_rho, n_theta = self.sinogram.shape

        if self.canvas_manager.line_keys:
            pairs = np.asarray(self.canvas_manager.line_keys, dtype=np.int32)
            i_arr = pairs[:, 0]
            j_arr = pairs[:, 1]
        else:
            ii, jj = np.triu_indices(num_pins, k=1)
            dists = np.minimum(jj - ii, num_pins - (jj - ii))
            valid_d = dists >= min_dist
            i_arr = ii[valid_d].astype(np.int32)
            j_arr = jj[valid_d].astype(np.int32)

        if len(i_arr) > 0:
            p1 = coords[i_arr]
            p2 = coords[j_arr]
            dx = p2[:, 0] - p1[:, 0]
            dy_cart = -(p2[:, 1] - p1[:, 1])  # Ось Y вверх в декартовых координатах skimage.radon
            line_ang = np.degrees(np.arctan2(dy_cart, dx))
            proj_theta = (line_ang + 90.0) % 180.0
            rad = np.radians(proj_theta)

            mx = (p1[:, 0] + p2[:, 0]) * 0.5 - center
            my_cart = center - (p1[:, 1] + p2[:, 1]) * 0.5
            rho = (mx * np.cos(rad) + my_cart * np.sin(rad)) * scale_rho

            rho_idx = np.round(rho + n_rho * 0.5).astype(np.int32)
            theta_idx = np.round(proj_theta).astype(np.int32) % n_theta
            in_bounds = (rho_idx >= 0) & (rho_idx < n_rho)

            weights = np.zeros(len(i_arr), dtype=np.float32)
            weights[in_bounds] = self.sinogram[rho_idx[in_bounds], theta_idx[in_bounds]]
            self.chord_weights = weights

            self.pin_adj_weights[i_arr, j_arr] = weights
            self.pin_adj_weights[j_arr, i_arr] = weights
            self.line_weights = dict(zip(zip(i_arr.tolist(), j_arr.tolist()), weights.tolist()))

        # Предвычисляем топ-36 лучших соседей для каждого гвоздя
        self.top_neighbors_cache = []
        for i in range(num_pins):
            order = np.argsort(self.pin_adj_weights[i])[::-1]
            dists = np.minimum(np.abs(order - i), num_pins - np.abs(order - i))
            valid = order[dists >= min_dist]
            self.top_neighbors_cache.append(valid[:36].astype(np.int32))

        dt = time.perf_counter() - t0
        logging.info(f"Преобразование Радона{' и FBP' if compute_fbp else ''} рассчитано за {dt:.3f} сек.")
