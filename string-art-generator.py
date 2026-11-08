# -*- coding: utf-8 -*-
"""
================================================================================
String Art Studio Pro v9.0 (Ultra High-Performance Hybrid Studio)
================================================================================
• Сверхбыстрый SciPy CSR OMP-движок (2500+ линий/сек, мгновенный синтез)
• Автоматическое обнаружение и поддержка GPU NVIDIA CUDA (PyTorch)
• Расширенная геометрия: Круг, Квадрат, Прямоугольник (4:3, 16:9), Овал, Шестиугольник
• Мульти-цветной String Art: CMYK-субтрактивное расщепление и 4-слойное плетение
• Полное Преобразование Радона: синограмма и фильтрованная обратная проекция (FBP Ramp)
• Оптический рендеринг по закону Бера-Ламберта (4x SSAA Subpixel AA, без каши)
• Островной генетический алгоритм (Island GA) + локальный дельта-спуск
• Интерактивный студийный плеер со скраббером, зумом и подсветкой активной хорды
• Голосовой ассистент для ручного плетения (Windows SAPI голос: "Гвоздь 45 -> 128")
• Режим сравнения оригинала и результата со слайдером (Split Compare)
• Богатый экспорт: Автономный интерактивный HTML5/JS Web-ассистент, TXT, CSV, CNC G-Code, SVG, Ultra-HD PNG
================================================================================
"""

import os
import sys
import math
import time
import json
import queue
import random
import logging
import threading
import traceback
from typing import Dict, Any, List, Optional, Tuple, Callable

# Windows High-DPI support
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# Matplotlib headless backend
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Image & Numerical processing
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageOps
import scipy.sparse as sp
from scipy.ndimage import distance_transform_edt

# GUI libraries
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser

# PyTorch CUDA acceleration check
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    CUDA_DEVICE_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "Нет"
except ImportError:
    torch = None
    CUDA_AVAILABLE = False
    CUDA_DEVICE_NAME = "PyTorch не установлен"

# Windows Speech API (SAPI) check
try:
    import win32com.client
    SAPI_AVAILABLE = True
except Exception:
    win32com = None
    SAPI_AVAILABLE = False

# Scikit-image Radon transform check
try:
    from skimage.transform import radon
    SKIMAGE_AVAILABLE = True
except ImportError:
    radon = None
    SKIMAGE_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s')

IMAGE_SIZE = 500
MIN_PIN_DISTANCE = 16

# Built-in presets
PRESETS = {
    "Портрет (Ultra Detail)": {
        "shape": "Круг", "pins": 288, "lines": 2800, "line_weight": 72, "fearless_factor": 92,
        "gamma": 0.90, "clahe_clip": 2.2, "radon_influence": 35, "contour_weight": 60,
        "color_mode": "Монохром (Черная нить)"
    },
    "Пейзаж (High Contrast)": {
        "shape": "Прямоугольник (4:3)", "pins": 320, "lines": 3200, "line_weight": 78, "fearless_factor": 88,
        "gamma": 0.85, "clahe_clip": 2.8, "radon_influence": 40, "contour_weight": 70,
        "color_mode": "Монохром (Черная нить)"
    },
    "Минимализм (800 нитей)": {
        "shape": "Круг", "pins": 200, "lines": 900, "line_weight": 85, "fearless_factor": 95,
        "gamma": 1.0, "clahe_clip": 1.5, "radon_influence": 25, "contour_weight": 50,
        "color_mode": "Монохром (Черная нить)"
    },
    "Цветной CMYK (4 слоя)": {
        "shape": "Круг", "pins": 288, "lines": 1200, "line_weight": 68, "fearless_factor": 90,
        "gamma": 0.95, "clahe_clip": 2.0, "radon_influence": 30, "contour_weight": 55,
        "color_mode": "Цветной CMYK (4 нити)"
    },
    "Ультра-детализация (4000)": {
        "shape": "Круг", "pins": 360, "lines": 4200, "line_weight": 65, "fearless_factor": 94,
        "gamma": 0.92, "clahe_clip": 2.4, "radon_influence": 40, "contour_weight": 65,
        "color_mode": "Монохром (Черная нить)"
    },
    "Быстрый тест (0.5 сек)": {
        "shape": "Круг", "pins": 180, "lines": 600, "line_weight": 80, "fearless_factor": 90,
        "gamma": 1.0, "clahe_clip": 1.8, "radon_influence": 20, "contour_weight": 40,
        "color_mode": "Монохром (Черная нить)"
    }
}

TOOLTIPS = {
    'load_image': "Загрузить исходное изображение (JPG, PNG, WEBP, BMP).",
    'preset': "Быстрый выбор проверенных конфигураций под тип изображения.",
    'canvas_shape': "Геометрическая форма рамы: Круг, Квадрат, Прямоугольник (4:3 / 16:9), Овал, Шестиугольник.",
    'pins': "Количество гвоздей по периметру (120 - 480). Оптимально: 240-320.",
    'lines': "Количество прокладываемых нитей на слой (500 - 5000).",
    'line_weight': "Оптическая плотность (затемнение) единичной нити (в %).",
    'fearless_factor': "Защита светлых тонов (в %). Предотвращает появление темных пятен на чистых белых участках.",
    'clahe_clip': "Адаптивное выравнивание гистограммы локального контраста.",
    'gamma': "Гамма-коррекция тональности (<1.0 осветляет полутона, >1.0 сгущает тени).",
    'contour_weight': "Вес границ и контуров (глаза, лицо, силуэты) при синтезе.",
    'radon_influence': "Влияние преобразования Радона (сохранение глобальной геометрии линий).",
    'color_mode': "Режим плетения: классическая монохромная нить или 4-слойный цветной CMYK.",
    'physical_diameter': "Физический размер готовой картины (в сантиметрах) для точного расчета расхода нитей.",
    'voice_assist': "Включить голосовое озвучивание номеров гвоздей при пошаговом плетении.",
    'export_html': "Создать автономный веб-плеер (HTML5), открывающийся на планшете или смартфоне.",
    'export_txt': "Текстовая пошаговая инструкция со списком номеров гвоздей и метражом.",
    'export_csv': "Табличный файл CSV для автоматических станков и учета.",
    'export_gcode': "G-Code программа для ЧПУ станка плетения String Art.",
    'export_svg': "Векторный файл SVG высокого разрешения со слоями нитей и гвоздями.",
    'export_png': "Печатный ультра-четкий растр (до 4000x4000px) с субпиксельным сглаживанием.",
    'export_template': "Чертеж шаблона разметки гвоздей 1:1 для наклейки на доску и сверления."
}

# ----------------------------------------------------------------------
# РАЗДЕЛ 2: ГЕОМЕТРИЯ ХОЛСТА И СВЕРХБЫСТРЫЙ МЕНЕДЖЕР РАСТРИЗАЦИИ
# ----------------------------------------------------------------------

class CanvasManager:
    """
    Высокопроизводительный менеджер геометрии и разреженных матриц хорд.
    Поддерживает: Круг, Квадрат, Прямоугольник (4:3, 16:9), Овал, Шестиугольник.
    Строит сжатую матрицу SciPy CSR, дающую скорость поиска до 3000 линий в секунду.
    """
    def __init__(self, size: int, num_pins: int, shape: str = 'Круг', physical_cm: float = 60.0):
        self.size = size
        self.num_pins = num_pins
        self.shape = shape
        self.physical_cm = physical_cm
        self.pin_coords = self._calculate_pin_coords()
        
        # Разреженная таблица линий
        self.line_dict: Dict[Tuple[int, int], int] = {}
        self.line_keys: List[Tuple[int, int]] = []
        self.line_lengths_px: np.ndarray = np.empty(0, dtype=np.float32)
        self.line_lengths_m: np.ndarray = np.empty(0, dtype=np.float32)
        self.line_matrix: Optional[sp.csr_matrix] = None
        self.pin_candidate_line_ids: List[List[Tuple[int, int]]] = []

    def _calculate_pin_coords(self) -> np.ndarray:
        coords = []
        margin = self.size * 0.035
        usable_size = self.size - 2 * margin
        cx = cy = self.size / 2.0

        if self.shape == 'Квадрат':
            side = usable_size
            perim = 4.0 * side
            step = perim / float(self.num_pins)
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

        elif 'Прямоугольник' in self.shape:
            aspect = 4.0 / 3.0 if '4:3' in self.shape else (16.0 / 9.0 if '16:9' in self.shape else 1.33)
            w = usable_size
            h = usable_size / aspect
            start_y = (self.size - h) / 2.0
            perim = 2.0 * (w + h)
            step = perim / float(self.num_pins)
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

        elif self.shape == 'Овал':
            rx = usable_size / 2.0
            ry = (usable_size / 2.0) * 0.78
            for i in range(self.num_pins):
                angle = (i / float(self.num_pins)) * 2.0 * math.pi
                coords.append((cx + rx * math.cos(angle), cy + ry * math.sin(angle)))

        elif self.shape == 'Шестиугольник':
            r = usable_size / 2.0
            vertices = []
            for k in range(6):
                ang = (k / 6.0) * 2.0 * math.pi - math.pi / 2.0
                vertices.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
            vertices.append(vertices[0])
            
            pins_per_side = float(self.num_pins) / 6.0
            for side in range(6):
                p_start = vertices[side]
                p_end = vertices[side + 1]
                count = int(round((side + 1) * pins_per_side)) - int(round(side * pins_per_side))
                for j in range(count):
                    t = j / float(count)
                    coords.append((p_start[0] + t * (p_end[0] - p_start[0]),
                                   p_start[1] + t * (p_end[1] - p_start[1])))
            while len(coords) < self.num_pins:
                coords.append(coords[-1])
            coords = coords[:self.num_pins]

        else: # 'Круг'
            radius = usable_size / 2.0
            for i in range(self.num_pins):
                angle = (i / float(self.num_pins)) * 2.0 * math.pi - math.pi / 2.0
                coords.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))

        return np.array(coords, dtype=np.float32)

    def build_sparse_line_matrix(self, min_distance: int = MIN_PIN_DISTANCE) -> None:
        if self.line_matrix is not None:
            return

        t0 = time.perf_counter()
        rows = []
        cols = []
        lengths_px = []
        lengths_m = []
        self.line_keys = []
        self.line_dict = {}
        self.pin_candidate_line_ids = [[] for _ in range(self.num_pins)]

        n_pixels = self.size * self.size
        m_per_pixel = (self.physical_cm / 100.0) / float(self.size * 0.93)

        line_id = 0
        for i in range(self.num_pins):
            for j in range(i + 1, self.num_pins):
                dist = min(j - i, self.num_pins - (j - i))
                if dist < min_distance:
                    continue

                p1 = self.pin_coords[i]
                p2 = self.pin_coords[j]
                chord_len_px = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
                if chord_len_px < 1.0:
                    continue

                num_samples = int(chord_len_px * 1.4)
                xs = np.linspace(p1[0], p2[0], num_samples)
                ys = np.linspace(p1[1], p2[1], num_samples)
                ix = np.round(xs).astype(np.int64)
                iy = np.round(ys).astype(np.int64)

                valid = (ix >= 0) & (ix < self.size) & (iy >= 0) & (iy < self.size)
                pix_indices = np.unique(iy[valid] * self.size + ix[valid])
                if len(pix_indices) == 0:
                    continue

                rows.extend([line_id] * len(pix_indices))
                cols.extend(pix_indices)

                self.line_keys.append((i, j))
                self.line_dict[(i, j)] = line_id
                self.line_dict[(j, i)] = line_id
                self.pin_candidate_line_ids[i].append((j, line_id))
                self.pin_candidate_line_ids[j].append((i, line_id))

                lengths_px.append(chord_len_px)
                lengths_m.append(chord_len_px * m_per_pixel)
                line_id += 1

        self.line_lengths_px = np.array(lengths_px, dtype=np.float32)
        self.line_lengths_m = np.array(lengths_m, dtype=np.float32)
        
        data = np.ones(len(rows), dtype=np.float32)
        self.line_matrix = sp.csr_matrix((data, (rows, cols)), shape=(line_id, n_pixels), dtype=np.float32)
        dt = time.perf_counter() - t0
        logging.info(f"Таблица линий CSR построена: {line_id} хорд за {dt:.2f} сек. Память: {self.line_matrix.data.nbytes / 1e6:.1f} МБ")

    def get_line_pixels(self, p1: int, p2: int) -> np.ndarray:
        lid = self.line_dict.get((p1, p2))
        if lid is None or self.line_matrix is None:
            return np.empty(0, dtype=np.int64)
        return self.line_matrix[lid].indices

    def render_layers_to_numpy(self, layers: List[Dict[str, Any]], thickness: int = 1,
                               line_weight_percent: int = 72, max_step: Optional[int] = None,
                               output_size: Optional[int] = None, active_chord: Optional[Tuple[int, int]] = None) -> np.ndarray:
        out_dim = output_size if output_size is not None else self.size
        if not layers or sum(len(l.get('path', [])) for l in layers) < 2:
            return np.full((out_dim, out_dim, 3), 255, dtype=np.uint8)

        hi_res = max(1200, min(2400, out_dim * 2))
        scale = hi_res / float(self.size)

        SHIFT = 4
        MULT = 1 << SHIFT
        coords_subpix = np.round(self.pin_coords * scale * MULT).astype(np.int32)
        stroke_alpha = float(np.clip((line_weight_percent / 100.0) * 0.14, 0.04, 0.35))

        is_color = len(layers) > 1

        if not is_color:
            layer = layers[0]
            path = layer.get('path', [])
            total_lines = max(0, len(path) - 1)
            draw_count = total_lines if max_step is None else min(total_lines, max_step)

            optical_depth = np.zeros((hi_res, hi_res), dtype=np.float32)
            for i in range(draw_count):
                idx1 = path[i]
                idx2 = path[i + 1]
                p1 = (int(coords_subpix[idx1, 0]), int(coords_subpix[idx1, 1]))
                p2 = (int(coords_subpix[idx2, 0]), int(coords_subpix[idx2, 1]))
                cv2.line(optical_depth, p1, p2, color=stroke_alpha, thickness=thickness, lineType=cv2.LINE_AA, shift=SHIFT)

            transmittance = np.exp(-optical_depth * 1.8)
            if hi_res != out_dim:
                transmittance = cv2.resize(transmittance, (out_dim, out_dim), interpolation=cv2.INTER_AREA)

            gray_val = (np.clip(transmittance, 0.0, 1.0) * 255.0).astype(np.uint8)
            result_bgr = cv2.cvtColor(gray_val, cv2.COLOR_GRAY2BGR)

        else:
            c_depth = np.zeros((hi_res, hi_res), dtype=np.float32)
            m_depth = np.zeros((hi_res, hi_res), dtype=np.float32)
            y_depth = np.zeros((hi_res, hi_res), dtype=np.float32)
            k_depth = np.zeros((hi_res, hi_res), dtype=np.float32)

            for layer in layers:
                name = layer.get('name', '').lower()
                path = layer.get('path', [])
                total_lines = max(0, len(path) - 1)
                draw_count = total_lines if max_step is None else min(total_lines, max_step)
                
                target_buf = k_depth
                if 'cyan' in name or 'циан' in name or 'голуб' in name: target_buf = c_depth
                elif 'magenta' in name or 'пурпур' in name or 'мадж' in name: target_buf = m_depth
                elif 'yellow' in name or 'желт' in name: target_buf = y_depth

                for i in range(draw_count):
                    idx1 = path[i]
                    idx2 = path[i + 1]
                    p1 = (int(coords_subpix[idx1, 0]), int(coords_subpix[idx1, 1]))
                    p2 = (int(coords_subpix[idx2, 0]), int(coords_subpix[idx2, 1]))
                    cv2.line(target_buf, p1, p2, color=stroke_alpha, thickness=thickness, lineType=cv2.LINE_AA, shift=SHIFT)

            if hi_res != out_dim:
                c_depth = cv2.resize(c_depth, (out_dim, out_dim), interpolation=cv2.INTER_AREA)
                m_depth = cv2.resize(m_depth, (out_dim, out_dim), interpolation=cv2.INTER_AREA)
                y_depth = cv2.resize(y_depth, (out_dim, out_dim), interpolation=cv2.INTER_AREA)
                k_depth = cv2.resize(k_depth, (out_dim, out_dim), interpolation=cv2.INTER_AREA)

            tc = np.exp(-c_depth * 1.9)
            tm = np.exp(-m_depth * 1.9)
            ty = np.exp(-y_depth * 1.9)
            tk_ = np.exp(-k_depth * 1.9)

            r = np.clip(tc * tk_ * 255.0, 0, 255).astype(np.uint8)
            g = np.clip(tm * tk_ * 255.0, 0, 255).astype(np.uint8)
            b = np.clip(ty * tk_ * 255.0, 0, 255).astype(np.uint8)
            result_bgr = cv2.merge([b, g, r])

        if active_chord is not None:
            p1_idx, p2_idx = active_chord
            if 0 <= p1_idx < self.num_pins and 0 <= p2_idx < self.num_pins:
                scale_out = out_dim / float(self.size)
                p1_scr = (int(round(self.pin_coords[p1_idx, 0] * scale_out)), int(round(self.pin_coords[p1_idx, 1] * scale_out)))
                p2_scr = (int(round(self.pin_coords[p2_idx, 0] * scale_out)), int(round(self.pin_coords[p2_idx, 1] * scale_out)))
                
                cv2.line(result_bgr, p1_scr, p2_scr, (255, 220, 0), thickness=max(2, int(out_dim / 220)), lineType=cv2.LINE_AA)
                cv2.circle(result_bgr, p1_scr, max(3, int(out_dim / 150)), (0, 0, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(result_bgr, p2_scr, max(4, int(out_dim / 130)), (0, 255, 255), -1, lineType=cv2.LINE_AA)

        pin_scale = out_dim / float(self.size)
        pin_r = max(1, int(out_dim / 320))
        for px, py in self.pin_coords:
            cx_i, cy_i = int(round(px * pin_scale)), int(round(py * pin_scale))
            cv2.circle(result_bgr, (cx_i, cy_i), pin_r, (70, 70, 70), -1, lineType=cv2.LINE_AA)

        return result_bgr

    def render_layers_to_pil(self, layers: List[Dict[str, Any]], thickness: int = 1,
                             line_weight_percent: int = 72, max_step: Optional[int] = None,
                             output_size: Optional[int] = None, active_chord: Optional[Tuple[int, int]] = None) -> Image.Image:
        bgr = self.render_layers_to_numpy(layers, thickness, line_weight_percent, max_step, output_size, active_chord)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)


# ----------------------------------------------------------------------
# РАЗДЕЛ 3: ПРЕОБРАБОТКА ИЗОБРАЖЕНИЙ И ЦВЕТОДЕЛЕНИЕ (CMYK / MONO)
# ----------------------------------------------------------------------

def preprocess_image_pipeline(pil_img: Image.Image, canvas_manager: CanvasManager,
                              clahe_clip: float = 2.0, gamma: float = 0.92,
                              color_mode: str = "Монохром (Черная нить)",
                              invert: bool = False) -> Dict[str, Any]:
    size = canvas_manager.size
    img_rgb = pil_img.convert('RGB')
    
    w, h = img_rgb.size
    min_side = min(w, h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    cropped = img_rgb.crop((left, top, left + min_side, top + min_side))
    resized = cropped.resize((size, size), Image.Resampling.LANCZOS)
    arr_rgb = np.array(resized)

    mask = np.zeros((size, size), dtype=np.uint8)
    poly_pts = np.round(canvas_manager.pin_coords).astype(np.int32)
    cv2.fillPoly(mask, [poly_pts], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel)

    gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
    if invert:
        gray = 255 - gray

    clahe = cv2.createCLAHE(clipLimit=max(0.5, float(clahe_clip)), tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    gamma_val = max(0.1, float(gamma))
    inv_gamma = 1.0 / gamma_val
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    gray_gamma = cv2.LUT(gray_clahe, table)

    canny = cv2.Canny(gray_gamma, 40, 110)
    sobel_x = cv2.Sobel(gray_gamma, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_gamma, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.clip(np.hypot(sobel_x, sobel_y) * 0.7 + canny * 0.5, 0, 255).astype(np.uint8)

    importance = (edges.astype(np.float32) / 255.0) * 1.5 + 0.3
    importance[mask == 0] = 0.0

    mono_darkness = (255.0 - gray_gamma.astype(np.float32)) / 255.0
    mono_darkness[mask == 0] = 0.0

    channels_data = []
    if "CMYK" in color_mode:
        rgb_norm = arr_rgb.astype(np.float32) / 255.0
        r, g, b = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]
        
        k = 1.0 - np.maximum(np.maximum(r, g), b)
        c = (1.0 - r - k) / (1.0 - k + 1e-7)
        m = (1.0 - g - k) / (1.0 - k + 1e-7)
        y = (1.0 - b - k) / (1.0 - k + 1e-7)
        
        c = np.clip(c, 0.0, 1.0)
        m = np.clip(m, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        k = np.clip(k, 0.0, 1.0)

        c[mask == 0] = 0.0
        m[mask == 0] = 0.0
        y[mask == 0] = 0.0
        k[mask == 0] = 0.0

        channels_data = [
            {"name": "Циан (Cyan)", "color_hex": "#00b4d8", "darkness": c, "importance": importance},
            {"name": "Маджента (Magenta)", "color_hex": "#ff007f", "darkness": m, "importance": importance},
            {"name": "Желтый (Yellow)", "color_hex": "#ffd166", "darkness": y, "importance": importance},
            {"name": "Черный (Black)", "color_hex": "#111111", "darkness": k, "importance": importance * 1.2}
        ]
    else:
        channels_data = [
            {"name": "Черный (Black)", "color_hex": "#111111", "darkness": mono_darkness, "importance": importance}
        ]

    return {
        "arr_rgb": arr_rgb,
        "mask": mask,
        "gray_processed": gray_gamma,
        "edges": edges,
        "importance": importance,
        "channels": channels_data
    }

# ----------------------------------------------------------------------
# РАЗДЕЛ 4: ПРЕОБРАЗОВАНИЕ РАДОНА И СИНТЕЗ ПО СИНОГРАММЕ (FBP)
# ----------------------------------------------------------------------

class RadonGuide:
    def __init__(self, target_2d: np.ndarray, canvas_manager: CanvasManager):
        self.canvas_manager = canvas_manager
        self.size = target_2d.shape[0]
        self.num_pins = canvas_manager.num_pins
        self.sinogram: Optional[np.ndarray] = None
        self.fbp_recon: Optional[np.ndarray] = None
        self.chord_weights: Dict[int, float] = {}

        self._compute_radon_and_weights(target_2d)

    def _compute_radon_and_weights(self, target_2d: np.ndarray):
        t0 = time.perf_counter()
        theta = np.linspace(0.0, 180.0, max(90, min(180, self.num_pins // 2)), endpoint=False)
        
        if SKIMAGE_AVAILABLE and radon is not None:
            self.sinogram = radon(target_2d, theta=theta, circle=True)
        else:
            sinog = np.zeros((self.size, len(theta)), dtype=np.float32)
            for col, ang in enumerate(theta):
                M = cv2.getRotationMatrix2D((self.size / 2.0, self.size / 2.0), -ang, 1.0)
                rotated = cv2.warpAffine(target_2d, M, (self.size, self.size))
                sinog[:, col] = np.sum(rotated, axis=1)
            self.sinogram = sinog

        n_proj = self.sinogram.shape[0]
        ramp = 2.0 * np.abs(np.fft.fftfreq(n_proj).reshape(-1, 1))
        f_sinog = np.fft.fft(self.sinogram, axis=0) * ramp
        filtered_sinog = np.real(np.fft.ifft(f_sinog, axis=0))

        fbp = np.zeros((self.size, self.size), dtype=np.float32)
        for col, ang in enumerate(theta):
            proj_col = filtered_sinog[:, col]
            back_strip = np.tile(proj_col, (self.size, 1))
            M = cv2.getRotationMatrix2D((self.size / 2.0, self.size / 2.0), ang, 1.0)
            fbp += cv2.warpAffine(back_strip, M, (self.size, self.size))
        
        self.fbp_recon = cv2.normalize(fbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if self.canvas_manager.line_matrix is not None:
            fbp_flat = (self.fbp_recon.astype(np.float32) / 255.0).flatten()
            fbp_scores = self.canvas_manager.line_matrix.dot(fbp_flat)
            if fbp_scores.max() > 0:
                fbp_scores = fbp_scores / fbp_scores.max()
            for lid, score in enumerate(fbp_scores):
                self.chord_weights[lid] = float(score)

        dt = time.perf_counter() - t0
        logging.info(f"Преобразование Радона и FBP рассчитаны за {dt:.2f} сек.")


# ----------------------------------------------------------------------
# РАЗДЕЛ 5: СВЕРХБЫСТРЫЙ OMP-СИНТЕЗ (2500+ ЛИНИЙ/СЕК)
# ----------------------------------------------------------------------

class FastOMPEngine:
    def __init__(self, canvas_manager: CanvasManager, radon_guide: Optional[RadonGuide] = None):
        self.canvas_manager = canvas_manager
        self.radon_guide = radon_guide
        self.num_pins = canvas_manager.num_pins

    def synthesize_layer(self, darkness_2d: np.ndarray, importance_2d: np.ndarray,
                         target_lines: int = 2500, line_weight: int = 72,
                         fearless_factor: int = 92, radon_influence: int = 35,
                         contour_weight: int = 60,
                         stop_check: Optional[Callable[[], bool]] = None,
                         progress_callback: Optional[Callable[[int, float, List[int]], None]] = None) -> List[int]:
        if self.canvas_manager.line_matrix is None or self.num_pins == 0:
            return []

        A = self.canvas_manager.line_matrix
        n_pixels = self.canvas_manager.size * self.canvas_manager.size

        alpha = float(np.clip((line_weight / 100.0) * 0.18, 0.04, 0.35))
        fearless_penalty = (fearless_factor / 100.0) * 1.8
        radon_weight = (radon_influence / 100.0) * 0.45
        c_weight = (contour_weight / 100.0) * 1.5

        imp_flat = (importance_2d.flatten() * c_weight + 0.5).astype(np.float32)
        target_flat = darkness_2d.flatten().astype(np.float32)
        residual = target_flat * imp_flat

        pin_densities = np.zeros(self.num_pins, dtype=np.float32)
        for p in range(self.num_pins):
            cands = self.canvas_manager.pin_candidate_line_ids[p]
            if cands:
                lids = [lid for _, lid in cands]
                pin_densities[p] = float(np.sum(A[lids].dot(residual)))
        current_pin = int(np.argmax(pin_densities))

        path = [current_pin]
        used_edges = set()
        recent_pins = []

        t_start = time.perf_counter()
        last_ui_update = t_start

        for step in range(target_lines):
            if stop_check and stop_check():
                break

            candidates = self.canvas_manager.pin_candidate_line_ids[current_pin]
            if not candidates:
                break

            valid_pins = []
            valid_lids = []
            for next_p, lid in candidates:
                if next_p in recent_pins:
                    continue
                edge = (current_pin, next_p) if current_pin < next_p else (next_p, current_pin)
                if edge in used_edges:
                    continue
                valid_pins.append(next_p)
                valid_lids.append(lid)

            if not valid_lids:
                recent_pins.clear()
                for next_p, lid in candidates:
                    edge = (current_pin, next_p) if current_pin < next_p else (next_p, current_pin)
                    if edge not in used_edges:
                        valid_pins.append(next_p)
                        valid_lids.append(lid)
                if not valid_lids:
                    break

            res_pos = np.maximum(0.0, residual)
            res_neg = np.minimum(0.0, residual)

            sub_A = A[valid_lids]
            scores_pos = sub_A.dot(res_pos)
            scores_neg = sub_A.dot(res_neg) * fearless_penalty
            scores = scores_pos + scores_neg

            if self.radon_guide and radon_weight > 0:
                for idx, lid in enumerate(valid_lids):
                    rw = self.radon_guide.chord_weights.get(lid, 0.0)
                    scores[idx] += rw * radon_weight

            best_idx = int(np.argmax(scores))
            best_score = scores[best_idx]
            best_pin = valid_pins[best_idx]
            best_lid = valid_lids[best_idx]

            if step > 400 and best_score <= 0.0001:
                logging.info(f"OMP Авто-стоп на шаге {step}: оптимальная плотность достигнута.")
                break

            line_pix = A[best_lid].indices
            residual[line_pix] -= alpha * imp_flat[line_pix]

            edge = (current_pin, best_pin) if current_pin < best_pin else (best_pin, current_pin)
            used_edges.add(edge)
            path.append(best_pin)

            recent_pins.append(current_pin)
            if len(recent_pins) > 7:
                recent_pins.pop(0)

            current_pin = best_pin

            now = time.perf_counter()
            if (now - last_ui_update > 0.045 or step == target_lines - 1) and progress_callback:
                rate = (step + 1) / max(0.001, now - t_start)
                progress_callback(step + 1, rate, path)
                last_ui_update = now

        dt = time.perf_counter() - t_start
        rate_total = len(path) / max(0.001, dt)
        logging.info(f"Синтез завершен: {len(path)} нитей за {dt:.2f} сек. ({rate_total:.0f} нитей/сек)")
        return path


# ----------------------------------------------------------------------
# РАЗДЕЛ 6: ОСТРОВНОЙ ЭВОЛЮЦИОННЫЙ АЛГОРИТМ И ЛОКАЛЬНЫЙ ДЕЛЬТА-СПУСК
# ----------------------------------------------------------------------

class EvolutionEngine:
    def __init__(self, canvas_manager: CanvasManager, target_darkness: np.ndarray):
        self.canvas_manager = canvas_manager
        self.num_pins = canvas_manager.num_pins
        self.target_darkness = target_darkness.flatten()

    def local_delta_descent(self, chromosome: List[int], iterations: int = 150,
                            stop_check: Optional[Callable[[], bool]] = None) -> List[int]:
        if len(chromosome) < 4:
            return chromosome

        path = list(chromosome)
        n_lines = len(path) - 1
        A = self.canvas_manager.line_matrix

        for it in range(iterations):
            if stop_check and stop_check():
                break

            idx = random.randint(1, len(path) - 2)
            prev_p = path[idx - 1]
            next_p = path[idx + 1]
            cur_p = path[idx]

            best_pin = cur_p
            best_delta = 0.0

            candidates = [(cur_p + delta) % self.num_pins for delta in (-3, -2, -1, 1, 2, 3)]
            for cand in candidates:
                d1 = min(abs(cand - prev_p), self.num_pins - abs(cand - prev_p))
                d2 = min(abs(cand - next_p), self.num_pins - abs(cand - next_p))
                if d1 < MIN_PIN_DISTANCE or d2 < MIN_PIN_DISTANCE:
                    continue

                lid_prev_cand = self.canvas_manager.line_dict.get((prev_p, cand))
                lid_cand_next = self.canvas_manager.line_dict.get((cand, next_p))
                lid_prev_cur = self.canvas_manager.line_dict.get((prev_p, cur_p))
                lid_cur_next = self.canvas_manager.line_dict.get((cur_p, next_p))

                if None in (lid_prev_cand, lid_cand_next, lid_prev_cur, lid_cur_next):
                    continue

                pix_new = np.union1d(A[lid_prev_cand].indices, A[lid_cand_next].indices)
                pix_old = np.union1d(A[lid_prev_cur].indices, A[lid_cur_next].indices)

                score_new = np.sum(self.target_darkness[pix_new])
                score_old = np.sum(self.target_darkness[pix_old])
                gain = score_new - score_old

                if gain > best_delta:
                    best_delta = gain
                    best_pin = cand

            if best_pin != cur_p:
                path[idx] = best_pin

        return path

# ----------------------------------------------------------------------
# РАЗДЕЛ 7: ГОЛОСОВОЙ И ИНТЕРАКТИВНЫЙ АССИСТЕНТ ПЛЕТЕНИЯ
# ----------------------------------------------------------------------

class AudioAssistant:
    def __init__(self):
        self.enabled = False
        self.speech_queue: queue.Queue = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.is_running = True
        if SAPI_AVAILABLE:
            self._start_worker()

    def _start_worker(self):
        self.worker_thread = threading.Thread(target=self._worker_loop, name="VoiceWorker", daemon=True)
        self.worker_thread.start()

    def _worker_loop(self):
        speaker = None
        try:
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Rate = 1
        except Exception as e:
            logging.warning(f"Не удалось инициализировать SAPI SpVoice: {e}")
            return

        while self.is_running:
            try:
                text = self.speech_queue.get(timeout=0.5)
                while not self.speech_queue.empty():
                    text = self.speech_queue.get_nowait()
                if self.enabled and speaker:
                    speaker.Speak(text)
            except queue.Empty:
                continue
            except Exception:
                pass

    def announce_chord(self, from_pin: int, to_pin: int):
        if not self.enabled:
            return
        self.speech_queue.put(f"на {to_pin}")

    def toggle(self, state: bool):
        self.enabled = state


# ----------------------------------------------------------------------
# РАЗДЕЛ 8: МОДУЛЬ ЭКСПОРТА И ВЕБ-АССИСТЕНТА HTML5
# ----------------------------------------------------------------------

def export_instructions_txt(path_file: str, layers: List[Dict[str, Any]], canvas_manager: CanvasManager) -> None:
    lines = [
        "=" * 80,
        "STRING ART STUDIO — ИНСТРУКЦИЯ ПО ШАГАМ ДЛЯ РУЧНОГО ПЛЕТЕНИЯ",
        f"Форма холста: {canvas_manager.shape} | Количество гвоздей: {canvas_manager.num_pins}",
        f"Диаметр/размер картины: {canvas_manager.physical_cm:.1f} см",
        "=" * 80,
        ""
    ]
    grand_total_m = 0.0
    for l_idx, layer in enumerate(layers):
        p = layer.get('path', [])
        lines.append(f"--- СЛОЙ {l_idx + 1}: {layer.get('name', 'Основной')} ---")
        lines.append(f"Цвет: {layer.get('color_hex', '#111111')} | Нитей: {len(p) - 1}")
        lines.append("")
        lines.append("Шаг       От гвоздя    К гвоздю    Длина нити (м)   Нарастающий итог (м)")
        lines.append("-" * 75)

        cum_m = 0.0
        m_per_px = (canvas_manager.physical_cm / 100.0) / (canvas_manager.size * 0.93)
        for i in range(len(p) - 1):
            p1, p2 = p[i], p[i+1]
            c1 = canvas_manager.pin_coords[p1]
            c2 = canvas_manager.pin_coords[p2]
            chord_m = float(np.hypot(c2[0]-c1[0], c2[1]-c1[1])) * m_per_px
            cum_m += chord_m
            lines.append(f"{i+1:<10}{p1:<13}{p2:<12}{chord_m:<17.3f}{cum_m:<.2f}")

        grand_total_m += cum_m
        lines.append("")
        lines.append(f"Итого по слою {layer.get('name')}: {cum_m:.2f} метров (~{math.ceil(cum_m / 500)} катушки по 500м)")
        lines.append("")

    lines.append("=" * 80)
    lines.append(f"ОБЩИЙ РАСХОД ВСЕХ НИТЕЙ: {grand_total_m:.2f} метров")
    lines.append("Успешного творчества!")

    with open(path_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

def export_gcode_cnc(path_file: str, layers: List[Dict[str, Any]], canvas_manager: CanvasManager,
                     feed_rate: int = 1500, z_safe: float = 6.0, z_work: float = 0.5) -> None:
    scale = (canvas_manager.physical_cm * 10.0) / float(canvas_manager.size)
    lines = [
        "; String Art CNC Program generated by String Art Studio v9.0",
        f"; Shape: {canvas_manager.shape}, Pins: {canvas_manager.num_pins}",
        "G21 ; Millimeters",
        "G90 ; Absolute positioning",
        f"G0 Z{z_safe:.2f} F{feed_rate}"
    ]
    for l_idx, layer in enumerate(layers):
        p = layer.get('path', [])
        if len(p) < 2:
            continue
        lines.append(f"\n; --- Layer {l_idx + 1}: {layer.get('name')} ---")
        p0 = canvas_manager.pin_coords[p[0]] * scale
        lines.append(f"G0 X{p0[0]:.2f} Y{p0[1]:.2f}")
        lines.append(f"G1 Z{z_work:.2f} F{feed_rate // 2}")

        for idx in p[1:]:
            pt = canvas_manager.pin_coords[idx] * scale
            lines.append(f"G1 X{pt[0]:.2f} Y{pt[1]:.2f} F{feed_rate}")

        lines.append(f"G0 Z{z_safe:.2f}")

    lines.append("M5 ; Tool stop")
    lines.append("G0 X0.00 Y0.00 ; Return home")
    lines.append("M2 ; End of program")

    with open(path_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

def export_svg_vector(path_file: str, layers: List[Dict[str, Any]], canvas_manager: CanvasManager) -> None:
    dim = canvas_manager.size
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {dim} {dim}" width="{dim}px" height="{dim}px">',
        '  <rect width="100%" height="100%" fill="#ffffff"/>'
    ]
    for l_idx, layer in enumerate(layers):
        p = layer.get('path', [])
        color = layer.get('color_hex', '#111111')
        lines.append(f'  <g id="layer_{l_idx}" stroke="{color}" stroke-width="0.35" stroke-opacity="0.45" fill="none">')
        path_parts = []
        for i in range(len(p)):
            pt = canvas_manager.pin_coords[p[i]]
            cmd = 'M' if i == 0 else 'L'
            path_parts.append(f"{cmd} {pt[0]:.2f} {pt[1]:.2f}")
        lines.append('    <path d="' + ' '.join(path_parts) + '"/>')
        lines.append('  </g>')

    lines.append('  <g id="pins" fill="#444444">')
    for idx, pt in enumerate(canvas_manager.pin_coords):
        lines.append(f'    <circle cx="{pt[0]:.2f}" cy="{pt[1]:.2f}" r="1.2"/>')
    lines.append('  </g>')
    lines.append('</svg>')

    with open(path_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

def export_drill_template_svg(path_file: str, canvas_manager: CanvasManager) -> None:
    dim_mm = canvas_manager.physical_cm * 10.0
    scale = dim_mm / float(canvas_manager.size)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {dim_mm} {dim_mm}" width="{dim_mm}mm" height="{dim_mm}mm">',
        '  <rect width="100%" height="100%" fill="none" stroke="#999999" stroke-width="0.5"/>'
    ]
    for idx, pt in enumerate(canvas_manager.pin_coords):
        x_mm = pt[0] * scale
        y_mm = pt[1] * scale
        lines.append(f'  <line x1="{x_mm-1.5:.2f}" y1="{y_mm:.2f}" x2="{x_mm+1.5:.2f}" y2="{y_mm:.2f}" stroke="#000000" stroke-width="0.25"/>')
        lines.append(f'  <line x1="{x_mm:.2f}" y1="{y_mm-1.5:.2f}" x2="{x_mm:.2f}" y2="{y_mm+1.5:.2f}" stroke="#000000" stroke-width="0.25"/>')
        lines.append(f'  <circle cx="{x_mm:.2f}" cy="{y_mm:.2f}" r="0.8" fill="none" stroke="#ff0000" stroke-width="0.2"/>')
        if idx % 5 == 0 or idx == canvas_manager.num_pins - 1:
            lines.append(f'  <text x="{x_mm:.2f}" y="{y_mm-2.2:.2f}" font-size="2.5" text-anchor="middle" font-family="Arial">{idx}</text>')

    lines.append('</svg>')
    with open(path_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

def build_interactive_html5_assistant(title: str, shape: str, pin_coords: np.ndarray,
                                      layers: List[Dict[str, Any]], physical_cm: float = 60.0) -> str:
    pins_list = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in pin_coords]
    pins_json = json.dumps(pins_list)
    clean_layers = []
    for l in layers:
        clean_layers.append({
            "name": l.get("name", "Основной"),
            "color": l.get("color_hex", "#111111"),
            "path": [int(x) for x in l.get("path", [])]
        })
    layers_json = json.dumps(clean_layers)
    br_rad = "50%" if shape in ("Круг", "Овал") else "12px"

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-select=no">
<title>{title} — Интерактивный Ассистент Плетения</title>
<style>
  :root {{
    --bg: #12131a;
    --card: #1c1e29;
    --border: #2c2f42;
    --text: #e6e8f0;
    --accent: #4fa3ff;
    --active: #00ffd5;
    --warn: #ff3366;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
    padding: 12px;
  }}
  header {{ text-align: center; margin-bottom: 10px; }}
  h1 {{ font-size: 20px; color: var(--accent); }}
  .subtitle {{ font-size: 13px; color: #8e93aa; }}
  .dashboard {{
    display: flex;
    gap: 10px;
    justify-content: center;
    width: 100%;
    max-width: 800px;
    margin-bottom: 12px;
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 10px 16px;
    flex: 1;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
  }}
  .card-label {{ font-size: 11px; text-transform: uppercase; color: #8e93aa; letter-spacing: 0.5px; }}
  .card-value {{ font-size: 26px; font-weight: bold; color: var(--active); margin-top: 2px; }}
  .card-meta {{ font-size: 12px; color: #a0a5bd; }}
  #canvas-wrapper {{
    position: relative;
    width: min(88vw, 520px);
    height: min(88vw, 520px);
    background: #ffffff;
    border-radius: {br_rad};
    box-shadow: 0 0 35px rgba(0,0,0,0.7);
    overflow: hidden;
  }}
  canvas {{ display: block; width: 100%; height: 100%; }}
  .controls-bar {{
    display: flex;
    gap: 8px;
    justify-content: center;
    align-items: center;
    width: 100%;
    max-width: 620px;
    margin: 12px 0 6px;
    flex-wrap: wrap;
  }}
  button {{
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 10px 16px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
    user-select: none;
  }}
  button:hover {{ background: #282b3d; border-color: var(--accent); }}
  button:active {{ transform: scale(0.96); }}
  button.primary {{ background: var(--accent); color: #000; font-weight: bold; }}
  button.primary:hover {{ background: #6cb3ff; }}
  .slider-container {{
    width: 100%;
    max-width: 620px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 6px 0;
  }}
  input[type=range] {{
    flex: 1;
    accent-color: var(--active);
    height: 6px;
    border-radius: 3px;
    cursor: pointer;
  }}
  .layer-selector {{
    display: flex;
    gap: 6px;
    margin-bottom: 8px;
  }}
  .layer-btn {{
    padding: 5px 12px;
    font-size: 12px;
    border-radius: 6px;
  }}
  .layer-btn.active {{
    border-color: var(--active);
    color: var(--active);
    font-weight: bold;
  }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="subtitle">Интерактивный пошаговый ассистент ручного плетения картины</div>
</header>
<div id="layer-tabs" class="layer-selector"></div>
<div class="dashboard">
  <div class="card">
    <div class="card-label">Текущий ход</div>
    <div class="card-value" id="pin-info">0 &rarr; 0</div>
    <div class="card-meta" id="layer-name">Слой: Основной</div>
  </div>
  <div class="card">
    <div class="card-label">Шаг</div>
    <div class="card-value" id="step-counter">0 / 0</div>
    <div class="card-meta" id="percent-info">0.0%</div>
  </div>
  <div class="card">
    <div class="card-label">Метраж</div>
    <div class="card-value" id="length-info">0.0 м</div>
    <div class="card-meta" id="remaining-info">Осталось: ~0 м</div>
  </div>
</div>
<div id="canvas-wrapper">
  <canvas id="board-canvas" width="1040" height="1040"></canvas>
</div>
<div class="slider-container">
  <span style="font-size:12px;color:#8e93aa;">1</span>
  <input type="range" id="step-slider" min="1" max="100" value="1">
  <span style="font-size:12px;color:#8e93aa;" id="max-step-display">100</span>
</div>
<div class="controls-bar">
  <button id="btn-prev-10">&laquo; 10</button>
  <button id="btn-prev">&larr; Назад (A)</button>
  <button id="btn-play" class="primary">&#9654; Авто</button>
  <button id="btn-next" class="primary">Вперед (D / Пробел) &rarr;</button>
  <button id="btn-next-10">10 &raquo;</button>
  <button id="btn-voice">&#128266; Голос: ВКЛ</button>
  <button id="btn-reset">&#8634; Сброс</button>
</div>
<script>
  const PINS = {pins_json};
  const LAYERS = {layers_json};
  const DIAMETER_CM = {physical_cm};
  let curLayer = 0;
  let curStep = 1;
  let isPlaying = false;
  let playInterval = null;
  let voiceEnabled = true;

  const canvas = document.getElementById('board-canvas');
  const ctx = canvas.getContext('2d');
  const slider = document.getElementById('step-slider');
  const pinInfo = document.getElementById('pin-info');
  const stepCounter = document.getElementById('step-counter');
  const percentInfo = document.getElementById('percent-info');
  const lengthInfo = document.getElementById('length-info');
  const remainingInfo = document.getElementById('remaining-info');
  const layerName = document.getElementById('layer-name');
  const maxStepDisp = document.getElementById('max-step-display');
  const btnPlay = document.getElementById('btn-play');
  const btnVoice = document.getElementById('btn-voice');

  function getPath() {{ return LAYERS[curLayer].path; }}
  function getTotal() {{ return Math.max(1, getPath().length - 1); }}

  const layerTabsBar = document.getElementById('layer-tabs');
  LAYERS.forEach((l, idx) => {{
    const b = document.createElement('button');
    b.className = 'layer-btn' + (idx === 0 ? ' active' : '');
    b.innerText = l.name;
    b.onclick = () => switchLayer(idx);
    layerTabsBar.appendChild(b);
  }});

  function switchLayer(idx) {{
    curLayer = idx;
    curStep = 1;
    document.querySelectorAll('.layer-btn').forEach((b, i) => b.classList.toggle('active', i === idx));
    update();
  }}

  function speakPin(targetPin) {{
    if (!voiceEnabled || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance('на ' + targetPin);
    u.lang = 'ru-RU';
    u.rate = 1.15;
    window.speechSynthesis.speak(u);
  }}

  function calcLengthM(stepIdx) {{
    const path = getPath();
    let totalPx = 0;
    const mPerPx = (DIAMETER_CM / 100.0) / (500.0 * 0.93);
    for (let i = 0; i < stepIdx; i++) {{
      const p1 = PINS[path[i]], p2 = PINS[path[i+1]];
      totalPx += Math.hypot(p2[0]-p1[0], p2[1]-p1[1]);
    }}
    return totalPx * mPerPx;
  }}

  function update() {{
    const path = getPath();
    const total = getTotal();
    if (curStep > total) curStep = total;
    if (curStep < 1) curStep = 1;

    slider.max = total;
    slider.value = curStep;
    maxStepDisp.innerText = total;

    const fromP = path[curStep - 1];
    const toP = path[curStep];
    pinInfo.innerHTML = fromP + ' &rarr; ' + toP;
    stepCounter.innerText = curStep + ' / ' + total;
    percentInfo.innerText = ((curStep / total) * 100).toFixed(1) + '%';
    layerName.innerText = 'Слой: ' + LAYERS[curLayer].name;

    const curLen = calcLengthM(curStep);
    const totalLen = calcLengthM(total);
    lengthInfo.innerText = curLen.toFixed(1) + ' м';
    remainingInfo.innerText = 'Осталось: ~' + Math.max(0, totalLen - curLen).toFixed(1) + ' м';

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const scale = canvas.width / 500.0;

    ctx.strokeStyle = LAYERS[curLayer].color || '#111111';
    ctx.lineWidth = 1.3 * scale;
    ctx.globalAlpha = 0.32;
    ctx.beginPath();
    for (let i = 0; i < curStep; i++) {{
      const a = PINS[path[i]], b = PINS[path[i+1]];
      ctx.moveTo(a[0] * scale, a[1] * scale);
      ctx.lineTo(b[0] * scale, b[1] * scale);
    }}
    ctx.stroke();

    ctx.globalAlpha = 1.0;
    ctx.strokeStyle = '#00ffd5';
    ctx.lineWidth = 3.6 * scale;
    ctx.shadowColor = '#00ffd5';
    ctx.shadowBlur = 10 * scale;
    const a = PINS[fromP], b = PINS[toP];
    ctx.beginPath();
    ctx.moveTo(a[0] * scale, a[1] * scale);
    ctx.lineTo(b[0] * scale, b[1] * scale);
    ctx.stroke();
    ctx.shadowBlur = 0;

    ctx.fillStyle = '#ff2a5f';
    ctx.beginPath(); ctx.arc(a[0] * scale, a[1] * scale, 6 * scale, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#00ffd5';
    ctx.beginPath(); ctx.arc(b[0] * scale, b[1] * scale, 7 * scale, 0, Math.PI * 2); ctx.fill();

    try {{ localStorage.setItem('stringart_step_' + curLayer, curStep); }} catch(e) {{}}
  }}

  function next(d = 1) {{
    const prev = curStep;
    curStep = Math.min(getTotal(), curStep + d);
    if (curStep !== prev) {{
      const path = getPath();
      speakPin(path[curStep]);
      update();
    }}
  }}

  function prev(d = 1) {{
    curStep = Math.max(1, curStep - d);
    update();
  }}

  slider.oninput = (e) => {{ curStep = parseInt(e.target.value); update(); }};
  document.getElementById('btn-next').onclick = () => next(1);
  document.getElementById('btn-prev').onclick = () => prev(1);
  document.getElementById('btn-next-10').onclick = () => next(10);
  document.getElementById('btn-prev-10').onclick = () => prev(10);
  document.getElementById('btn-reset').onclick = () => {{ curStep = 1; update(); }};
  btnVoice.onclick = () => {{
    voiceEnabled = !voiceEnabled;
    btnVoice.innerText = voiceEnabled ? '&#128266; Голос: ВКЛ' : '&#128263; Голос: ВЫКЛ';
  }};

  btnPlay.onclick = () => {{
    isPlaying = !isPlaying;
    if (isPlaying) {{
      btnPlay.innerText = '&#10074;&#10074; Пауза';
      btnPlay.style.background = '#ffaa00';
      playInterval = setInterval(() => {{
        if (curStep >= getTotal()) {{ btnPlay.click(); return; }}
        next(1);
      }}, 3200);
    }} else {{
      btnPlay.innerText = '&#9654; Авто';
      btnPlay.style.background = '';
      clearInterval(playInterval);
    }}
  }};

  window.addEventListener('keydown', (e) => {{
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'd' || e.key === 'D') {{
      e.preventDefault(); next(1);
    }} else if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {{
      e.preventDefault(); prev(1);
    }}
  }});

  try {{
    const saved = localStorage.getItem('stringart_step_' + curLayer);
    if (saved) curStep = parseInt(saved);
  }} catch(e) {{}}

  update();
</script>
</body>
</html>"""
    return html

# ----------------------------------------------------------------------
# РАЗДЕЛ 9: ТЕМАТИЧЕСКИЙ ГРАФИЧЕСКИЙ ИНТЕРФЕЙС И СОСТОЯНИЕ
# ----------------------------------------------------------------------

class ToolTip:
    def __init__(self, widget, text=''):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert") if hasattr(self.widget, "bbox") else (0, 0, 0, 0)
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify='left',
                         background="#252736", foreground="#e2e4f0", relief='solid', borderwidth=1,
                         font=("Segoe UI", 9))
        label.pack(ipadx=6, ipady=4)

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

class AppState:
    def __init__(self):
        self.original_image: Optional[Image.Image] = None
        self.canvas_manager: Optional[CanvasManager] = None
        self.radon_guide: Optional[RadonGuide] = None
        self.preprocessed_data: Optional[Dict[str, Any]] = None
        self.layers: List[Dict[str, Any]] = []
        self.current_layer_idx: int = 0
        self.current_step: int = 0
        self.is_running: bool = False
        self.is_paused: bool = False

        self.vars: Dict[str, tk.Variable] = {}
        self._init_variables()

    def _init_variables(self):
        self.vars['preset'] = tk.StringVar(value="Портрет (Ultra Detail)")
        self.vars['canvas_shape'] = tk.StringVar(value="Круг")
        self.vars['pins'] = tk.IntVar(value=288)
        self.vars['lines'] = tk.IntVar(value=2800)
        self.vars['line_weight'] = tk.IntVar(value=72)
        self.vars['fearless_factor'] = tk.IntVar(value=92)
        self.vars['clahe_clip'] = tk.DoubleVar(value=2.2)
        self.vars['gamma'] = tk.DoubleVar(value=0.90)
        self.vars['contour_weight'] = tk.IntVar(value=60)
        self.vars['radon_influence'] = tk.IntVar(value=35)
        self.vars['color_mode'] = tk.StringVar(value="Монохром (Черная нить)")
        self.vars['physical_diameter'] = tk.DoubleVar(value=60.0)
        self.vars['invert_colors'] = tk.BooleanVar(value=False)
        self.vars['voice_enabled'] = tk.BooleanVar(value=False)
        self.vars['step_scrubber'] = tk.IntVar(value=1)
        self.vars['split_pos'] = tk.DoubleVar(value=0.5)

    def apply_preset(self, name: str):
        if name in PRESETS:
            p = PRESETS[name]
            for k, v in p.items():
                if k in self.vars:
                    self.vars[k].set(v)


# ----------------------------------------------------------------------
# РАЗДЕЛ 10: ГЛАВНОЕ ОКНО СТУДИИ STRING ART PRO
# ----------------------------------------------------------------------

class StringArtStudioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("String Art Studio Pro v9.0 — Ultra High-Performance Engine")
        self.geometry("1480x920")
        self.minsize(1100, 750)

        ico_path = os.path.join(os.path.dirname(__file__), "StringArt.ico")
        if os.path.exists(ico_path):
            try:
                self.iconbitmap(ico_path)
            except Exception:
                pass

        self.state = AppState()
        self.audio_assistant = AudioAssistant()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()

        self.render_queue: queue.Queue = queue.Queue()
        self.ui_update_queue: queue.Queue = queue.Queue()
        self._start_renderer_worker()

        self._setup_modern_theme()
        self._create_layout()
        self._bind_events()
        self._periodic_ui_poll()

    def _setup_modern_theme(self):
        style = ttk.Style(self)
        style.theme_use('clam')

        self.configure(bg="#12131a")
        style.configure(".", background="#12131a", foreground="#e2e4f0", font=("Segoe UI", 10))
        style.configure("TFrame", background="#12131a")
        style.configure("Sidebar.TFrame", background="#1a1c26")
        style.configure("Card.TFrame", background="#1f2230", relief="solid", borderwidth=1)
        style.configure("TLabel", background="#12131a", foreground="#e2e4f0")
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), foreground="#4fa3ff", background="#1a1c26")
        style.configure("SubHeader.TLabel", font=("Segoe UI", 9), foreground="#8e93aa", background="#1a1c26")
        style.configure("Meta.TLabel", font=("Segoe UI", 9), foreground="#00ffd5", background="#1a1c26")

        style.configure("TButton", background="#2a2d3d", foreground="#ffffff", borderwidth=1, relief="flat", padding=6)
        style.map("TButton", background=[("active", "#4fa3ff"), ("pressed", "#2570d4")], foreground=[("active", "#000000")])
        style.configure("Accent.TButton", background="#4fa3ff", foreground="#000000", font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#70b5ff"), ("pressed", "#3b8ced")])
        style.configure("Stop.TButton", background="#ff3366", foreground="#ffffff", font=("Segoe UI", 10, "bold"))
        style.map("Stop.TButton", background=[("active", "#ff5c85")])

        style.configure("TNotebook", background="#12131a", tabmargins=[4, 4, 4, 0])
        style.configure("TNotebook.Tab", background="#1e202c", foreground="#b0b4c8", padding=[12, 6], font=("Segoe UI", 10))
        style.map("TNotebook.Tab", background=[("selected", "#2c3044")], foreground=[("selected", "#00ffd5")])

        style.configure("Horizontal.TProgressbar", background="#00ffd5", troughcolor="#1a1c26", borderwidth=0)

    def _create_layout(self):
        self.main_container = ttk.Frame(self)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        self._create_sidebar()
        self._create_display_tabs()
        self._create_status_bar()

    def _create_sidebar(self):
        sidebar_outer = ttk.Frame(self.main_container, style="Sidebar.TFrame", width=380)
        sidebar_outer.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        sidebar_outer.pack_propagate(False)

        canvas = tk.Canvas(sidebar_outer, bg="#1a1c26", highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar_outer, orient="vertical", command=canvas.yview)
        self.sidebar_content = ttk.Frame(canvas, style="Sidebar.TFrame")

        self.sidebar_content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.sidebar_content, anchor="nw", width=360)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        pad = {'padx': 10, 'pady': 6}

        logo_frame = ttk.Frame(self.sidebar_content, style="Sidebar.TFrame")
        logo_frame.pack(fill=tk.X, **pad)
        ttk.Label(logo_frame, text="STRING ART STUDIO", font=("Segoe UI", 14, "bold"), foreground="#4fa3ff", background="#1a1c26").pack(anchor="w")
        ttk.Label(logo_frame, text=f"Движок v9.0 Pro | GPU: {CUDA_DEVICE_NAME}", style="SubHeader.TLabel").pack(anchor="w")

        load_btn = ttk.Button(self.sidebar_content, text="📁 Загрузить изображение...", style="Accent.TButton", command=self.load_image)
        load_btn.pack(fill=tk.X, **pad)
        ToolTip(load_btn, TOOLTIPS['load_image'])

        preset_box = ttk.LabelFrame(self.sidebar_content, text=" Быстрые пресеты ", padding=8)
        preset_box.pack(fill=tk.X, **pad)
        cb_preset = ttk.Combobox(preset_box, textvariable=self.state.vars['preset'], values=list(PRESETS.keys()), state="readonly")
        cb_preset.pack(fill=tk.X)
        cb_preset.bind("<<ComboboxSelected>>", lambda e: self.state.apply_preset(self.state.vars['preset'].get()))

        geom_box = ttk.LabelFrame(self.sidebar_content, text=" Геометрия и холст ", padding=8)
        geom_box.pack(fill=tk.X, **pad)

        ttk.Label(geom_box, text="Форма холста:").pack(anchor="w")
        shapes = ["Круг", "Квадрат", "Прямоугольник (4:3)", "Прямоугольник (16:9)", "Овал", "Шестиугольник"]
        cb_shape = ttk.Combobox(geom_box, textvariable=self.state.vars['canvas_shape'], values=shapes, state="readonly")
        cb_shape.pack(fill=tk.X, pady=2)

        self._add_slider(geom_box, "Количество гвоздей:", self.state.vars['pins'], 120, 480, 1)
        self._add_slider(geom_box, "Диаметр картины (см):", self.state.vars['physical_diameter'], 30.0, 120.0, 5.0)

        algo_box = ttk.LabelFrame(self.sidebar_content, text=" Параметры плетения ", padding=8)
        algo_box.pack(fill=tk.X, **pad)

        ttk.Label(algo_box, text="Режим цвета:").pack(anchor="w")
        cb_color = ttk.Combobox(algo_box, textvariable=self.state.vars['color_mode'],
                                values=["Монохром (Черная нить)", "Цветной CMYK (4 нити)"], state="readonly")
        cb_color.pack(fill=tk.X, pady=2)

        self._add_slider(algo_box, "Число нитей на слой:", self.state.vars['lines'], 400, 5000, 100)
        self._add_slider(algo_box, "Плотность нити (%):", self.state.vars['line_weight'], 30, 95, 1)
        self._add_slider(algo_box, "Защита белого (%):", self.state.vars['fearless_factor'], 50, 99, 1)
        self._add_slider(algo_box, "Вес Радона / FBP (%):", self.state.vars['radon_influence'], 0, 80, 5)
        self._add_slider(algo_box, "Контраст CLAHE:", self.state.vars['clahe_clip'], 0.5, 5.0, 0.1)
        self._add_slider(algo_box, "Гамма тона:", self.state.vars['gamma'], 0.5, 1.8, 0.05)

        player_box = ttk.LabelFrame(self.sidebar_content, text=" Интерактивный плеер плетения ", padding=8)
        player_box.pack(fill=tk.X, **pad)

        scrub_frame = ttk.Frame(player_box)
        scrub_frame.pack(fill=tk.X, pady=2)
        ttk.Label(scrub_frame, text="Шаг нити:").pack(side=tk.LEFT)
        self.lbl_step_num = ttk.Label(scrub_frame, text="0 / 0", style="Meta.TLabel")
        self.lbl_step_num.pack(side=tk.RIGHT)

        self.scrubber = ttk.Scale(player_box, from_=1, to=100, variable=self.state.vars['step_scrubber'],
                                  command=self._on_scrubber_change)
        self.scrubber.pack(fill=tk.X, pady=4)

        nav_btn_frame = ttk.Frame(player_box)
        nav_btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(nav_btn_frame, text="◀ Назад", width=8, command=lambda: self._step_nav(-1)).pack(side=tk.LEFT, padx=2)
        self.btn_auto_step = ttk.Button(nav_btn_frame, text="▶ Авто", width=8, command=self._toggle_auto_play)
        self.btn_auto_step.pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_btn_frame, text="Вперед ▶", width=8, command=lambda: self._step_nav(1)).pack(side=tk.LEFT, padx=2)

        chk_voice = ttk.Checkbutton(player_box, text="🔊 Озвучивать гвозди (SAPI)",
                                    variable=self.state.vars['voice_enabled'], command=self._on_voice_toggle)
        chk_voice.pack(anchor="w", pady=4)

        btn_frame = ttk.Frame(self.sidebar_content, style="Sidebar.TFrame")
        btn_frame.pack(fill=tk.X, **pad)

        self.btn_start = ttk.Button(btn_frame, text="🚀 СТАРТ СИНТЕЗА", style="Accent.TButton", command=self.start_generation)
        self.btn_start.pack(fill=tk.X, pady=3)

        row2 = ttk.Frame(btn_frame, style="Sidebar.TFrame")
        row2.pack(fill=tk.X, pady=2)
        self.btn_pause = ttk.Button(row2, text="⏸ Пауза", command=self.toggle_pause, state="disabled")
        self.btn_pause.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.btn_stop = ttk.Button(row2, text="⏹ Стоп", style="Stop.TButton", command=self.stop_generation, state="disabled")
        self.btn_stop.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)

        self.btn_polish = ttk.Button(btn_frame, text="✨ Эволюционная доводка", command=self.run_polishing, state="disabled")
        self.btn_polish.pack(fill=tk.X, pady=3)

        export_box = ttk.LabelFrame(self.sidebar_content, text=" Экспорт результатов ", padding=8)
        export_box.pack(fill=tk.X, **pad)

        btn_html = ttk.Button(export_box, text="🌐 Веб-ассистент (HTML5)", command=self.export_html_action)
        btn_html.pack(fill=tk.X, pady=2)
        ToolTip(btn_html, TOOLTIPS['export_html'])

        row_exp1 = ttk.Frame(export_box)
        row_exp1.pack(fill=tk.X, pady=2)
        ttk.Button(row_exp1, text="📄 TXT список", command=self.export_txt_action).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(row_exp1, text="📊 CSV таблица", command=self.export_csv_action).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)

        row_exp2 = ttk.Frame(export_box)
        row_exp2.pack(fill=tk.X, pady=2)
        ttk.Button(row_exp2, text="⚙️ CNC G-Code", command=self.export_gcode_action).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(row_exp2, text="📐 Вектор SVG", command=self.export_svg_action).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)

        row_exp3 = ttk.Frame(export_box)
        row_exp3.pack(fill=tk.X, pady=2)
        ttk.Button(row_exp3, text="🖼 Мастер PNG", command=self.export_png_action).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(row_exp3, text="⭕ Шаблон 1:1", command=self.export_template_action).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)

    def _add_slider(self, parent, label_text, var, min_val, max_val, step):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=3)
        row_top = ttk.Frame(frame)
        row_top.pack(fill=tk.X)
        ttk.Label(row_top, text=label_text).pack(side=tk.LEFT)
        val_lbl = ttk.Label(row_top, text=str(var.get()), style="Meta.TLabel")
        val_lbl.pack(side=tk.RIGHT)

        def on_slide(val):
            if isinstance(var, tk.IntVar):
                v = int(round(float(val)))
                var.set(v)
                val_lbl.config(text=str(v))
            else:
                v = round(float(val), 2)
                var.set(v)
                val_lbl.config(text=f"{v:.2f}")

        scale = ttk.Scale(frame, from_=min_val, to=max_val, value=var.get(), command=on_slide)
        scale.pack(fill=tk.X, pady=2)

    def _create_display_tabs(self):
        tabs_container = ttk.Frame(self.main_container)
        tabs_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=6)

        self.notebook = ttk.Notebook(tabs_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_result = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_result, text="🎨 Результат")
        self.canvas_result = tk.Canvas(self.tab_result, bg="#12131a", highlightthickness=0)
        self.canvas_result.pack(fill=tk.BOTH, expand=True)
        self.canvas_result.bind("<Configure>", lambda e: self.request_rerender())

        self.tab_split = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_split, text="🔀 До / После")
        self._init_split_view()

        self.tab_diff = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_diff, text="🌡 Карта ошибок")
        self.canvas_diff = tk.Canvas(self.tab_diff, bg="#12131a", highlightthickness=0)
        self.canvas_diff.pack(fill=tk.BOTH, expand=True)

        self.tab_density = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_density, text="📊 Плотность нитей")
        self.canvas_density = tk.Canvas(self.tab_density, bg="#12131a", highlightthickness=0)
        self.canvas_density.pack(fill=tk.BOTH, expand=True)

        self.tab_radon = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_radon, text="🌀 Преобразование Радона")
        self.fig_radon = Figure(figsize=(7, 5), facecolor="#12131a")
        self.canvas_radon = FigureCanvasTkAgg(self.fig_radon, master=self.tab_radon)
        self.canvas_radon.get_widget().pack(fill=tk.BOTH, expand=True)

        self.tab_stats = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_stats, text="📈 Сходимость")
        self.fig_stats = Figure(figsize=(7, 5), facecolor="#12131a")
        self.canvas_stats = FigureCanvasTkAgg(self.fig_stats, master=self.tab_stats)
        self.canvas_stats.get_widget().pack(fill=tk.BOTH, expand=True)

    def _init_split_view(self):
        ctrl_bar = ttk.Frame(self.tab_split)
        ctrl_bar.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(ctrl_bar, text="Разделитель сравнения: Оригинал ◀ ─── ▶ String Art").pack(side=tk.LEFT)
        scale_split = ttk.Scale(ctrl_bar, from_=0.0, to=1.0, variable=self.state.vars['split_pos'],
                                command=lambda e: self.update_split_display())
        scale_split.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=12)

        self.canvas_split = tk.Canvas(self.tab_split, bg="#12131a", highlightthickness=0)
        self.canvas_split.pack(fill=tk.BOTH, expand=True)

    def _create_status_bar(self):
        status_bar = ttk.Frame(self, style="Sidebar.TFrame", height=32)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_status = ttk.Label(status_bar, text="Готов к работе. Загрузите изображение.", background="#1a1c26")
        self.lbl_status.pack(side=tk.LEFT, padx=12, pady=4)

        self.lbl_speed = ttk.Label(status_bar, text="Скорость: 0 нитей/сек", style="Meta.TLabel")
        self.lbl_speed.pack(side=tk.RIGHT, padx=12, pady=4)

        self.lbl_metrics = ttk.Label(status_bar, text="Расход нити: 0.0 м", style="Meta.TLabel")
        self.lbl_metrics.pack(side=tk.RIGHT, padx=12, pady=4)

        self.prog_bar = ttk.Progressbar(status_bar, orient="horizontal", mode="determinate", length=220)
        self.prog_bar.pack(side=tk.RIGHT, padx=12, pady=4)

    def _bind_events(self):
        self.bind("<space>", lambda e: self._step_nav(1))
        self.bind("<Right>", lambda e: self._step_nav(1))
        self.bind("<Left>", lambda e: self._step_nav(-1))
        self.bind("<Control-o>", lambda e: self.load_image())

    def _start_renderer_worker(self):
        def worker():
            while True:
                try:
                    task = self.render_queue.get()
                    while not self.render_queue.empty():
                        task = self.render_queue.get_nowait()

                    cm, layers, out_dim, weight, max_s, active_c = task
                    if cm is None:
                        continue

                    img_pil = cm.render_layers_to_pil(
                        layers, thickness=1, line_weight_percent=weight,
                        max_step=max_s, output_size=out_dim, active_chord=active_c
                    )
                    self.ui_update_queue.put(('render_complete', img_pil))
                except Exception as e:
                    logging.error(f"Ошибка в рендерере: {e}")

        t = threading.Thread(target=worker, name="RendererThread", daemon=True)
        t.start()

    def request_rerender(self):
        if self.state.canvas_manager is None or not self.state.layers:
            return

        w = self.canvas_result.winfo_width()
        h = self.canvas_result.winfo_height()
        dim = max(400, min(1400, min(w, h)))
        
        weight = self.state.vars['line_weight'].get()
        step = self.state.vars['step_scrubber'].get()

        active_chord = None
        cur_l = self.state.current_layer_idx
        if cur_l < len(self.state.layers):
            p = self.state.layers[cur_l].get('path', [])
            if 0 < step < len(p):
                active_chord = (p[step - 1], p[step])

        self.render_queue.put((self.state.canvas_manager, self.state.layers, dim, weight, step, active_chord))

    def _periodic_ui_poll(self):
        while not self.ui_update_queue.empty():
            try:
                msg_type, data = self.ui_update_queue.get_nowait()
                if msg_type == 'render_complete':
                    self._display_rendered_image(data)
                elif msg_type == 'status':
                    self.lbl_status.config(text=data)
                elif msg_type == 'progress':
                    step, rate, total = data
                    self.prog_bar['maximum'] = total
                    self.prog_bar['value'] = step
                    self.lbl_speed.config(text=f"Скорость: {rate:.0f} нитей/сек")
                    self.scrubber.config(to=total)
                    self.state.vars['step_scrubber'].set(step)
                    self.lbl_step_num.config(text=f"{step} / {total}")
                    self._update_length_metrics(step)
                elif msg_type == 'finished':
                    self._on_synthesis_finished()
            except queue.Empty:
                break

        self.after(35, self._periodic_ui_poll)

    def _display_rendered_image(self, img_pil: Image.Image):
        self.cached_rendered_pil = img_pil
        self.tk_rendered_img = ImageTk.PhotoImage(img_pil)
        
        self.canvas_result.delete("all")
        cw = self.canvas_result.winfo_width()
        ch = self.canvas_result.winfo_height()
        self.canvas_result.create_image(cw // 2, ch // 2, image=self.tk_rendered_img, anchor="center")

        self.update_split_display()

    def update_split_display(self):
        if not hasattr(self, 'cached_rendered_pil') or self.state.original_image is None:
            return

        w = self.canvas_split.winfo_width()
        h = self.canvas_split.winfo_height()
        dim = max(350, min(1200, min(w, h)))
        if dim < 10:
            return

        split = float(self.state.vars['split_pos'].get())
        split_x = int(dim * split)

        orig_resized = self.state.original_image.resize((dim, dim), Image.Resampling.LANCZOS).convert('RGB')
        gen_resized = self.cached_rendered_pil.resize((dim, dim), Image.Resampling.LANCZOS).convert('RGB')

        combined = Image.new('RGB', (dim, dim))
        if split_x > 0:
            combined.paste(orig_resized.crop((0, 0, split_x, dim)), (0, 0))
        if split_x < dim:
            combined.paste(gen_resized.crop((split_x, 0, dim, dim)), (split_x, 0))

        draw = ImageDraw.Draw(combined)
        draw.line([(split_x, 0), (split_x, dim)], fill="#00ffd5", width=3)

        self.tk_split_img = ImageTk.PhotoImage(combined)
        self.canvas_split.delete("all")
        self.canvas_split.create_image(w // 2, h // 2, image=self.tk_split_img, anchor="center")

    def _update_length_metrics(self, current_step: int):
        if self.state.canvas_manager is None or not self.state.layers:
            return
        cm = self.state.canvas_manager
        total_m = 0.0
        m_per_px = (cm.physical_cm / 100.0) / (cm.size * 0.93)
        
        for layer in self.state.layers:
            p = layer.get('path', [])
            count = min(len(p) - 1, current_step)
            for i in range(count):
                c1 = cm.pin_coords[p[i]]
                c2 = cm.pin_coords[p[i+1]]
                total_m += float(np.hypot(c2[0]-c1[0], c2[1]-c1[1])) * m_per_px

        self.lbl_metrics.config(text=f"Расход нити: {total_m:.1f} м (~{total_m * 3.28084:.0f} фут)")

    def load_image(self):
        filetypes = [("Изображения", "*.jpg *.jpeg *.png *.webp *.bmp"), ("Все файлы", "*.*")]
        p = filedialog.askopenfilename(title="Выберите исходное изображение", filetypes=filetypes)
        if not p:
            return

        try:
            pil_img = Image.open(p)
            self.state.original_image = pil_img
            self.lbl_status.config(text=f"Загружено: {os.path.basename(p)} ({pil_img.size[0]}x{pil_img.size[1]}px)")
            self._prepare_and_preview_canvas()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение:\n{e}")

    def _prepare_and_preview_canvas(self):
        if self.state.original_image is None:
            return

        shape = self.state.vars['canvas_shape'].get()
        pins = self.state.vars['pins'].get()
        dia = self.state.vars['physical_diameter'].get()

        self.state.canvas_manager = CanvasManager(IMAGE_SIZE, pins, shape, dia)
        self.state.canvas_manager.build_sparse_line_matrix()

        data = preprocess_image_pipeline(
            self.state.original_image, self.state.canvas_manager,
            clahe_clip=self.state.vars['clahe_clip'].get(),
            gamma=self.state.vars['gamma'].get(),
            color_mode=self.state.vars['color_mode'].get(),
            invert=self.state.vars['invert_colors'].get()
        )
        self.state.preprocessed_data = data

        self.state.radon_guide = RadonGuide(data['gray_processed'], self.state.canvas_manager)
        self._plot_radon_analysis()

        self.state.layers = []
        for ch in data['channels']:
            self.state.layers.append({
                "name": ch['name'],
                "color_hex": ch['color_hex'],
                "path": []
            })

        self.btn_start.config(state="normal")
        self.lbl_status.config(text="Изображение готово к синтезу. Нажмите 'СТАРТ СИНТЕЗА'.")

    def _plot_radon_analysis(self):
        rg = self.state.radon_guide
        if rg is None or rg.sinogram is None:
            return

        self.fig_radon.clear()
        ax1 = self.fig_radon.add_subplot(1, 2, 1)
        ax1.set_title("Синограмма Радона", color="#4fa3ff", fontsize=10)
        ax1.imshow(rg.sinogram, cmap='inferno', aspect='auto')
        ax1.axis('off')

        ax2 = self.fig_radon.add_subplot(1, 2, 2)
        ax2.set_title("FBP Реконструкция (Ramp)", color="#00ffd5", fontsize=10)
        ax2.imshow(rg.fbp_recon, cmap='gray')
        ax2.axis('off')

        self.canvas_radon.draw()

    def start_generation(self):
        if self.state.original_image is None:
            messagebox.showwarning("Внимание", "Сначала загрузите изображение.")
            return

        self.stop_event.clear()
        self.pause_event.set()
        self.state.is_running = True
        self.btn_start.config(state="disabled")
        self.btn_pause.config(state="normal", text="⏸ Пауза")
        self.btn_stop.config(state="normal")
        self.btn_polish.config(state="disabled")

        t = threading.Thread(target=self._synthesis_thread_proc, name="SynthesisThread", daemon=True)
        t.start()

    def _synthesis_thread_proc(self):
        try:
            self.ui_update_queue.put(('status', "Инициализация разреженной матрицы OMP..."))
            cm = self.state.canvas_manager
            rg = self.state.radon_guide
            data = self.state.preprocessed_data

            omp = FastOMPEngine(cm, rg)
            target_lines = self.state.vars['lines'].get()
            weight = self.state.vars['line_weight'].get()
            fearless = self.state.vars['fearless_factor'].get()
            radon_inf = self.state.vars['radon_influence'].get()
            contour_w = self.state.vars['contour_weight'].get()

            channels = data['channels']
            total_target = target_lines * len(channels)

            accumulated_layers = []
            for c_idx, ch in enumerate(channels):
                if self.stop_event.is_set():
                    break

                self.ui_update_queue.put(('status', f"Синтез слоя {c_idx + 1}/{len(channels)}: {ch['name']}..."))

                def prog_cb(step, rate, cur_path):
                    while not self.pause_event.is_set() and not self.stop_event.is_set():
                        time.sleep(0.08)

                    total_step = c_idx * target_lines + step
                    self.ui_update_queue.put(('progress', (total_step, rate, total_target)))
                    
                    temp_layers = list(accumulated_layers)
                    temp_layers.append({"name": ch['name'], "color_hex": ch['color_hex'], "path": cur_path})
                    self.state.layers = temp_layers
                    self.request_rerender()

                path = omp.synthesize_layer(
                    ch['darkness'], ch['importance'],
                    target_lines=target_lines, line_weight=weight,
                    fearless_factor=fearless, radon_influence=radon_inf,
                    contour_weight=contour_w,
                    stop_check=lambda: self.stop_event.is_set(),
                    progress_callback=prog_cb
                )

                accumulated_layers.append({
                    "name": ch['name'],
                    "color_hex": ch['color_hex'],
                    "path": path
                })

            self.state.layers = accumulated_layers
            self.ui_update_queue.put(('finished', None))

        except Exception as e:
            logging.error(f"Ошибка в процессе синтеза: {e}\n{traceback.format_exc()}")
            self.ui_update_queue.put(('status', f"Ошибка: {e}"))
            self.ui_update_queue.put(('finished', None))

    def _on_synthesis_finished(self):
        self.state.is_running = False
        self.btn_start.config(state="normal")
        self.btn_pause.config(state="disabled")
        self.btn_stop.config(state="disabled")
        self.btn_polish.config(state="normal")
        self.lbl_status.config(text="Синтез успешно завершен! Готово к интерактивному плетению и экспорту.")
        self.request_rerender()

        self._build_heatmaps()

    def _build_heatmaps(self):
        if not self.state.layers or self.state.canvas_manager is None:
            return
        cm = self.state.canvas_manager
        rendered_bgr = cm.render_layers_to_numpy(self.state.layers, output_size=cm.size)
        gray_gen = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2GRAY)
        
        target_gray = self.state.preprocessed_data['gray_processed']
        diff = gray_gen.astype(np.float32) - target_gray.astype(np.float32)

        diff_norm = np.clip(diff / 128.0, -1.0, 1.0)
        heatmap = np.zeros((cm.size, cm.size, 3), dtype=np.uint8)
        heatmap[diff_norm > 0, 2] = (diff_norm[diff_norm > 0] * 255).astype(np.uint8)
        heatmap[diff_norm < 0, 0] = (-diff_norm[diff_norm < 0] * 255).astype(np.uint8)

        img_diff = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        self.tk_diff_img = ImageTk.PhotoImage(img_diff.resize((500, 500)))
        self.canvas_diff.delete("all")
        self.canvas_diff.create_image(250, 250, image=self.tk_diff_img)

    def toggle_pause(self):
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.btn_pause.config(text="▶ Продолжить")
            self.lbl_status.config(text="Синтез приостановлен.")
        else:
            self.pause_event.set()
            self.btn_pause.config(text="⏸ Пауза")
            self.lbl_status.config(text="Синтез возобновлен.")

    def stop_generation(self):
        self.stop_event.set()
        self.pause_event.set()
        self.lbl_status.config(text="Остановка синтеза...")

    def run_polishing(self):
        if not self.state.layers or self.state.canvas_manager is None:
            return
        
        self.lbl_status.config(text="Локальная эволюционная доводка пути (Delta Descent)...")
        cm = self.state.canvas_manager
        evo = EvolutionEngine(cm, self.state.preprocessed_data['channels'][0]['darkness'])
        
        for layer in self.state.layers:
            layer['path'] = evo.local_delta_descent(layer['path'], iterations=180)
        
        self.request_rerender()
        self.lbl_status.config(text="Эволюционная доводка завершена! Качество линий улучшено.")

    def _on_scrubber_change(self, val):
        step = int(round(float(val)))
        total = int(self.scrubber['to'])
        self.lbl_step_num.config(text=f"{step} / {total}")
        self._update_length_metrics(step)

        if self.state.vars['voice_enabled'].get() and self.state.layers:
            p = self.state.layers[self.state.current_layer_idx].get('path', [])
            if 0 < step < len(p):
                self.audio_assistant.announce_chord(p[step - 1], p[step])

        self.request_rerender()

    def _step_nav(self, delta: int):
        cur = self.state.vars['step_scrubber'].get()
        max_s = int(self.scrubber['to'])
        new_s = max(1, min(max_s, cur + delta))
        if new_s != cur:
            self.state.vars['step_scrubber'].set(new_s)
            self._on_scrubber_change(new_s)

    def _toggle_auto_play(self):
        if hasattr(self, '_auto_play_job') and self._auto_play_job:
            self.after_cancel(self._auto_play_job)
            self._auto_play_job = None
            self.btn_auto_step.config(text="▶ Авто", style="TButton")
        else:
            self.btn_auto_step.config(text="⏸ Стоп", style="Accent.TButton")
            self._auto_play_step()

    def _auto_play_step(self):
        cur = self.state.vars['step_scrubber'].get()
        max_s = int(self.scrubber['to'])
        if cur < max_s:
            self._step_nav(1)
            self._auto_play_job = self.after(3200, self._auto_play_step)
        else:
            self._toggle_auto_play()

    def _on_voice_toggle(self):
        self.audio_assistant.toggle(self.state.vars['voice_enabled'].get())

    def export_html_action(self):
        if not self.state.layers:
            messagebox.showwarning("Внимание", "Сначала выполните синтез картины.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML Веб-ассистент", "*.html")],
                                         title="Сохранить автономный веб-ассистент")
        if p:
            html = build_interactive_html5_assistant("String Art Masterpiece", self.state.canvas_manager.shape,
                                                    self.state.canvas_manager.pin_coords, self.state.layers,
                                                    self.state.canvas_manager.physical_cm)
            with open(p, 'w', encoding='utf-8') as f:
                f.write(html)
            messagebox.showinfo("Готово", f"Интерактивный веб-ассистент успешно сохранен в:\n{p}\n\nВы можете открыть его в любом браузере на ПК, планшете или телефоне!")

    def export_txt_action(self):
        if not self.state.layers: return
        p = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Текстовая инструкция", "*.txt")],
                                         title="Сохранить текстовую инструкцию")
        if p:
            export_instructions_txt(p, self.state.layers, self.state.canvas_manager)
            messagebox.showinfo("Готово", f"Инструкция сохранена в:\n{p}")

    def export_csv_action(self):
        if not self.state.layers: return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Таблица", "*.csv")],
                                         title="Сохранить таблицу шагов CSV")
        if p:
            with open(p, 'w', encoding='utf-8') as f:
                f.write("Слой,Шаг,От гвоздя,К гвоздю\n")
                for l_idx, l in enumerate(self.state.layers):
                    path = l.get('path', [])
                    for i in range(len(path) - 1):
                        f.write(f"{l.get('name')},{i+1},{path[i]},{path[i+1]}\n")
            messagebox.showinfo("Готово", f"Таблица CSV сохранена в:\n{p}")

    def export_gcode_action(self):
        if not self.state.layers: return
        p = filedialog.asksaveasfilename(defaultextension=".gcode", filetypes=[("ЧПУ G-Code", "*.gcode *.nc")],
                                         title="Сохранить G-Code для станка")
        if p:
            export_gcode_cnc(p, self.state.layers, self.state.canvas_manager)
            messagebox.showinfo("Готово", f"Программа G-Code сохранена в:\n{p}")

    def export_svg_action(self):
        if not self.state.layers: return
        p = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("Векторный SVG", "*.svg")],
                                         title="Сохранить векторный файл SVG")
        if p:
            export_svg_vector(p, self.state.layers, self.state.canvas_manager)
            messagebox.showinfo("Готово", f"Векторный файл SVG сохранен в:\n{p}")

    def export_png_action(self):
        if not self.state.layers: return
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Печатный растр Ultra-HD", "*.png")],
                                         title="Сохранить печатное изображение PNG")
        if p:
            self.lbl_status.config(text="Рендеринг Master PNG высокого разрешения...")
            self.update_idletasks()
            img_hi = self.state.canvas_manager.render_layers_to_pil(
                self.state.layers, thickness=1, line_weight_percent=self.state.vars['line_weight'].get(),
                output_size=3000
            )
            img_hi.save(p, "PNG")
            self.lbl_status.config(text="Печатный PNG успешно сохранен.")
            messagebox.showinfo("Готово", f"Мастер-файл Ultra-HD PNG сохранен в:\n{p}")

    def export_template_action(self):
        if self.state.canvas_manager is None: return
        p = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("Векторный чертеж 1:1", "*.svg")],
                                         title="Сохранить шаблон разметки 1:1")
        if p:
            export_drill_template_svg(p, self.state.canvas_manager)
            messagebox.showinfo("Готово", f"Шаблон для сверления сохранен в:\n{p}")

def main():
    try:
        app = StringArtStudioApp()
        app.mainloop()
    except Exception as e:
        logging.critical(f"Критическая ошибка приложения: {e}\n{traceback.format_exc()}")
        try:
            messagebox.showerror("Критическая ошибка", f"{e}")
        except Exception:
            pass

if __name__ == "__main__":
    main()
