# -*- coding: utf-8 -*-
"""
String Art Studio Pro v9.0 Full Studio (Island GA + Live Analytics + Fast Fallback + Light Theme)
Полная версия со всеми элементами управления, аналитическими вкладками, 
островным генетическим алгоритмом, дельта-меметикой, синограммой Радона 
и идеально сглаженным (4x SSAA Subpixel AA) рендером угольно-черной нити без "каши".
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import queue
import os
import sys
import traceback
import threading
import time
import math
import io
import random
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable

from PIL import Image, ImageTk
import numpy as np
import cv2
import tempfile
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'matplotlib_stringart'))
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

try:
    from skimage.metrics import structural_similarity as ssim_cpu
except ImportError:
    ssim_cpu = None

try:
    from skimage.transform import radon
    SKIMAGE_AVAILABLE = True
except ImportError:
    radon, SKIMAGE_AVAILABLE = None, False

try:
    from scipy.ndimage import distance_transform_edt
    SCIPY_AVAILABLE = True
except ImportError:
    distance_transform_edt, SCIPY_AVAILABLE = None, False

try:
    import torch
    GPU_ENABLED_FLAG = torch.cuda.is_available()
except ImportError:
    torch, GPU_ENABLED_FLAG = None, False

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    njit = lambda *args, **kwargs: (lambda f: f)
    NUMBA_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

IMAGE_SIZE = 500
MIN_PIN_DISTANCE = 20
PATIENCE_BASE = 22
PATIENCE_STAGE_FACTOR = 14
STAGNATION_MUTATION_FACTOR = 2.0
BASE_MUTATION_DIVISOR = 50.0

TOOLTIPS = {
    'load_image': "Открыть файл изображения (JPG, PNG, WEBP).\nРекомендуются контрастные портреты с четким фоном.",
    'clahe_clip': "Локальное усиление контраста (CLAHE).\nВыделяет глаза, губы и мелкие детали.",
    'gamma': "Гамма-коррекция (< 1.0 осветляет средние тона и убирает лишнюю грязь/кашу).",
    'canvas_shape': "Форма рамы для расположения гвоздей (Круг или Квадрат).",
    'pins': "Количество гвоздей по периметру.\nРекомендуется: 240-320.",
    'lines': "Максимальное количество нитей.\nДля черной нити без каши рекомендуется: 950-1850.",
    'line_weight': "Чернота (оптическая плотность) нити (%).\nВысокие значения дают четкую угольно-черную нить.",
    'thread_thickness': "Толщина рендера линий при отображении и экспорте.",
    'fearless_factor': "Защита белых зон от перетемнения (%).\nПредотвращает появление серой 'каши' на светлых участках лица и фона.",
    'num_islands': "Количество независимых потоков (островов) генетического алгоритма.",
    'population_size': "Размер популяции на каждом острове.",
    'stages': "Количество этапов пошаговой эволюционной доводки.",
    'mutation_rate': "Базовая вероятность мутации (%).",
    'migration_interval': "Интервал обмена лучшими решениями между островами (в поколениях).",
    'structural_metric': "Метрика точности попадания нитей в контуры изображения.",
    'contour_weight': "Вес контуров и важных черт лица (%).",
    'memetic_enabled': "Включить сверхбыстрый дельта-поиск (меметическая полировка каждого витка).",
    'memetic_intensity': "Доля лучших особей популяции для локальной полировки (%).",
    'memetic_depth': "Глубина итераций локального дельта-поиска.",
    'radon_enabled': "Использовать томографическое преобразование Радона (с фильтром Рам-Лака) для глобального наведения.",
    'radon_influence': "Сила влияния синограммы Радона при выборе нитей (%).",
    'step_scrubber': "Интерактивный просмотр плетения от 1-й до последней нити в реальном времени.",
    'split_compare': "Шторка сравнения 'До / После' прямо поверх результата (0% - только нити, 50% - половина оригинала).",
    'start': "Запустить полный гибридный синтез (OMP + Островной ГА + Дельта-поиск).",
    'polish': "Запустить эволюционную доводку (Островной ГА) на базе текущей нити без перезапуска OMP.",
    'pause': "Приостановить или возобновить эволюцию.",
    'stop': "Остановить генерацию и зафиксировать текущий лучший результат.",
    'prune': "Оптимизировать путь нити, удалив холостые возвратные петли A->B->A.",
    'export_txt': "Экспортировать пошаговую инструкцию гвоздей в .txt с расчетом метража.",
    'export_png': "Экспортировать изображение в ультра-высоком разрешении (2400x2400px) с 4x сглаживанием.",
    'export_svg': "Экспортировать векторный чертеж в формате .svg.",
    'export_html': "Экспортировать интерактивный автономный веб-ассистент (.html) для физической сборки.",
}


# ----------------------------------------------------------------------
# РАЗДЕЛ 1: NUMBA-ЯДРА ВЫСОКОЙ ТОЧНОСТИ И СКОРОСТИ
# ----------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def evaluate_line_omp(residual: np.ndarray, target_darkness: np.ndarray, importance: np.ndarray, 
                      indices: np.ndarray, line_alpha: float, fearless_penalty: float) -> float:
    """
    Строгая оценка линии по истинному уменьшению взвешенной ошибки (Delta L2) 
    с жесткой защитой светлых участков (white_guard) от 'каши' и грязи.
    """
    n = len(indices)
    if n == 0:
        return -1e12
    
    net_improvement = 0.0
    bright_zone_hits = 0
    
    for i in range(n):
        idx = indices[i]
        res = residual[idx]
        tar = target_darkness[idx]
        imp = importance[idx]
        
        if res > 0.03:
            after_res = res - line_alpha
            if after_res >= 0.0:
                delta = (res * res - after_res * after_res) * imp
            else:
                good_part = (res * res) * imp
                bad_part = (after_res * after_res) * imp * (1.2 + fearless_penalty * 2.5)
                delta = good_part - bad_part
            net_improvement += delta
        else:
            over = line_alpha - res
            white_guard = 3.2 if tar < 0.12 else 1.6
            if tar < 0.08:
                bright_zone_hits += 1
            penalty = (over * over) * imp * fearless_penalty * white_guard
            net_improvement -= penalty

    bright_ratio = bright_zone_hits / n
    if bright_ratio > 0.55:
        net_improvement -= (bright_ratio - 0.55) * n * (line_alpha * line_alpha) * fearless_penalty * 2.0

    return net_improvement / (n ** 0.92)


@njit(fastmath=True, cache=True)
def update_residual_omp(residual: np.ndarray, indices: np.ndarray, line_alpha: float):
    for i in range(len(indices)):
        residual[indices[i]] -= line_alpha


@njit(fastmath=True, cache=True)
def fast_evaluate_chromosome_1d(chromo_arr: np.ndarray, line_indices_flat: np.ndarray,
                                line_offsets: np.ndarray, line_lengths: np.ndarray,
                                num_pins: int, flat_size: int, line_alpha: float,
                                target_darkness: np.ndarray, importance: np.ndarray,
                                fearless_penalty: float) -> float:
    """
    Сверхбыстрая оценка полной хромосомы в 1D (0.4 мс) без вызова OpenCV в цикле ГА.
    """
    optical_depth = np.zeros(flat_size, dtype=np.float32)
    n_lines = len(chromo_arr) - 1
    for i in range(n_lines):
        p1 = chromo_arr[i]
        p2 = chromo_arr[i + 1]
        if p1 == p2:
            continue
        if p1 > p2:
            p1, p2 = p2, p1
        pair_idx = p1 * num_pins + p2
        offset = line_offsets[pair_idx]
        length = line_lengths[pair_idx]
        for k in range(offset, offset + length):
            optical_depth[line_indices_flat[k]] += line_alpha

    total_err = 0.0
    over_mult = 1.0 + fearless_penalty * 1.8
    for idx in range(flat_size):
        d = optical_depth[idx]
        # Быстрая аппроксимация экспоненты поглощения света 1 - exp(-d * 1.6)
        cur_dark = 1.0 if d > 2.2 else (1.0 - 1.0 / (1.0 + d * 1.5 + 0.5 * d * d))
        tar = target_darkness[idx]
        diff = tar - cur_dark
        if cur_dark > tar:
            total_err += (diff * diff) * importance[idx] * over_mult
        else:
            total_err += (diff * diff) * importance[idx]
    return 1.0 - (total_err / flat_size)


@njit(fastmath=True, cache=True)
def compute_delta_hinge_error(optical_depth: np.ndarray, target_darkness: np.ndarray, importance: np.ndarray,
                              rem1: np.ndarray, rem2: np.ndarray, add1: np.ndarray, add2: np.ndarray,
                              line_alpha: float, fearless_penalty: float) -> float:
    """
    Мгновенно вычисляет разницу ошибки (Delta L2) при замене узла A -> B -> C на A -> X -> C.
    Отрицательное значение означает улучшение (уменьшение ошибки).
    """
    delta_err = 0.0
    over_mult = 1.0 + fearless_penalty * 1.8

    # Вспомогательный двухпроходный расчет: сначала вычитаем старую ошибку и обновляем оптическую плотность
    for arr_idx in range(4):
        indices = rem1 if arr_idx == 0 else (rem2 if arr_idx == 1 else (add1 if arr_idx == 2 else add2))
        shift = -line_alpha if arr_idx < 2 else line_alpha
        for i in range(len(indices)):
            px = indices[i]
            old_d = optical_depth[px]
            new_d = max(0.0, old_d + shift)
            optical_depth[px] = new_d

            old_dark = 1.0 if old_d > 2.2 else (1.0 - 1.0 / (1.0 + old_d * 1.5 + 0.5 * old_d * old_d))
            new_dark = 1.0 if new_d > 2.2 else (1.0 - 1.0 / (1.0 + new_d * 1.5 + 0.5 * new_d * new_d))

            tar = target_darkness[px]
            imp = importance[px]

            diff_old = tar - old_dark
            err_old = (diff_old * diff_old) * imp * (over_mult if old_dark > tar else 1.0)

            diff_new = tar - new_dark
            err_new = (diff_new * diff_new) * imp * (over_mult if new_dark > tar else 1.0)

            delta_err += (err_new - err_old)

    # Откатываем изменения optical_depth обратно
    for arr_idx in range(4):
        indices = rem1 if arr_idx == 0 else (rem2 if arr_idx == 1 else (add1 if arr_idx == 2 else add2))
        shift = line_alpha if arr_idx < 2 else -line_alpha
        for i in range(len(indices)):
            px = indices[i]
            optical_depth[px] = max(0.0, optical_depth[px] + shift)

    return delta_err


@njit(fastmath=True, cache=True)
def apply_hinge_to_depth(optical_depth: np.ndarray, rem1: np.ndarray, rem2: np.ndarray,
                         add1: np.ndarray, add2: np.ndarray, line_alpha: float):
    for i in range(len(rem1)):
        optical_depth[rem1[i]] = max(0.0, optical_depth[rem1[i]] - line_alpha)
    for i in range(len(rem2)):
        optical_depth[rem2[i]] = max(0.0, optical_depth[rem2[i]] - line_alpha)
    for i in range(len(add1)):
        optical_depth[add1[i]] += line_alpha
    for i in range(len(add2)):
        optical_depth[add2[i]] += line_alpha



# ----------------------------------------------------------------------
# СВЕРХБЫСТРЫЕ ВЕКТОРИЗОВАННЫЕ ВЕРСИИ ФУНКЦИЙ NUMPY (ДЛЯ СИСТЕМ БЕЗ NUMBA)
# ----------------------------------------------------------------------

def evaluate_line_omp_fast(residual: np.ndarray, target_darkness: np.ndarray, importance: np.ndarray, 
                           indices: np.ndarray, line_alpha: float, fearless_penalty: float) -> float:
    n = len(indices)
    if n == 0:
        return -1e12
    res = residual[indices]
    tar = target_darkness[indices]
    imp = importance[indices]
    
    pos = res > 0.03
    after = res - line_alpha
    good_m = pos & (after >= 0.0)
    clip_m = pos & (after < 0.0)
    neg_m = ~pos

    delta = np.zeros(n, dtype=np.float32)
    if np.any(good_m):
        delta[good_m] = (res[good_m]**2 - after[good_m]**2) * imp[good_m]
    if np.any(clip_m):
        good = (res[clip_m]**2) * imp[clip_m]
        bad = (after[clip_m]**2) * imp[clip_m] * (1.2 + fearless_penalty * 2.5)
        delta[clip_m] = good - bad
    if np.any(neg_m):
        over = line_alpha - res[neg_m]
        wg = np.where(tar[neg_m] < 0.12, 3.2, 1.6)
        delta[neg_m] = -(over**2) * imp[neg_m] * fearless_penalty * wg

    net_imp = float(np.sum(delta))
    bright_ratio = float(np.sum(tar < 0.08)) / n
    if bright_ratio > 0.55:
        net_imp -= (bright_ratio - 0.55) * n * (line_alpha * line_alpha) * fearless_penalty * 2.0

    return net_imp / (n ** 0.92)

def fast_evaluate_chromosome_1d_fast(chromo_arr: np.ndarray, line_indices_flat: np.ndarray,
                                     line_offsets: np.ndarray, line_lengths: np.ndarray,
                                     num_pins: int, flat_size: int, line_alpha: float,
                                     target_darkness: np.ndarray, importance: np.ndarray,
                                     fearless_penalty: float) -> float:
    n_lines = len(chromo_arr) - 1
    all_slices = []
    for i in range(n_lines):
        p1 = chromo_arr[i]
        p2 = chromo_arr[i + 1]
        if p1 == p2: continue
        if p1 > p2: p1, p2 = p2, p1
        pair_idx = p1 * num_pins + p2
        off = line_offsets[pair_idx]
        length = line_lengths[pair_idx]
        if length > 0:
            all_slices.append(line_indices_flat[off : off + length])

    if all_slices:
        cat = np.concatenate(all_slices)
        depth = np.bincount(cat, minlength=flat_size).astype(np.float32) * line_alpha
    else:
        depth = np.zeros(flat_size, dtype=np.float32)

    cur_dark = np.where(depth > 2.2, 1.0, 1.0 - 1.0 / (1.0 + depth * 1.5 + 0.5 * depth * depth))
    diff = target_darkness - cur_dark
    over_mult = 1.0 + fearless_penalty * 1.8
    err = np.where(cur_dark > target_darkness, (diff * diff) * importance * over_mult, (diff * diff) * importance)
    return 1.0 - (float(np.sum(err)) / flat_size)

if not NUMBA_AVAILABLE:
    evaluate_line_omp = evaluate_line_omp_fast
    fast_evaluate_chromosome_1d = fast_evaluate_chromosome_1d_fast

# ----------------------------------------------------------------------
# РАЗДЕЛ 2: ПРЕДОБРАБОТКА ИЗОБРАЖЕНИЯ, РАДОН FBP И АВТО-ПОДБОР
# ----------------------------------------------------------------------

def prepare_target_image(original_image: Optional[Image.Image], target_size: int, 
                         clahe_clip: float = 1.3, gamma: float = 0.80, shape: str = 'Круг') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Готовит целевое изображение, очищает светлые зоны от серого шума (White Clipping)
    и строит карты контуров Собеля, визуальной значимости (Saliency + DoG) и весов важности.
    """
    if original_image is None:
        empty_img = np.full((target_size, target_size), 255, dtype=np.uint8)
        empty_maps = np.zeros((target_size, target_size), dtype=np.uint8)
        empty_imp = np.ones((target_size, target_size), dtype=np.float32)
        return empty_img, empty_maps, empty_maps, empty_imp

    img_gray = np.array(original_image.convert('L'))

    background = np.full((target_size, target_size), 255, dtype=np.uint8)
    h, w = img_gray.shape
    scale = min(target_size / max(1, w), target_size / max(1, h))
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    background[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    # Мягкий билатеральный фильтр сохраняет резкость зрачков и губ, но убирает шум кожи
    background = cv2.bilateralFilter(background, d=5, sigmaColor=22, sigmaSpace=22)

    if clahe_clip > 0.1:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        background = clahe.apply(background)

    # Очистка светлых полутонов (84-й перцентиль становится чистым белым), чтобы не плести кашу на светлом фоне
    p_low, p_high = np.percentile(background, (4, 84))
    if p_high - p_low > 20:
        norm_float = (background.astype(np.float32) - p_low) / (p_high - p_low)
        background = np.clip(norm_float * 255.0, 0, 255).astype(np.uint8)

    if abs(gamma - 1.0) > 0.03:
        inv_gamma = 1.0 / max(0.1, gamma)
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        background = cv2.LUT(background, table)

    mask = np.full((target_size, target_size), 255, dtype=np.uint8)
    if shape == 'Круг':
        mask.fill(0)
        radius = int(target_size / 2 - (target_size / 40))
        cv2.circle(mask, (target_size // 2, target_size // 2), radius, 255, -1)
        background = np.where(mask == 255, background, 255)
    elif shape == 'Овал':
        mask.fill(0)
        rx = int(target_size / 2 - (target_size / 40))
        ry = int(rx * 0.78)
        cv2.ellipse(mask, (target_size // 2, target_size // 2), (rx, ry), 0, 0, 360, 255, -1)
        background = np.where(mask == 255, background, 255)
    elif shape == 'Шестиугольник':
        mask.fill(0)
        r = target_size / 2 - (target_size / 40)
        pts = [[int(target_size/2 + r*math.cos(k*math.pi/3 - math.pi/2)), int(target_size/2 + r*math.sin(k*math.pi/3 - math.pi/2))] for k in range(6)]
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
        background = np.where(mask == 255, background, 255)

    blurred = cv2.GaussianBlur(background, (3, 3), 0)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    edges = (grad_mag / np.max(grad_mag) * 255).astype(np.uint8) if np.max(grad_mag) > 0 else np.zeros_like(background, dtype=np.uint8)

    saliency_map = edges.astype(np.float32) / 255.0
    try:
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        _, sal_raw = saliency.computeSaliency(cv2.cvtColor(background, cv2.COLOR_GRAY2BGR))
        saliency_map = 0.5 * sal_raw.astype(np.float32) + 0.5 * saliency_map
    except (cv2.error, AttributeError):
        pass

    g1 = cv2.GaussianBlur(background.astype(np.float32), (3, 3), 0)
    g2 = cv2.GaussianBlur(background.astype(np.float32), (13, 13), 0)
    dog = np.abs(g1 - g2)
    if dog.max() > 0:
        dog /= dog.max()

    combined_saliency = np.clip(0.35 * saliency_map + 0.65 * dog, 0, 1)
    if shape == 'Круг':
        combined_saliency = np.where(mask == 255, combined_saliency, 0)
    
    saliency_uint8 = (combined_saliency * 255.0).astype(np.uint8)
    importance = (combined_saliency * 3.5 + 0.6).astype(np.float32)

    return background, edges, saliency_uint8, importance


def auto_recommend_parameters(original_image: Image.Image, shape: str = 'Круг') -> Dict[str, Any]:
    """
    Интеллектуальный расчет параметров под изображение для ЧЕРНОЙ нити (950-1850 линий без каши).
    """
    img_gray = np.array(original_image.convert('L'))
    h, w = img_gray.shape
    scale = min(400 / max(1, w), 400 / max(1, h))
    resized = cv2.resize(img_gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    p5, p95 = np.percentile(resized, (5, 95))
    contrast_range = float(p95 - p5)
    mean_lum = float(np.mean(resized))

    clahe_clip = 1.4 if contrast_range < 135 else 1.0
    gamma = 0.72 if mean_lum < 115 else (0.90 if mean_lum > 175 else 0.80)

    temp_img, edges, _, _ = prepare_target_image(original_image, 400, clahe_clip, gamma, shape)

    edge_density = float(np.mean(edges > 45))
    lap_std = float(np.std(cv2.Laplacian(temp_img, cv2.CV_64F)))
    complexity_score = float(np.clip((edge_density - 0.04) / 0.18 * 0.65 + (lap_std - 8.0) / 38.0 * 0.35, 0.0, 1.0))
    darkness_ratio = float(np.mean(np.maximum(0.0, (215.0 - temp_img) / 215.0)))

    raw_pins = 240 + complexity_score * 64
    recommended_pins = int(np.clip(round(raw_pins / 4.0) * 4, 220, 308))

    raw_lines = 950 + darkness_ratio * 650 + complexity_score * 250
    recommended_lines = int(np.clip(round(raw_lines / 50.0) * 50, 950, 1850))

    line_weight = int(np.clip(round(75 - (recommended_lines - 1100) / 45.0), 62, 85))
    fearless_factor = int(np.clip(round(94 - complexity_score * 6), 86, 98))
    contour_weight = int(np.clip(round(50 + complexity_score * 20), 40, 75))
    radon_influence = int(np.clip(round(30 + complexity_score * 15), 20, 50))
    stages = 5 if recommended_lines <= 1500 else 6

    summary = (
        f"Детали: {int(complexity_score * 100)}% | Тень: {int(darkness_ratio * 100)}% → "
        f"{recommended_pins} гвоздей, {recommended_lines} черных нитей"
    )

    return {
        'pins': recommended_pins,
        'lines': recommended_lines,
        'line_weight': line_weight,
        'clahe_clip': round(clahe_clip, 1),
        'gamma': round(gamma, 2),
        'fearless_factor': fearless_factor,
        'contour_weight': contour_weight,
        'radon_influence': radon_influence,
        'stages': stages,
        'summary': summary
    }


class RadonGuide:
    """
    Томографический гид на базе преобразования Радона с частотным фильтром Рам-Лака (Ramp Filter / FBP).
    Предвычисляет лучших соседей для каждого гвоздя.
    """
    def __init__(self, target_image_np: np.ndarray, edges_np: np.ndarray, canvas_manager: 'CanvasManager', config: Dict[str, Any]):
        self.config = config
        self.canvas_manager = canvas_manager
        self.sinogram: Optional[np.ndarray] = None
        self.line_weights: Dict[Tuple[int, int], float] = {}
        self.pin_adj_weights = np.zeros((canvas_manager.num_pins, canvas_manager.num_pins), dtype=np.float32)
        self.top_neighbors_cache: List[np.ndarray] = []

        if self.config.get('radon_enabled', False) and radon is not None:
            self._compute_filtered_sinogram(target_image_np, edges_np)

    def _apply_ramp_filter(self, sinogram: np.ndarray) -> np.ndarray:
        """Применяет фильтр Рам-Лака с окном Хэмминга в частотной области (устраняет 1/r размытие)."""
        n_rho, n_theta = sinogram.shape
        padded_len = int(2 ** np.ceil(np.log2(n_rho * 2)))
        freqs = np.fft.rfftfreq(padded_len).reshape(-1, 1)
        ramp = np.abs(freqs) * (0.54 + 0.46 * np.cos(np.pi * freqs / (freqs.max() + 1e-8)))
        sino_padded = np.pad(sinogram, ((0, padded_len - n_rho), (0, 0)), mode='constant')
        sino_fft = np.fft.rfft(sino_padded, axis=0)
        filtered = np.fft.irfft(sino_fft * ramp, n=padded_len, axis=0)[:n_rho, :]
        return np.maximum(0.0, filtered)

    def _compute_filtered_sinogram(self, target_image: np.ndarray, edges_np: np.ndarray):
        darkness = (255.0 - target_image.astype(np.float32)) / 255.0
        edge_norm = edges_np.astype(np.float32) / 255.0
        hybrid_density = 0.65 * darkness + 0.35 * edge_norm

        # Вычисляем Радона на уменьшенной копии 250x250 для высокой скорости старта
        small_size = 250
        small_density = cv2.resize(hybrid_density, (small_size, small_size), interpolation=cv2.INTER_AREA)
        theta = np.linspace(0., 180., 180, endpoint=False)
        raw_sino = radon(small_density, theta=theta, circle=True)
        filtered_sino = self._apply_ramp_filter(raw_sino)

        if filtered_sino.max() > filtered_sino.min():
            self.sinogram = (filtered_sino - filtered_sino.min()) / (filtered_sino.max() - filtered_sino.min())
        else:
            self.sinogram = filtered_sino

        center = self.canvas_manager.size / 2.0
        scale_rho = self.sinogram.shape[0] / float(self.canvas_manager.size)
        num_pins = self.canvas_manager.num_pins
        coords = self.canvas_manager.pin_coords

        for i in range(num_pins):
            for j in range(i + 1, num_pins):
                dist = min(j - i, num_pins - (j - i))
                if dist < MIN_PIN_DISTANCE:
                    continue
                p1, p2 = coords[i], coords[j]
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                line_angle = math.degrees(math.atan2(dy, dx)) % 180.0
                proj_theta = (line_angle + 90.0) % 180.0
                mx, my = (p1[0] + p2[0]) * 0.5 - center, (p1[1] + p2[1]) * 0.5 - center
                rad = math.radians(proj_theta)
                rho = (mx * math.cos(rad) + my * math.sin(rad)) * scale_rho
                rho_idx = int(round(rho + self.sinogram.shape[0] * 0.5))
                theta_idx = int(round(proj_theta)) % 180

                if 0 <= rho_idx < self.sinogram.shape[0]:
                    w = float(self.sinogram[rho_idx, theta_idx])
                    self.line_weights[(i, j)] = w
                    self.pin_adj_weights[i, j] = w
                    self.pin_adj_weights[j, i] = w

        # Предвычисляем топ-36 лучших соседей для каждого гвоздя
        for i in range(num_pins):
            order = np.argsort(self.pin_adj_weights[i])[::-1]
            valid = [p for p in order if min(abs(p - i), num_pins - abs(p - i)) >= MIN_PIN_DISTANCE]
            self.top_neighbors_cache.append(np.array(valid[:36], dtype=np.int32))


# ----------------------------------------------------------------------
# РАЗДЕЛ 3: УПРАВЛЕНИЕ ХОЛСТОМ (4x SSAA SUBPIXEL AA) И ФИТНЕС-КАЛЬКУЛЯТОР
# ----------------------------------------------------------------------

class CanvasManager:
    def __init__(self, size: int, num_pins: int, shape: str = 'Круг'):
        self.size = size
        self.num_pins = num_pins
        self.shape = shape
        self.pin_coords = self._calculate_pin_coords()
        self.pixel_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self.flat_ready = False
        self.line_indices_flat = np.empty(0, dtype=np.int64)
        self.line_offsets = np.zeros(num_pins * num_pins, dtype=np.int64)
        self.line_lengths = np.zeros(num_pins * num_pins, dtype=np.int64)

    def _calculate_pin_coords(self) -> np.ndarray:
        if self.shape == 'Квадрат':
            coords = []
            margin = self.size / 40.0
            side = self.size - 2 * margin
            perim = side * 4.0
            step = perim / max(1, self.num_pins)
            for i in range(self.num_pins):
                d = i * step
                if d <= side:
                    x, y = margin + d, margin
                elif d <= 2 * side:
                    x, y = self.size - margin, margin + (d - side)
                elif d <= 3 * side:
                    x, y = (self.size - margin) - (d - 2 * side), self.size - margin
                else:
                    x, y = margin, (self.size - margin) - (d - 3 * side)
                coords.append((x, y))
            return np.array(coords, dtype=np.float32)
        elif 'Прямоугольник' in self.shape:
            aspect = 4.0 / 3.0 if '4:3' in self.shape else (16.0 / 9.0 if '16:9' in self.shape else 1.33)
            margin = self.size / 40.0
            w = self.size - 2 * margin
            h = w / aspect
            start_y = (self.size - h) / 2.0
            perim = 2.0 * (w + h)
            step = perim / max(1, self.num_pins)
            coords = []
            for i in range(self.num_pins):
                d = (i * step) % perim
                if d <= w:
                    x, y = margin + d, start_y
                elif d <= w + h:
                    x, y = margin + w, start_y + (d - w)
                elif d <= 2 * w + h:
                    x, y = margin + w - (d - w - h), start_y + h
                else:
                    x, y = margin, start_y + h - (d - 2 * w - h)
                coords.append((x, y))
            return np.array(coords, dtype=np.float32)
        elif self.shape == 'Овал':
            coords = []
            rx = self.size / 2.0 - (self.size / 40.0)
            ry = rx * 0.78
            center = self.size / 2.0
            for i in range(self.num_pins):
                angle = (i / self.num_pins) * 2.0 * math.pi
                coords.append((center + rx * math.cos(angle), center + ry * math.sin(angle)))
            return np.array(coords, dtype=np.float32)
        elif self.shape == 'Шестиугольник':
            r = self.size / 2.0 - (self.size / 40.0)
            center = self.size / 2.0
            vertices = []
            for k in range(6):
                ang = (k / 6.0) * 2.0 * math.pi - math.pi / 2.0
                vertices.append((center + r * math.cos(ang), center + r * math.sin(ang)))
            vertices.append(vertices[0])
            coords = []
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
            return np.array(coords[:self.num_pins], dtype=np.float32)
        else:
            coords = []
            radius = self.size / 2.0 - (self.size / 40.0)
            center = self.size / 2.0
            for i in range(self.num_pins):
                angle = (i / self.num_pins) * 2.0 * math.pi
                coords.append((center + radius * math.cos(angle), center + radius * math.sin(angle)))
            return np.array(coords, dtype=np.float32)

    def get_line_pixel_indices(self, p1_idx: int, p2_idx: int) -> np.ndarray:
        key = (p1_idx, p2_idx) if p1_idx < p2_idx else (p2_idx, p1_idx)
        if key in self.pixel_cache:
            return self.pixel_cache[key]

        p1 = self.pin_coords[p1_idx]
        p2 = self.pin_coords[p2_idx]
        length = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if length == 0:
            res = np.empty(0, dtype=np.int64)
            self.pixel_cache[key] = res
            return res

        x = np.linspace(p1[0], p2[0], length)
        y = np.linspace(p1[1], p2[1], length)
        rr = np.round(y).astype(np.int64)
        cc = np.round(x).astype(np.int64)
        valid = (rr >= 0) & (rr < self.size) & (cc >= 0) & (cc < self.size)
        indices = np.unique((rr[valid] * self.size + cc[valid]).astype(np.int64))
        self.pixel_cache[key] = indices
        return indices

    def build_flat_line_table(self):
        """Упаковывает все хорды в единый 1D-массив для мгновенной работы Numba в потоках островов."""
        if self.flat_ready:
            return
        all_chunks = []
        current_offset = 0
        for i in range(self.num_pins):
            for j in range(i + 1, self.num_pins):
                pts = self.get_line_pixel_indices(i, j)
                pair_idx = i * self.num_pins + j
                self.line_offsets[pair_idx] = current_offset
                self.line_lengths[pair_idx] = len(pts)
                if len(pts) > 0:
                    all_chunks.append(pts)
                    current_offset += len(pts)
        self.line_indices_flat = np.concatenate(all_chunks) if all_chunks else np.empty(0, dtype=np.int64)
        self.flat_ready = True

    def get_line_alpha(self, num_lines: int, line_weight_percent: int = 72) -> float:
        alpha = (line_weight_percent / 100.0) * 0.42
        return float(np.clip(alpha, 0.12, 0.55))

    def compute_optical_depth_1d(self, chromosome: List[int], line_alpha: float) -> np.ndarray:
        depth = np.zeros(self.size * self.size, dtype=np.float32)
        for i in range(len(chromosome) - 1):
            pts = self.get_line_pixel_indices(chromosome[i], chromosome[i + 1])
            if len(pts) > 0:
                depth[pts] += line_alpha
        return depth

    def render_chromosome_to_numpy(self, chromosome: List[int], thickness: int = 1, 
                                   line_weight_percent: int = 72, max_step: Optional[int] = None,
                                   output_size: Optional[int] = None) -> np.ndarray:
        """
        Кристально четкий рендеринг угольно-черной нити с 4x Суперсэмплингом (SSAA) 
        и субпиксельным сглаживанием (cv2.LINE_AA + shift=4).
        """
        target_out = output_size if output_size is not None else self.size
        if not chromosome or len(chromosome) < 2:
            return np.full((target_out, target_out), 255, dtype=np.uint8)

        total_lines = len(chromosome) - 1
        lines_to_draw = total_lines if max_step is None else min(total_lines, max_step)

        hi_res = max(1600, target_out * 2)
        scale_to_hi = hi_res / float(self.size)

        SHIFT = 4
        MULT = 1 << SHIFT
        coords_subpix = np.round(self.pin_coords * scale_to_hi * MULT).astype(np.int32)

        stroke_darkness = float(np.clip((line_weight_percent / 100.0) * 1.15, 0.45, 0.96))
        optical_depth = np.zeros((hi_res, hi_res), dtype=np.float32)

        for i in range(lines_to_draw):
            idx1 = chromosome[i]
            idx2 = chromosome[i + 1]
            p1 = (int(coords_subpix[idx1, 0]), int(coords_subpix[idx1, 1]))
            p2 = (int(coords_subpix[idx2, 0]), int(coords_subpix[idx2, 1]))
            cv2.line(optical_depth, p1, p2, color=stroke_darkness, thickness=thickness, lineType=cv2.LINE_AA, shift=SHIFT)

        transmittance = np.exp(-optical_depth * 2.2)
        if hi_res != target_out:
            transmittance = cv2.resize(transmittance, (target_out, target_out), interpolation=cv2.INTER_AREA)

        transmittance = np.clip(transmittance, 0.0, 1.0) ** 1.25
        result_uint8 = (transmittance * 255.0).astype(np.uint8)

        # Аккуратные точки гвоздей по периметру
        pin_scale = target_out / float(self.size)
        pin_r = max(1, int(target_out / 340))
        for px, py in self.pin_coords:
            cx, cy = int(round(px * pin_scale)), int(round(py * pin_scale))
            cv2.circle(result_uint8, (cx, cy), pin_r, 55, -1, lineType=cv2.LINE_AA)

        return result_uint8

    def render_chromosome_to_pil(self, chromosome: List[int], thickness: int = 1, 
                                 line_weight_percent: int = 72, max_step: Optional[int] = None,
                                 output_size: Optional[int] = None) -> Image.Image:
        return Image.fromarray(
            self.render_chromosome_to_numpy(chromosome, thickness, line_weight_percent, max_step, output_size)
        )


class FitnessCalculator:
    """
    Быстрый вычислитель приспособленности (Fitness) и аналитических карт расхождений/плотности.
    """
    def __init__(self, target_2d: np.ndarray, target_edges_2d: np.ndarray,
                 saliency_uint8: np.ndarray, importance_2d: np.ndarray,
                 canvas_manager: CanvasManager, config: Dict[str, Any],
                 queues: Dict[str, queue.Queue]):
        self.canvas_manager = canvas_manager
        self.canvas_manager.build_flat_line_table()
        self.config = config
        self.queues = queues

        self.target_2d = target_2d
        self.target_darkness_flat = ((255.0 - target_2d.astype(np.float32)) / 255.0).flatten()
        
        contour_boost = (config.get('contour_weight', 55) / 100.0) * 2.5
        edge_norm = (target_edges_2d.astype(np.float32) / 255.0).flatten()
        self.importance_flat = (importance_2d.flatten() + edge_norm * contour_boost).astype(np.float32)

        self.line_alpha = self.canvas_manager.get_line_alpha(config['lines'], config.get('line_weight', 72))
        self.fearless_penalty = config.get('fearless_factor', 94) / 100.0

        self.distance_transform_map = None
        if config.get('structural_metric') == 'Преобразование расстояний' and SCIPY_AVAILABLE:
            inv_edges = 255 - target_edges_2d
            dt = distance_transform_edt(inv_edges)
            if dt.max() > 0:
                dt = dt / dt.max()
            self.distance_transform_map = dt

        self.queues['status_queue'].put({'type': 'gpu_status', 'data': GPU_ENABLED_FLAG})

    def calculate_single_fitness(self, chromosome: List[int]) -> float:
        if not chromosome or len(chromosome) < 2:
            return 0.0
        arr = np.asarray(chromosome, dtype=np.int64)
        return float(fast_evaluate_chromosome_1d(
            arr,
            self.canvas_manager.line_indices_flat,
            self.canvas_manager.line_offsets,
            self.canvas_manager.line_lengths,
            self.canvas_manager.num_pins,
            self.canvas_manager.size * self.canvas_manager.size,
            self.line_alpha,
            self.target_darkness_flat,
            self.importance_flat,
            self.fearless_penalty
        ))

    def calculate_fitness_scores(self, chromosomes: List[List[int]]) -> List[float]:
        return [self.calculate_single_fitness(c) for c in chromosomes]

    def calculate_single_fitness_and_maps(self, chromosome: List[int]) -> Tuple[float, np.ndarray, np.ndarray]:
        """Возвращает фитнес, 2D-карту ошибок и 2D-тепловую карту плотности нитей."""
        fitness = self.calculate_single_fitness(chromosome)
        depth_1d = self.canvas_manager.compute_optical_depth_1d(chromosome, self.line_alpha)
        depth_2d = depth_1d.reshape((self.canvas_manager.size, self.canvas_manager.size))
        cur_dark_2d = 1.0 - np.exp(-depth_2d * 1.6)
        tar_dark_2d = (255.0 - self.target_2d.astype(np.float32)) / 255.0
        # Положительное = перетемнили (красный), Отрицательное = недотемнили (синий)
        diff_map = cur_dark_2d - tar_dark_2d
        return fitness, diff_map, depth_2d


# ----------------------------------------------------------------------
# РАЗДЕЛ 4: ТОЧНЫЙ OMP-НАБРОСОК И ОСТРОВНОЙ ГЕНЕТИЧЕСКИЙ АЛГОРИТМ
# ----------------------------------------------------------------------

class GreedyInitializer:
    def __init__(self, canvas_manager: CanvasManager, target_darkness_flat: np.ndarray,
                 importance_flat: np.ndarray, config: Dict[str, Any],
                 radon_guide: Optional[RadonGuide] = None,
                 stop_Check: Optional[Callable[[], bool]] = None):
        self.canvas_manager = canvas_manager
        self.target_darkness_flat = target_darkness_flat
        self.importance_flat = importance_flat
        self.config = config
        self.radon_guide = radon_guide
        self.stop_check = stop_Check
        self.num_pins = canvas_manager.num_pins

    def run(self, num_lines: int, update_callback: Callable[[str, List[int]], None]) -> List[int]:
        if self.num_pins == 0:
            return []

        line_alpha = self.canvas_manager.get_line_alpha(num_lines, self.config.get('line_weight', 72))
        fearless = self.config.get('fearless_factor', 94) / 100.0
        radon_inf = (self.config.get('radon_influence', 35) / 100.0) * 0.22

        start_pin = 0
        if self.radon_guide and self.radon_guide.pin_adj_weights is not None:
            start_pin = int(np.argmax(np.sum(self.radon_guide.pin_adj_weights, axis=1)))

        path = [start_pin]
        current_pin = start_pin
        residual = self.target_darkness_flat.copy()
        used_edges = set()
        recent_pins = []

        for i in range(num_lines):
            if self.stop_check and self.stop_check():
                break

            best_score = -1e12
            best_next = -1

            for next_pin in range(self.num_pins):
                dist = min(abs(next_pin - current_pin), self.num_pins - abs(next_pin - current_pin))
                if dist < MIN_PIN_DISTANCE:
                    continue
                if next_pin in recent_pins:
                    continue
                edge_key = (current_pin, next_pin) if current_pin < next_pin else (next_pin, current_pin)
                if edge_key in used_edges:
                    continue

                line_pixels = self.canvas_manager.get_line_pixel_indices(current_pin, next_pin)
                if len(line_pixels) == 0:
                    continue

                score = evaluate_line_omp(residual, self.target_darkness_flat, self.importance_flat, line_pixels, line_alpha, fearless)
                if score > 0 and self.radon_guide is not None:
                    w_radon = self.radon_guide.pin_adj_weights[current_pin, next_pin]
                    score *= (1.0 + radon_inf * w_radon)

                if score > best_score:
                    best_score = score
                    best_next = next_pin

            if best_next == -1 or (i > 350 and best_score <= 0.0001):
                logging.info(f"OMP Авто-стоп на шаге {i}: изображение сформировано без перетемнения.")
                break

            best_pixels = self.canvas_manager.get_line_pixel_indices(current_pin, best_next)
            if len(best_pixels) > 0:
                update_residual_omp(residual, best_pixels, line_alpha)

            used_edges.add((current_pin, best_next) if current_pin < best_next else (best_next, current_pin))
            path.append(best_next)

            recent_pins.append(current_pin)
            if len(recent_pins) > 6:
                recent_pins.pop(0)
            current_pin = best_next

            if (i % 25 == 0 or i < 30) and i > 0:
                prog = int(((i + 1) / num_lines) * 100)
                update_callback(f"Этап 1: OMP-синтез черной нитью... {prog}% ({len(path)-1} нитей)", list(path))

        update_callback(f"OMP-синтез завершен ({len(path)-1} нитей)", list(path))
        return path


class IslandThread(threading.Thread):
    """
    Поток острова эволюционной доводки: графовые мутации (Smart Hinge, Radon Bridge, 2-Opt)
    и сверхбыстрый локальный дельта-поиск.
    """
    def __init__(self, island_id: int, config: Dict[str, Any],
                 fitness_calculator: FitnessCalculator,
                 queues: Dict[str, queue.Queue], base_chromosome: List[int],
                 radon_guide: Optional[RadonGuide] = None):
        super().__init__(daemon=True)
        self.island_id = island_id
        self.config = config
        self.fitness_calculator = fitness_calculator
        self.queues = queues
        self.base_chromosome = list(base_chromosome)
        self.radon_guide = radon_guide
        self.status_queue = queues['status_queue']
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.population: List[Dict[str, Any]] = []
        self.best_on_island: Optional[Dict[str, Any]] = None
        self.migrant_queue: queue.Queue = queue.Queue()

        self.mutation_operators = {
            "Smart Hinge": self._smart_hinge_mutation,
            "Radon Bridge": self._radon_bridge_mutation,
            "2-Opt Invert": self._two_opt_mutation,
            "Jitter Pin": self._jitter_pin_mutation,
        }
        self.mutation_stats = {name: {"success": 1, "attempts": 1} for name in self.mutation_operators.keys()}

    def stop(self): self.stop_event.set()
    def pause(self): self.pause_event.clear()
    def resume(self): self.pause_event.set()
    def get_best_individual(self) -> Optional[Dict[str, Any]]: return self.best_on_island
    def accept_migrant(self, migrant: Dict[str, Any]): self.migrant_queue.put(migrant)

    def _smart_hinge_mutation(self, c: List[int], s: int, e: int) -> List[int]:
        if e - s < 3:
            return c
        idx = random.randint(max(1, s), min(len(c) - 2, e - 1))
        prev_p, next_p = c[idx - 1], c[idx + 1]
        num_pins = self.fitness_calculator.canvas_manager.num_pins
        for _ in range(10):
            cand = random.randint(0, num_pins - 1)
            d1 = min(abs(cand - prev_p), num_pins - abs(cand - prev_p))
            d2 = min(abs(cand - next_p), num_pins - abs(cand - next_p))
            if d1 >= MIN_PIN_DISTANCE and d2 >= MIN_PIN_DISTANCE:
                c[idx] = cand
                break
        return c

    def _radon_bridge_mutation(self, c: List[int], s: int, e: int) -> List[int]:
        if e - s < 3 or self.radon_guide is None or not self.radon_guide.top_neighbors_cache:
            return self._smart_hinge_mutation(c, s, e)
        idx = random.randint(max(1, s), min(len(c) - 2, e - 1))
        prev_p, next_p = c[idx - 1], c[idx + 1]
        candidates = self.radon_guide.top_neighbors_cache[prev_p]
        if len(candidates) > 0:
            # Выбираем узел, сильный одновременно и с prev_p, и с next_p
            best_c = int(candidates[0])
            best_w = -1.0
            for cand in candidates[:12]:
                w = float(self.radon_guide.pin_adj_weights[prev_p, cand] + self.radon_guide.pin_adj_weights[cand, next_p])
                if w > best_w:
                    best_w = w
                    best_c = int(cand)
            c[idx] = best_c
        return c

    def _two_opt_mutation(self, c: List[int], s: int, e: int) -> List[int]:
        if e - s < 4:
            return c
        i1 = random.randint(s, e - 3)
        i2 = random.randint(i1 + 2, min(e, i1 + 25))
        c[i1:i2] = reversed(c[i1:i2])
        return c

    def _jitter_pin_mutation(self, c: List[int], s: int, e: int) -> List[int]:
        if e - s < 2:
            return c
        idx = random.randint(max(1, s), min(len(c) - 2, e - 1))
        num_pins = self.fitness_calculator.canvas_manager.num_pins
        shift = random.choice([-3, -2, -1, 1, 2, 3])
        cand = (c[idx] + shift) % num_pins
        c[idx] = cand
        return c

    def _graph_splice_crossover(self, p1: List[int], p2: List[int], start: int, end: int) -> List[int]:
        if len(p1) != len(p2) or end - start < 4:
            return list(p1)
        child = list(p1)
        cut = random.randint(start + 1, min(end - 1, len(p1) - 2))
        child[cut:end] = p2[cut:end]
        return child

    def _fast_delta_coordinate_descent(self, chromosome: List[int], start: int, end: int, depth: int) -> List[int]:
        """Сверхбыстрая локальная полировка пути за счет вычисления только дельты 2 смежных хорд."""
        if len(chromosome) < 4 or end - start < 3:
            return chromosome
        cm = self.fitness_calculator.canvas_manager
        num_pins = cm.num_pins
        line_alpha = self.fitness_calculator.line_alpha
        fearless = self.fitness_calculator.fearless_penalty
        target_dark = self.fitness_calculator.target_darkness_flat
        importance = self.fitness_calculator.importance_flat

        best_c = list(chromosome)
        optical_depth = cm.compute_optical_depth_1d(best_c, line_alpha)

        low_idx = max(1, start)
        high_idx = min(len(best_c) - 2, end - 1)
        if low_idx >= high_idx:
            return best_c

        for _ in range(depth):
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
            d1 = min(abs(cand_p - prev_p), num_pins - abs(cand_p - prev_p))
            d2 = min(abs(cand_p - next_p), num_pins - abs(cand_p - next_p))
            if d1 < MIN_PIN_DISTANCE or d2 < MIN_PIN_DISTANCE:
                continue

            rem1 = cm.get_line_pixel_indices(prev_p, cur_p)
            rem2 = cm.get_line_pixel_indices(cur_p, next_p)
            add1 = cm.get_line_pixel_indices(prev_p, cand_p)
            add2 = cm.get_line_pixel_indices(cand_p, next_p)

            delta_err = compute_delta_hinge_error(
                optical_depth, target_dark, importance, rem1, rem2, add1, add2, line_alpha, fearless
            )
            if delta_err < -1e-7:
                apply_hinge_to_depth(optical_depth, rem1, rem2, add1, add2, line_alpha)
                best_c[idx] = cand_p

        return best_c

    def run(self):
        try:
            init_fit = self.fitness_calculator.calculate_single_fitness(self.base_chromosome)
            self.best_on_island = {'chromosome': list(self.base_chromosome), 'fitness': init_fit}
            pop_size = max(4, self.config.get('population_size', 24))
            self.population = [{'chromosome': list(self.base_chromosome), 'fitness': init_fit} for _ in range(pop_size)]

            # Разнообразим стартовую популяцию легкими мутациями для островов > 0
            if self.island_id > 0:
                for k in range(1, pop_size):
                    c_copy = self._smart_hinge_mutation(list(self.base_chromosome), 1, len(self.base_chromosome) - 1)
                    self.population[k] = {'chromosome': c_copy, 'fitness': self.fitness_calculator.calculate_single_fitness(c_copy)}
                self.population.sort(key=lambda x: x['fitness'], reverse=True)

            self._send_island_update()

            actual_lines = max(2, len(self.base_chromosome) - 1)
            stages = max(1, self.config.get('stages', 5))
            lines_per_stage = max(1, actual_lines // stages)

            for stage in range(stages):
                if self.stop_event.is_set():
                    break
                start_line = stage * lines_per_stage
                end_line = actual_lines if stage == stages - 1 else min((stage + 1) * lines_per_stage, actual_lines)
                patience = 0
                patience_limit = PATIENCE_BASE + stage * PATIENCE_STAGE_FACTOR
                last_best = self.best_on_island['fitness']

                while patience < patience_limit:
                    self.pause_event.wait()
                    if self.stop_event.is_set():
                        break

                    self._send_generation_tick(stage, patience, patience_limit)
                    self._process_migrant()

                    stagnation = 1.0 + (patience / max(1, patience_limit)) * STAGNATION_MUTATION_FACTOR
                    mut_rate = min((self.config.get('mutation_rate', 18) / BASE_MUTATION_DIVISOR) * stagnation, 0.85)

                    self.population = self._evolve_population(start_line, end_line, mut_rate)

                    if self.config.get('memetic_enabled', True):
                        num_mem = max(1, int(len(self.population) * (self.config.get('memetic_intensity', 15) / 100.0)))
                        depth = self.config.get('memetic_depth', 25)
                        for idx_p in range(min(num_mem, len(self.population))):
                            polished = self._fast_delta_coordinate_descent(
                                self.population[idx_p]['chromosome'], start_line, end_line, depth
                            )
                            new_f = self.fitness_calculator.calculate_single_fitness(polished)
                            if new_f > self.population[idx_p]['fitness']:
                                self.population[idx_p]['chromosome'] = polished
                                self.population[idx_p]['fitness'] = new_f
                        self.population.sort(key=lambda x: x['fitness'], reverse=True)

                    if self.population[0]['fitness'] > self.best_on_island['fitness'] + 1e-7:
                        self.best_on_island = {
                            'chromosome': list(self.population[0]['chromosome']),
                            'fitness': self.population[0]['fitness']
                        }
                        self._send_island_update()

                    if self.best_on_island['fitness'] > last_best + 1e-6:
                        last_best = self.best_on_island['fitness']
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
        pop_size = self.config.get('population_size', 24)
        new_pop = [dict(self.population[0])]

        while len(new_pop) < pop_size:
            p1 = max(random.sample(self.population, min(4, len(self.population))), key=lambda x: x['fitness'])
            p2 = max(random.sample(self.population, min(4, len(self.population))), key=lambda x: x['fitness'])
            child = self._graph_splice_crossover(p1['chromosome'], p2['chromosome'], start_line, end_line)

            mut_info = None
            if random.random() < mut_rate:
                m_name, m_func = self._choose_mutation_operator()
                child = m_func(child, start_line, end_line)
                mut_info = (m_name, p1['fitness'])

            f_val = self.fitness_calculator.calculate_single_fitness(child)
            if mut_info is not None:
                m_name, parent_f = mut_info
                self.mutation_stats[m_name]["attempts"] += 1
                if f_val > parent_f:
                    self.mutation_stats[m_name]["success"] += 1

            new_pop.append({'chromosome': child, 'fitness': f_val})

        new_pop.sort(key=lambda x: x['fitness'], reverse=True)
        return new_pop

    def _process_migrant(self):
        try:
            migrant = self.migrant_queue.get_nowait()
            if self.population and migrant['fitness'] > self.population[-1]['fitness']:
                self.population[-1] = {'chromosome': list(migrant['chromosome']), 'fitness': migrant['fitness']}
                self.population.sort(key=lambda x: x['fitness'], reverse=True)
        except queue.Empty:
            pass

    def _send_island_update(self):
        if self.best_on_island:
            self.status_queue.put({
                'type': 'island_update',
                'data': {
                    'id': self.island_id,
                    'chromosome': list(self.best_on_island['chromosome']),
                    'fitness': self.best_on_island['fitness'],
                    'mut_stats': {k: dict(v) for k, v in self.mutation_stats.items()}
                }
            })

    def _send_generation_tick(self, stage: int, patience: int, limit: int):
        self.status_queue.put({'type': 'generation_tick', 'data': (self.island_id, stage, patience, limit)})


class IslandManagerThread(threading.Thread):
    def __init__(self, config: Dict[str, Any], queues: Dict[str, queue.Queue],
                 fitness_calculator: FitnessCalculator, base_chromosome: List[int],
                 radon_guide: Optional[RadonGuide] = None):
        super().__init__(daemon=True)
        self.config = config
        self.queues = queues
        self.fitness_calculator = fitness_calculator
        self.base_chromosome = base_chromosome
        self.radon_guide = radon_guide
        self.islands: List[IslandThread] = []
        self.stop_event = threading.Event()
        self.status_queue = queues['status_queue']

    def stop(self):
        self.stop_event.set()
        for island in self.islands:
            island.stop()

    def pause(self):
        for island in self.islands:
            island.pause()

    def resume(self):
        for island in self.islands:
            island.resume()

    def run(self):
        try:
            num_islands = max(1, self.config.get('num_islands', 3))
            self.status_queue.put({'type': 'status', 'data': f"Этап 2: Эволюционная полировка на {num_islands} островах..."})
            self.islands = [
                IslandThread(i, self.config, self.fitness_calculator, self.queues, self.base_chromosome, self.radon_guide)
                for i in range(num_islands)
            ]
            for island in self.islands:
                island.start()

            tick_count = 0
            mig_interval = max(5, self.config.get('migration_interval', 20))
            while any(island.is_alive() for island in self.islands):
                if self.stop_event.is_set():
                    break
                time.sleep(0.15)
                tick_count += 1
                if tick_count % mig_interval == 0 and len(self.islands) > 1:
                    self.status_queue.put({'type': 'migration'})
                    migrants = [isl.get_best_individual() for isl in self.islands]
                    if all(m is not None for m in migrants):
                        for i, isl in enumerate(self.islands):
                            isl.accept_migrant(migrants[(i + 1) % len(self.islands)])

            for island in self.islands:
                island.join(timeout=1.0)
            msg = "Остановлено пользователем." if self.stop_event.is_set() else "Генерация и полировка успешно завершены!"
            self.status_queue.put({'type': 'done', 'data': msg})
        except Exception:
            logging.error("Ошибка в IslandManagerThread", exc_info=True)
            self.status_queue.put({'type': 'error', 'data': "Ошибка в диспетчере островов."})


# ----------------------------------------------------------------------
# РАЗДЕЛ 5: СОСТОЯНИЕ ПРИЛОЖЕНИЯ, ПОЛНЫЙ UI И КООРДИНАТОР
# ----------------------------------------------------------------------

class AppState:
    def __init__(self):
        self.vars: Dict[str, tk.Variable] = {
            'clahe_clip': tk.DoubleVar(value=1.3),
            'gamma': tk.DoubleVar(value=0.80),
            'canvas_shape': tk.StringVar(value='Круг'),
            'pins': tk.IntVar(value=260),
            'lines': tk.IntVar(value=1400),
            'line_weight': tk.IntVar(value=72),
            'thread_thickness': tk.IntVar(value=1),
            'fearless_factor': tk.IntVar(value=94),
            'num_islands': tk.IntVar(value=min(4, max(2, (os.cpu_count() or 4) // 2))),
            'population_size': tk.IntVar(value=24),
            'stages': tk.IntVar(value=5),
            'mutation_rate': tk.IntVar(value=18),
            'migration_interval': tk.IntVar(value=20),
            'structural_metric': tk.StringVar(value='Преобразование расстояний' if SCIPY_AVAILABLE else 'Контуры'),
            'contour_weight': tk.IntVar(value=55),
            'memetic_enabled': tk.BooleanVar(value=True),
            'memetic_intensity': tk.IntVar(value=20),
            'memetic_depth': tk.IntVar(value=25),
            'radon_enabled': tk.BooleanVar(value=True),
            'radon_influence': tk.IntVar(value=35),
            'step_scrubber': tk.IntVar(value=1400),
            'split_compare': tk.IntVar(value=0),
        }
        self.original_image: Optional[Image.Image] = None
        self.original_image_path: Optional[str] = None
        self.target_image_tk: Optional[Any] = None
        self.preprocessed_target_np: Optional[np.ndarray] = None
        self.overall_best_chromosome: Optional[List[int]] = None
        self.overall_best_fitness: float = -1.0
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

    def get_config(self) -> Dict[str, Any]:
        return {k: v.get() for k, v in self.vars.items()}

    def reset_for_new_run(self):
        self.overall_best_chromosome = None
        self.overall_best_fitness = -1.0
        self.omp_baseline_fitness = 0.0
        self.migration_count = 0
        self.is_paused = False
        self.start_time = time.time()
        num_islands = self.vars['num_islands'].get()
        self.chart_data_by_island = {i: [] for i in range(num_islands)}
        self.island_best_fitnesses = {i: -1.0 for i in range(num_islands)}
        self.generation_counters = {i: 0 for i in range(num_islands)}
        self.island_progress_data = {i: (0, 1) for i in range(num_islands)}
        self.aggregated_mut_stats = {}


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tipwindow: Optional[tk.Toplevel] = None
        self.widget.bind("<Enter>", self.showtip)
        self.widget.bind("<Leave>", self.hidetip)

    def showtip(self, event=None):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 22
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{int(x)}+{int(y)}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT, background="#ffffe0",
                         relief=tk.SOLID, borderwidth=1, font=("Segoe UI", 8, "normal"))
        label.pack(ipadx=5, ipady=2)

    def hidetip(self, event=None):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


class UIManager:
    def __init__(self, root: 'StringArtApp', state: AppState):
        self.root = root
        self.state = state
        self.coordinator: Optional['EvolutionCoordinator'] = None
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

    def create_widgets(self):

        style = ttk.Style(self.root)
        style.theme_use('clam')

        BG_MAIN = '#f6f8fa'
        BG_SIDE = '#ffffff'
        TEXT_MAIN = '#1f2328'
        TEXT_SUB = '#57606a'
        ACCENT = '#0969da'
        BORDER = '#d0d7de'

        self.root.configure(bg=BG_MAIN)
        style.configure(".", background=BG_MAIN, foreground=TEXT_MAIN, font=('Segoe UI', 9))
        style.configure("TFrame", background=BG_MAIN)
        style.configure("TLabel", background=BG_MAIN, foreground=TEXT_MAIN)
        style.configure("TLabelframe", background=BG_SIDE, bordercolor=BORDER)
        style.configure("TLabelframe.Label", background=BG_SIDE, foreground=TEXT_MAIN, font=('Segoe UI', 9, 'bold'))

        style.configure("TButton", background='#f6f8fa', foreground=TEXT_MAIN, borderwidth=1, relief='solid', padding=4)
        style.map("TButton", background=[("active", "#eaeef2"), ("pressed", "#d0d7de")], foreground=[("active", ACCENT)])
        
        style.configure("TNotebook", background=BG_MAIN, tabmargins=[2, 2, 2, 0])
        style.configure("TNotebook.Tab", background='#e1e4e8', foreground=TEXT_SUB, padding=[10, 5], font=('Segoe UI', 9))
        style.map("TNotebook.Tab", background=[("selected", "#ffffff")], foreground=[("selected", ACCENT)])

        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor='#e1e4e8', borderwidth=0)

        main_pane = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        top_area = ttk.PanedWindow(main_pane, orient=tk.HORIZONTAL)
        main_pane.add(top_area, weight=3)

        self._create_control_panel(top_area)
        self._create_display_area(top_area)

        progress_frame = ttk.LabelFrame(main_pane, text="Дашборд Эволюции и Аналитика Алгоритма", height=230)
        main_pane.add(progress_frame, weight=1)
        self._create_stats_and_chart_area(progress_frame)

    def _create_control_panel(self, parent: ttk.PanedWindow):
        container = ttk.Frame(parent, width=520)
        parent.add(container, weight=0)

        left_col = ttk.Frame(container)
        left_col.pack(side=tk.LEFT, fill=tk.Y, anchor='n', padx=(5, 2), expand=True)
        right_col = ttk.Frame(container)
        right_col.pack(side=tk.LEFT, fill=tk.Y, anchor='n', padx=(2, 5), expand=True)

        self._create_image_settings_section(left_col)
        self._create_geometry_section(left_col)
        self._create_algorithm_section(left_col)

        self._create_fitness_and_radon_section(right_col)
        self._create_inspector_section(right_col)
        self._create_control_and_export_section(right_col)

    def _create_display_area(self, parent: ttk.PanedWindow):
        display_area = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        parent.add(display_area, weight=1)

        target_frame = ttk.LabelFrame(display_area, text="Целевой шаблон (с живой предобработкой)", width=350)
        display_area.add(target_frame, weight=0)
        canvas = tk.Canvas(target_frame, bg='#ffffff', relief='sunken', borderwidth=1)
        canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        canvas.bind("<Configure>", self.root.redraw_target_canvas)
        self.widgets['target_canvas'] = canvas

        results_area = ttk.Frame(display_area)
        display_area.add(results_area, weight=1)
        results_notebook = ttk.Notebook(results_area)
        results_notebook.pack(fill=tk.BOTH, expand=True)
        results_notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.widgets['results_notebook'] = results_notebook

    def _on_tab_changed(self, event=None):
        if not self.coordinator:
            return
        notebook = self.widgets['results_notebook']
        sel = notebook.select()
        if not sel:
            return
        idx = notebook.index(sel)
        if idx == 0:
            self.on_result_canvas_resize(None, 'overall')
        elif 1 <= idx <= self.state.vars['num_islands'].get():
            self.on_result_canvas_resize(None, idx - 1)

    def _create_section(self, parent: tk.Widget, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=6)
        frame.pack(fill=tk.X, pady=3, padx=3, anchor='n')
        return frame

    def _create_image_settings_section(self, parent: tk.Widget):
        frame = self._create_section(parent, "1. Изображение и Авто-подбор")
        btn = ttk.Button(frame, text="📁 Загрузить изображение")
        btn.pack(fill=tk.X, pady=2)
        ToolTip(btn, TOOLTIPS['load_image'])
        self.widgets['load_image_button'] = btn

        lbl_info = ttk.Label(frame, text="Анализ: загрузите фото...", font=('Segoe UI', 8), foreground='#333333', wraplength=235)
        lbl_info.pack(fill=tk.X, pady=2)
        self.widgets['analysis_label'] = lbl_info

        btn_auto = ttk.Button(frame, text="🔮 Авто-подбор параметров (без каши)", command=self.root.apply_auto_recommendations)
        btn_auto.pack(fill=tk.X, pady=2)

        self._create_double_slider_row(frame, 'gamma', "Светлота (Гамма)", (0.50, 1.40), command=self.root.redraw_target_canvas)
        self._create_double_slider_row(frame, 'clahe_clip', "Контраст (CLAHE)", (0.0, 3.5), command=self.root.redraw_target_canvas)
        self.widgets['image_settings_frame'] = frame

    def _create_geometry_section(self, parent: tk.Widget):
        frame = self._create_section(parent, "2. Геометрия и Черная Нить")
        f_shape = ttk.Frame(frame)
        f_shape.pack(fill=tk.X, pady=2)
        ttk.Label(f_shape, text="Форма холста:", width=16).pack(side=tk.LEFT)
        shape_combo = ttk.Combobox(f_shape, textvariable=self.state.vars['canvas_shape'], values=['Круг', 'Квадрат', 'Прямоугольник (4:3)', 'Прямоугольник (16:9)', 'Овал', 'Шестиугольник'], state='readonly')
        shape_combo.pack(fill=tk.X, expand=True)
        shape_combo.bind("<<ComboboxSelected>>", self.root.redraw_target_canvas)
        ToolTip(shape_combo, TOOLTIPS['canvas_shape'])

        self._create_slider_row(frame, 'pins', "Гвозди", (120, 480))
        self._create_slider_row(frame, 'lines', "Макс. нитей", (400, 3500))
        self._create_slider_row(frame, 'line_weight', "Чернота нити (%)", (40, 95), command=self._on_visual_param_change)
        self._create_slider_row(frame, 'thread_thickness', "Толщина нити", (1, 3), command=self._on_visual_param_change)
        self._create_slider_row(frame, 'fearless_factor', "Защита белого (%)", (70, 100))
        self.widgets['geometry_frame'] = frame

    def _create_algorithm_section(self, parent: tk.Widget):
        frame = self._create_section(parent, "3. Островной ГА (Полировка)")
        sliders = [
            ('num_islands', "Острова (потоки)", (1, max(2, os.cpu_count() or 4))),
            ('population_size', "Популяция", (8, 80)),
            ('stages', "Этапы доводки", (2, 12)),
            ('mutation_rate', "Мутация (%)", (5, 70)),
            ('migration_interval', "Интервал миграции", (5, 50))
        ]
        for name, text, limits in sliders:
            self._create_slider_row(frame, name, text, limits)
        self.widgets['algo_frame'] = frame

    def _create_fitness_and_radon_section(self, parent: tk.Widget):
        frame = self._create_section(parent, "4. Радон FBP и Дельта-Поиск")
        f_struct = ttk.Frame(frame)
        f_struct.pack(fill=tk.X, pady=2)
        ttk.Label(f_struct, text="Метрика краев:", width=16).pack(side=tk.LEFT)
        struct_values = ['Контуры']
        if SCIPY_AVAILABLE:
            struct_values.insert(0, 'Преобразование расстояний')
        struct_combo = ttk.Combobox(f_struct, textvariable=self.state.vars['structural_metric'], values=struct_values, state='readonly')
        struct_combo.pack(fill=tk.X, expand=True)
        ToolTip(struct_combo, TOOLTIPS['structural_metric'])
        self._create_slider_row(frame, 'contour_weight', "Вес контуров (%)", (10, 90))

        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=4)
        check_mem = ttk.Checkbutton(frame, text="Включить быстрый дельта-поиск", variable=self.state.vars['memetic_enabled'])
        check_mem.pack(anchor='w')
        ToolTip(check_mem, TOOLTIPS['memetic_enabled'])
        self._create_slider_row(frame, 'memetic_intensity', "Интенсивность (%)", (5, 50))
        self._create_slider_row(frame, 'memetic_depth', "Глубина поиска", (5, 80))

        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=4)
        check_radon = ttk.Checkbutton(frame, text="Наведение Радона (Ramp FBP)", variable=self.state.vars['radon_enabled'])
        if not SKIMAGE_AVAILABLE:
            check_radon.config(state=tk.DISABLED)
        check_radon.pack(anchor='w')
        ToolTip(check_radon, TOOLTIPS['radon_enabled'])
        self._create_slider_row(frame, 'radon_influence', "Сила Радона (%)", (0, 100))
        self.widgets['fitness_frame'] = frame

    def _create_inspector_section(self, parent: tk.Widget):
        frame = self._create_section(parent, "5. Интерактивный Инспектор")
        self._create_slider_row(frame, 'step_scrubber', "Слой плетения", (10, 3500), command=self._on_visual_param_change)
        self._create_slider_row(frame, 'split_compare', "Шторка До/После %", (0, 100), command=self._on_visual_param_change)
        self.widgets['inspector_frame'] = frame

    def _create_control_and_export_section(self, parent: tk.Widget):
        frame = self._create_section(parent, "6. Управление и Экспорт")
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=3)
        buttons = [
            ('start', "▶ Старт (OMP + Острова)", self.coordinator.start_generation, tk.NORMAL),
            ('polish', "✨ Эволюционная доводка", self.coordinator.start_evolution_only, tk.NORMAL),
            ('pause', "❚❚ Пауза", self.coordinator.toggle_pause, tk.DISABLED),
            ('stop', "■ Стоп", self.coordinator.stop_generation, tk.DISABLED)
        ]
        for name, text, command, state in buttons:
            btn = ttk.Button(btn_frame, text=text, command=command, state=state)
            btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
            ToolTip(btn, TOOLTIPS.get(name, ''))
            self.widgets[f'{name}_button'] = btn

        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=4)
        export_buttons = [
            ('prune', "✂ Оптимизировать путь (убрать петли)", self.coordinator.prune_chromosome),
            ('export_txt', "📄 Экспорт схемы гвоздей (.txt)", self.coordinator.export_txt),
            ('export_png', "🖼 Экспорт Ultra HD 2400px (.png)", self.coordinator.export_png),
            ('export_svg', "📐 Экспорт вектора (.svg)", self.coordinator.export_svg),
            ('export_html', "🌐 Автономный веб-ассистент (.html)", self.coordinator.export_html)
        ]
        for name, text, command in export_buttons:
            btn = ttk.Button(frame, text=text, command=command, state=tk.DISABLED)
            btn.pack(fill=tk.X, pady=1)
            self.widgets[f'{name}_button'] = btn
            ToolTip(btn, TOOLTIPS.get(name, ""))

    def _create_stats_and_chart_area(self, parent: tk.Widget):
        bottom_panel = ttk.Frame(parent)
        bottom_panel.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left_panel = ttk.Frame(bottom_panel, width=280)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)

        self.widgets['status_label'] = ttk.Label(left_panel, text="Ожидание запуска...", font=('Segoe UI', 9, 'bold'), wraplength=260)
        self.widgets['status_label'].pack(anchor='w')
        self.widgets['fitness_label'] = ttk.Label(left_panel, text="Точность (Фитнес): --", foreground='#0055aa')
        self.widgets['fitness_label'].pack(anchor='w', pady=(0, 3))
        self.widgets['compute_mode_label'] = ttk.Label(left_panel, text=f"Ядро: {'Numba JIT + CUDA' if GPU_ENABLED_FLAG else ('Numba JIT (Быстрый CPU)' if NUMBA_AVAILABLE else 'CPU')}", font=('Segoe UI', 8))
        self.widgets['compute_mode_label'].pack(anchor='w', pady=(0, 4))

        for key, text in [('time', "Время:"), ('eta', "Осталось (ETA):"), ('gens', "Поколений:"), ('mig', "Миграций:"), ('len', "Длина нити:")]:
            self.widgets[key + '_label'] = self._create_stat_row(left_panel, text)

        self.widgets['main_progress_bars_frame'] = ttk.Frame(left_panel)
        self.widgets['main_progress_bars_frame'].pack(fill=tk.BOTH, expand=True, pady=3)

        fig = Figure(figsize=(7, 2.2), facecolor='#ffffff')
        self.ax_fitness = fig.add_subplot(121)
        self.ax_mut = fig.add_subplot(122)
        fig.tight_layout(pad=1.2)
        self.chart_canvas = FigureCanvasTkAgg(fig, master=bottom_panel)
        self.chart_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=4, pady=2)

    def _create_slider_row(self, parent: tk.Widget, name: str, text: str, limits: tuple, command: Optional[Callable] = None):
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=1)
        lbl = ttk.Label(f, text=f"{text}:", width=16)
        lbl.pack(side=tk.LEFT)
        entry = ttk.Entry(f, textvariable=self.state.vars[name], width=5)
        entry.pack(side=tk.RIGHT)
        var = self.state.vars[name]
        def cmd(v, n=name):
            var.set(int(float(v)))
            if command:
                command()
        scale = ttk.Scale(f, from_=limits[0], to=limits[1], variable=var, orient=tk.HORIZONTAL, command=cmd)
        scale.pack(fill=tk.X, expand=True, padx=4)
        tip = TOOLTIPS.get(name, "")
        ToolTip(lbl, tip); ToolTip(entry, tip); ToolTip(scale, tip)

    def _create_double_slider_row(self, parent: tk.Widget, name: str, text: str, limits: tuple, command: Optional[Callable] = None):
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=1)
        lbl = ttk.Label(f, text=f"{text}:", width=16)
        lbl.pack(side=tk.LEFT)
        entry = ttk.Entry(f, textvariable=self.state.vars[name], width=5)
        entry.pack(side=tk.RIGHT)
        var = self.state.vars[name]
        def cmd(v, n=name):
            var.set(round(float(v), 2))
            if command:
                command()
        scale = ttk.Scale(f, from_=limits[0], to=limits[1], variable=var, orient=tk.HORIZONTAL, command=cmd)
        scale.pack(fill=tk.X, expand=True, padx=4)
        tip = TOOLTIPS.get(name, "")
        ToolTip(lbl, tip); ToolTip(entry, tip); ToolTip(scale, tip)

    def _create_stat_row(self, parent: tk.Widget, text: str) -> ttk.Label:
        row_frame = ttk.Frame(parent)
        row_frame.pack(fill='x')
        ttk.Label(row_frame, text=text, font=('Segoe UI', 8, 'bold'), width=15, anchor='w').pack(side=tk.LEFT)
        value_label = ttk.Label(row_frame, text="--", font=('Segoe UI', 8), anchor='w')
        value_label.pack(side=tk.LEFT)
        return value_label

    def _on_visual_param_change(self, *args):
        if self.coordinator and self.state.overall_best_chromosome:
            config = self.state.get_config()
            render_data = (self.state.overall_best_chromosome, config['pins'], config['thread_thickness'])
            self.coordinator.request_canvas_update('overall', render_data, {'force': True})

    def on_result_canvas_resize(self, event, key):
        canvas = self.result_canvases.get(key)
        if not canvas:
            return
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 20 or h < 20:
            return
        if self.coordinator and key in self.coordinator.best_chromosomes_by_key:
            chromo = self.coordinator.best_chromosomes_by_key[key]
            config = self.state.get_config()
            self.coordinator.request_canvas_update(key, (chromo, config['pins'], config['thread_thickness']), {})
        elif key in self.raw_pil_images:
            self.update_canvas_image(key, self.raw_pil_images[key])

    def setup_for_new_run(self, num_islands: int):
        self.setup_notebook_tabs(num_islands)
        self.setup_progress_bars(num_islands)
        self.chart_needs_update = True
        self.update_chart()

    def setup_notebook_tabs(self, num_islands: int):
        notebook = self.widgets['results_notebook']
        for tab_id in notebook.tabs():
            notebook.forget(tab_id)
        self.result_canvases.clear()
        self.result_photo_refs.clear()

        self._create_notebook_tab('overall', "★ Лучший результат", notebook)
        for i in range(num_islands):
            self._create_notebook_tab(i, f"Остров {i + 1}", notebook)

        self._create_analysis_canvas_tab(notebook, 'diff_map', "Карта Ошибок (Кр/Син)")
        self._create_analysis_canvas_tab(notebook, 'density_heatmap', "Плотность нитей (Heatmap)")
        self._create_analysis_canvas_tab(notebook, 'fit_saliency', "Карта Значимости (DoG)")
        self._create_analysis_canvas_tab(notebook, 'radon_sinogram', "Синограмма Радона (FBP)")

    def _create_analysis_canvas_tab(self, parent_notebook: ttk.Notebook, key: str, title: str):
        frame = ttk.Frame(parent_notebook)
        parent_notebook.add(frame, text=title)
        canvas = tk.Canvas(frame, bg='#18181b', highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        canvas.bind("<Configure>", lambda e, k=key: self.on_result_canvas_resize(e, k))
        self.result_canvases[key] = canvas

    def _create_notebook_tab(self, key: Any, text: str, notebook: ttk.Notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=text)
        canvas = tk.Canvas(tab, bg='#ffffff', highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.bind("<Configure>", lambda e, k=key: self.on_result_canvas_resize(e, k))
        self.result_canvases[key] = canvas

    def setup_progress_bars(self, num_islands: int):
        pb_frame = self.widgets['main_progress_bars_frame']
        for widget in pb_frame.winfo_children():
            widget.destroy()
        self.main_island_progress_bars.clear()
        self.main_island_labels.clear()
        for i in range(num_islands):
            frame = ttk.Frame(pb_frame)
            frame.pack(fill=tk.X, expand=False, pady=1)
            label = ttk.Label(frame, text=f"Остров {i+1}: старт", width=19, font=('Segoe UI', 8))
            label.pack(side=tk.LEFT, padx=(0, 4))
            self.main_island_labels[i] = label
            pb = ttk.Progressbar(frame, orient='horizontal', mode='determinate')
            pb.pack(fill=tk.X, expand=True)
            self.main_island_progress_bars[i] = pb

    def update_chart(self):
        if not self.ax_fitness or not self.ax_mut or not self.chart_canvas:
            return
        self.ax_fitness.clear()
        self.ax_mut.clear()
        self.ax_fitness.set_facecolor('#ffffff')
        self.ax_mut.set_facecolor('#ffffff')

        num_islands = self.state.get_config()['num_islands']
        colors = plt.cm.tab10(np.linspace(0, 0.8, max(1, num_islands)))
        has_lines = False
        for i in range(num_islands):
            data = self.state.chart_data_by_island.get(i, [])
            if data:
                gens, fitnesses = zip(*data)
                self.ax_fitness.plot(gens, fitnesses, label=f'Остров {i + 1}', color=colors[i], linewidth=1.5)
                has_lines = True
        if self.state.omp_baseline_fitness > 0:
            self.ax_fitness.axhline(self.state.omp_baseline_fitness, color='gray', linestyle=':', label='OMP База')
            has_lines = True

        self.ax_fitness.set_title("Рост точности (Фитнес)", fontsize=9)
        self.ax_fitness.set_xlabel("Поколения", fontsize=8)
        self.ax_fitness.tick_params(labelsize=8)
        if has_lines:
            self.ax_fitness.legend(loc='lower right', fontsize=7)
        self.ax_fitness.grid(True, linestyle='--', alpha=0.6, color='#e1e4e8')

        if self.state.aggregated_mut_stats:
            names = list(self.state.aggregated_mut_stats.keys())
            rates = [
                100.0 * self.state.aggregated_mut_stats[n]['success'] / max(1, self.state.aggregated_mut_stats[n]['attempts'])
                for n in names
            ]
            bars = self.ax_mut.barh(names, rates, color='#3b82f6', height=0.55)
            self.ax_mut.set_xlim(0, max(35.0, max(rates) * 1.15))
            for bar, r in zip(bars, rates):
                self.ax_mut.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2, f"{r:.1f}%", va='center', fontsize=7)
        self.ax_mut.set_title("КПД операторов мутации (%)", fontsize=9)
        self.ax_mut.tick_params(labelsize=8)
        self.ax_mut.grid(True, axis='x', linestyle='--', alpha=0.6, color='#e1e4e8')

        self.chart_canvas.figure.tight_layout(pad=1.0)
        self.chart_canvas.draw()
        self.chart_needs_update = False

    def toggle_ui_state(self, is_running: bool):
        self.state.is_running = is_running
        state = tk.DISABLED if is_running else tk.NORMAL
        for frame_key in ['image_settings_frame', 'algo_frame', 'fitness_frame', 'geometry_frame']:
            frame = self.widgets.get(frame_key)
            if not frame:
                continue
            for child in frame.winfo_children():
                try:
                    child.configure(state=state)
                except (tk.TclError, AttributeError):
                    pass

        self.widgets['load_image_button'].config(state=state)
        self.widgets['start_button'].config(state=state)
        if 'polish_button' in self.widgets:
            self.widgets['polish_button'].config(state=state)
        control_state = tk.NORMAL if is_running else tk.DISABLED
        self.widgets['stop_button'].config(state=control_state)
        self.widgets['pause_button'].config(state=control_state)

        export_state = tk.NORMAL if (not is_running and self.state.overall_best_chromosome) else tk.DISABLED
        for key in ['prune_button', 'export_txt_button', 'export_png_button', 'export_svg_button', 'export_html_button']:
            if key in self.widgets:
                self.widgets[key].config(state=export_state)

        if not is_running:
            self.widgets['pause_button'].config(text="❚❚ Пауза")
            for i, pb in self.main_island_progress_bars.items():
                pb['value'] = 0
                if i in self.main_island_labels:
                    self.main_island_labels[i].config(text=f"Остров {i+1}: готово")

    def update_island_progress(self, island_id: int, stage: int, patience: int, limit: int):
        if island_id in self.main_island_labels:
            pct = int((patience / max(1, limit)) * 100)
            total_st = self.state.get_config().get('stages', 5)
            self.main_island_labels[island_id].config(text=f"Ост.{island_id+1} Эт.{stage + 1}/{total_st}: {pct}%")
        if island_id in self.main_island_progress_bars:
            pb = self.main_island_progress_bars[island_id]
            pb['maximum'] = max(1, limit)
            pb['value'] = patience

    def update_best_fitness_display(self, fitness: float):
        n_lines = max(0, len(self.state.overall_best_chromosome or []) - 1)
        self.widgets['fitness_label'].config(text=f"Точность: {fitness:.5f} ({n_lines} нитей)")

    def update_thread_length_display(self, length_in_meters: float):
        self.widgets['len_label'].config(text=f"{length_in_meters:.1f} м (на Ø30см: {length_in_meters*3:.0f} м)")

    def update_stats_display(self, current_state: AppState):
        elapsed = time.time() - current_state.start_time
        self.widgets['time_label'].config(text=time.strftime('%H:%M:%S', time.gmtime(elapsed)))
        self.widgets['gens_label'].config(text=f"{sum(current_state.generation_counters.values())}")
        self.widgets['mig_label'].config(text=f"{current_state.migration_count}")

        num_islands = max(1, current_state.get_config()['num_islands'])
        total_stages = max(1, current_state.get_config()['stages'])
        total_progress = 0.0
        for i in range(num_islands):
            stage, limit = current_state.island_progress_data.get(i, (0, 1))
            pb = self.main_island_progress_bars.get(i)
            patience = float(pb['value']) if pb else 0.0
            prog_per_stage = 1.0 / total_stages
            total_progress += (stage * prog_per_stage) + (patience / max(1.0, float(limit))) * prog_per_stage

        avg_progress = total_progress / num_islands
        if avg_progress > 0.02:
            eta = (elapsed / avg_progress) * max(0.0, 1.0 - avg_progress)
            self.widgets['eta_label'].config(text=time.strftime('%H:%M:%S', time.gmtime(eta)))
        else:
            self.widgets['eta_label'].config(text="Расчет...")

    def update_canvas_image(self, key: Any, img_pil: Image.Image):
        self.raw_pil_images[key] = img_pil
        canvas = self.result_canvases.get(key)
        if not canvas:
            return
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 10 or h < 10:
            return

        display_size = max(50, min(w, h) - 8)
        if img_pil.width != display_size or img_pil.height != display_size:
            img_shown = img_pil.resize((display_size, display_size), Image.Resampling.LANCZOS)
        else:
            img_shown = img_pil

        # Если это главный холст и включена шторка сравнения "До / После"
        if key == 'overall' and self.state.preprocessed_target_np is not None:
            split_pct = self.state.vars['split_compare'].get()
            if split_pct > 0:
                split_x = int(display_size * (split_pct / 100.0))
                target_resized = cv2.resize(self.state.preprocessed_target_np, (display_size, display_size), interpolation=cv2.INTER_AREA)
                comp_np = np.array(img_shown.convert('L'))
                comp_np[:, :split_x] = target_resized[:, :split_x]
                cv2.line(comp_np, (split_x, 0), (split_x, display_size), 80, 2)
                img_shown = Image.fromarray(comp_np)

        photo_ref = ImageTk.PhotoImage(img_shown)
        self.result_photo_refs[key] = photo_ref
        canvas.delete("all")
        canvas.create_image(w // 2, h // 2, anchor=tk.CENTER, image=photo_ref)

    def update_title(self):
        base_title = "String Art Generator v8.6 Full Studio (Crystal Clear 4x SSAA + Island GA + Radon FBP)"
        if self.state.original_image_path:
            filename = os.path.basename(self.state.original_image_path)
            self.root.title(f"{base_title} — [{filename}]")
        else:
            self.root.title(base_title)


class EvolutionCoordinator:
    def __init__(self, root: 'StringArtApp', state: AppState, ui_manager: UIManager, queues: Dict[str, queue.Queue]):
        self.root = root
        self.state = state
        self.ui_manager = ui_manager
        self.queues = queues
        self.manager_thread: Optional[IslandManagerThread] = None
        self.omp_stop_flag: bool = False
        self.render_queue: queue.Queue = queue.Queue()
        self.render_results_queue: queue.Queue = queue.Queue()
        self.best_chromosomes_by_key: Dict[Any, List[int]] = {}
        self.stats_update_job: Optional[str] = None
        self.canvas_manager: Optional[CanvasManager] = None
        self.radon_guide: Optional[RadonGuide] = None
        self._start_renderer_thread()

    def _start_renderer_thread(self):
        threading.Thread(target=self._renderer_worker, name="RendererThread", daemon=True).start()

    def _renderer_worker(self):
        while True:
            try:
                key, data, render_params = self.render_queue.get()
                # Пропускаем устаревшие кадры в очереди (Frame Dropping), чтобы UI работал мгновенно
                while not self.render_queue.empty():
                    try:
                        next_key, next_data, next_params = self.render_queue.get_nowait()
                        if next_key == key:
                            key, data, render_params = next_key, next_data, next_params
                        else:
                            self.render_queue.put((next_key, next_data, next_params))
                            break
                    except queue.Empty:
                        break

                if self.canvas_manager is None:
                    continue

                chromosome, _, thread_thickness = data
                out_size = render_params.get('output_size', 550)
                line_weight = render_params.get('line_weight', 72)
                max_step = render_params.get('max_step', None)

                img_pil = self.canvas_manager.render_chromosome_to_pil(
                    chromosome,
                    thickness=thread_thickness,
                    line_weight_percent=line_weight,
                    max_step=max_step,
                    output_size=out_size
                )
                self.render_results_queue.put((key, img_pil))
            except Exception:
                logging.error("Ошибка в потоке рендеринга", exc_info=True)


    def start_evolution_only(self):
        """Запуск параллельной островной эволюции на текущем пути без повторного OMP."""
        if self.state.original_image is None:
            messagebox.showwarning("Внимание", "Пожалуйста, сначала загрузите изображение.")
            return
        if not self.state.overall_best_chromosome or len(self.state.overall_best_chromosome) < 4:
            messagebox.showinfo("Информация", "Базовый путь еще не создан. Запускаем полный цикл OMP + Острова...")
            self.start_generation()
            return

        config = self.state.get_config()
        self.omp_stop_flag = False
        self.state.is_running = True
        self.ui_manager.toggle_ui_state(True)
        self.best_chromosomes_by_key.clear()
        self.ui_manager.setup_for_new_run(config['num_islands'])
        self.stats_update_job = self.root.after(1000, self._update_progress_stats)
        threading.Thread(target=self._run_islands_only_worker, args=(config,), name="IslandsOnlyWorker", daemon=True).start()

    def _run_islands_only_worker(self, config: Dict[str, Any]):
        try:
            self._send_status(f"Подготовка фитнес-калькулятора для {config['num_islands']} островов...")
            target_np, edges_np, saliency_uint8, importance_2d = prepare_target_image(
                self.state.original_image, IMAGE_SIZE,
                clahe_clip=config['clahe_clip'], gamma=config['gamma'], shape=config['canvas_shape']
            )
            self.canvas_manager = CanvasManager(IMAGE_SIZE, config['pins'], shape=config['canvas_shape'])
            self.canvas_manager.build_flat_line_table()

            fitness_calc = FitnessCalculator(
                target_np, edges_np, saliency_uint8, importance_2d,
                self.canvas_manager, config, self.queues
            )
            base_chromosome = list(self.state.overall_best_chromosome)
            self._send_status(f"Запуск {config['num_islands']} параллельных островов эволюции...")
            self.manager_thread = IslandManagerThread(config, self.queues, fitness_calc, base_chromosome, self.radon_guide)
            self.manager_thread.start()
        except Exception:
            logging.error("Ошибка при запуске эволюционной доводки", exc_info=True)
            self.queues['status_queue'].put({'type': 'error', 'data': f"Ошибка при доводке: {traceback.format_exc()}"})

    def start_generation(self):
        if self.state.original_image is None:
            messagebox.showwarning("Внимание", "Пожалуйста, сначала загрузите изображение.")
            return
        config = self.state.get_config()
        self.omp_stop_flag = False
        self.state.reset_for_new_run()
        self.ui_manager.toggle_ui_state(True)
        self.best_chromosomes_by_key.clear()
        self.ui_manager.setup_for_new_run(config['num_islands'])
        self.stats_update_job = self.root.after(1000, self._update_progress_stats)
        threading.Thread(target=self._prepare_and_run_evolution, args=(config,), name="PreparationThread", daemon=True).start()

    def _prepare_and_run_evolution(self, config: Dict[str, Any]):
        try:
            self._send_status("Анализ изображения и построение карт значимости...")
            target_np, edges_np, saliency_uint8, importance_2d = prepare_target_image(
                self.state.original_image, IMAGE_SIZE,
                clahe_clip=config['clahe_clip'],
                gamma=config['gamma'],
                shape=config['canvas_shape']
            )
            self.state.preprocessed_target_np = target_np

            self.canvas_manager = CanvasManager(IMAGE_SIZE, config['pins'], config['canvas_shape'])

            radon_guide = None
            self.radon_guide = None
            if config.get('radon_enabled', True) and SKIMAGE_AVAILABLE:
                self._send_status("Вычисление томографической синограммы Радона (Ramp FBP)...")
                self.radon_guide = RadonGuide(target_np, edges_np, self.canvas_manager, config)
                radon_guide = self.radon_guide
                if radon_guide.sinogram is not None:
                    self.queues['status_queue'].put({'type': 'radon_map', 'data': radon_guide.sinogram})

            fc = FitnessCalculator(target_np, edges_np, saliency_uint8, importance_2d, self.canvas_manager, config, self.queues)
            self.queues['status_queue'].put({'type': 'fit_func_maps', 'data': {'saliency': saliency_uint8}})

            self._send_status("Этап 1: Высокоточный OMP-синтез черной нитью...")
            greedy = GreedyInitializer(
                self.canvas_manager, fc.target_darkness_flat, fc.importance_flat,
                config, radon_guide, stop_Check=lambda: self.omp_stop_flag
            )

            def greedy_callback(status: str, path: List[int]):
                self.queues['status_queue'].put({'type': 'greedy_update', 'data': {'text': status, 'path': path}})

            base_chromosome = greedy.run(config['lines'], greedy_callback)
            if not base_chromosome or len(base_chromosome) < 2:
                self.queues['status_queue'].put({'type': 'error', 'data': "Не удалось построить начальный путь."})
                return

            omp_fit, diff_map, depth_2d = fc.calculate_single_fitness_and_maps(base_chromosome)
            self.state.omp_baseline_fitness = omp_fit
            self.queues['status_queue'].put({
                'type': 'island_update',
                'data': {'id': 0, 'chromosome': base_chromosome, 'fitness': omp_fit, 'mut_stats': {}}
            })
            self._push_analytic_maps(diff_map, depth_2d)

            if self.omp_stop_flag:
                self.queues['status_queue'].put({'type': 'done', 'data': "Остановлено на этапе OMP."})
                return

            self.manager_thread = IslandManagerThread(config, self.queues, fc, base_chromosome, radon_guide)
            self.manager_thread.start()
            self.stats_update_job = self.root.after(1000, self._update_progress_stats)

        except Exception:
            logging.error("Ошибка при подготовке эволюции", exc_info=True)
            self.queues['status_queue'].put({'type': 'error', 'data': f"Ошибка при подготовке: {traceback.format_exc()}"})

    def _push_analytic_maps(self, diff_map: np.ndarray, depth_2d: np.ndarray):
        self.render_results_queue.put(('diff_map', self._create_fast_diff_pil(diff_map)))
        self.render_results_queue.put(('density_heatmap', self._create_fast_heatmap_pil(depth_2d)))

    def _create_fast_diff_pil(self, diff_map: np.ndarray) -> Image.Image:
        """Мгновенная генерация красно-синей карты ошибок через NumPy (1 мс вместо вызовов matplotlib)."""
        h, w = diff_map.shape
        rgb = np.full((h, w, 3), 255, dtype=np.uint8)
        max_v = max(0.05, float(np.max(np.abs(diff_map))))
        norm = np.clip(diff_map / max_v, -1.0, 1.0)

        # Положительные значения (перетемнение) -> Красный
        pos = norm > 0
        inten_pos = (norm[pos] * 255).astype(np.uint8)
        rgb[pos, 1] = 255 - inten_pos
        rgb[pos, 2] = 255 - inten_pos

        # Отрицательные значения (недотемнение) -> Синий
        neg = norm < 0
        inten_neg = ((-norm[neg]) * 255).astype(np.uint8)
        rgb[neg, 0] = 255 - inten_neg
        rgb[neg, 1] = 255 - inten_neg
        return Image.fromarray(rgb)

    def _create_fast_heatmap_pil(self, depth_2d: np.ndarray) -> Image.Image:
        """Мгновенная генерация тепловой карты плотности нитей (COLORMAP_INFERNO)."""
        max_d = max(0.1, float(np.max(depth_2d)))
        norm_u8 = np.clip((depth_2d / max_d) * 255.0, 0, 255).astype(np.uint8)
        colored_bgr = cv2.applyColorMap(norm_u8, cv2.COLORMAP_INFERNO)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(colored_rgb)

    def stop_generation(self):
        self.omp_stop_flag = True
        if self.stats_update_job:
            self.root.after_cancel(self.stats_update_job)
            self.stats_update_job = None
        if self.manager_thread and self.manager_thread.is_alive():
            self.manager_thread.stop()
            self._send_status("Остановка потоков...")
        self.root.after(250, self._finalize_run)

    def _finalize_run(self):
        self.ui_manager.toggle_ui_state(False)

    def toggle_pause(self):
        if not self.manager_thread or not self.manager_thread.is_alive():
            return
        self.state.is_paused = not self.state.is_paused
        if self.state.is_paused:
            self.manager_thread.pause()
            self.ui_manager.widgets['pause_button'].config(text="► Продолжить")
            self._send_status("Пауза...")
        else:
            self.manager_thread.resume()
            self.ui_manager.widgets['pause_button'].config(text="❚❚ Пауза")
            self.stats_update_job = self.root.after(1000, self._update_progress_stats)

    def process_queues(self):
        try:
            while not self.queues['status_queue'].empty():
                msg = self.queues['status_queue'].get_nowait()
                handler = self._get_message_handler(msg.get('type'))
                handler(msg.get('data'))

            while not self.render_results_queue.empty():
                key, img_pil = self.render_results_queue.get_nowait()
                self.ui_manager.update_canvas_image(key, img_pil)
        except queue.Empty:
            pass
        finally:
            self.root.after(80, self.process_queues)

    def _get_message_handler(self, msg_type: str) -> Callable:
        return {
            'greedy_update': self._handle_greedy_update,
            'status': self._handle_status,
            'error': self._handle_error,
            'gpu_status': self._handle_gpu_status,
            'generation_tick': self._handle_generation_tick,
            'island_update': self._handle_island_update,
            'done': self._handle_done,
            'migration': self._handle_migration,
            'fit_func_maps': self._handle_fit_func_maps,
            'radon_map': self._handle_radon_map,
        }.get(msg_type, lambda d: None)

    def _handle_greedy_update(self, data: Dict[str, Any]):
        self.ui_manager.widgets['status_label'].config(text=data['text'])
        path = data['path']
        self.state.overall_best_chromosome = path
        self.best_chromosomes_by_key['overall'] = path
        self.state.vars['step_scrubber'].set(max(10, len(path) - 1))
        config = self.state.get_config()
        self.request_canvas_update('overall', (path, config['pins'], config['thread_thickness']), {})

    def _handle_status(self, data: str):
        self.ui_manager.widgets['status_label'].config(text=data)

    def _handle_error(self, data: str):
        messagebox.showerror("Ошибка", data)
        self.ui_manager.widgets['status_label'].config(text="Ошибка!")
        self.ui_manager.toggle_ui_state(False)

    def _handle_gpu_status(self, data: bool):
        mode = "Numba JIT + CUDA" if data else ("Numba JIT (Быстрый CPU)" if NUMBA_AVAILABLE else "Стандартный CPU")
        self.ui_manager.widgets['compute_mode_label'].config(text=f"Ядро: {mode}")

    def _handle_generation_tick(self, data: Tuple[int, int, int, int]):
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

    def _handle_island_update(self, data: Dict[str, Any]):
        island_id, chromosome, fitness = data['id'], data['chromosome'], data['fitness']
        mut_stats = data.get('mut_stats', {})
        if mut_stats:
            self.state.aggregated_mut_stats = mut_stats
            self.ui_manager.chart_needs_update = True

        self.state.island_best_fitnesses[island_id] = fitness
        self.best_chromosomes_by_key[island_id] = chromosome
        config = self.state.get_config()
        render_data = (chromosome, config['pins'], config['thread_thickness'])

        # Рисуем вкладку острова только если она сейчас открыта пользователем
        notebook = self.ui_manager.widgets['results_notebook']
        sel = notebook.select()
        if sel and notebook.index(sel) == island_id + 1:
            self.request_canvas_update(island_id, render_data, {})

        if fitness > self.state.overall_best_fitness:
            self.state.overall_best_fitness = fitness
            self.state.overall_best_chromosome = list(chromosome)
            self.best_chromosomes_by_key['overall'] = list(chromosome)
            self.state.vars['step_scrubber'].set(max(10, len(chromosome) - 1))
            self.ui_manager.update_best_fitness_display(fitness)
            self.ui_manager.update_thread_length_display(self._calculate_thread_length(chromosome))
            self.request_canvas_update('overall', render_data, {})

            if self.manager_thread and self.manager_thread.fitness_calculator:
                _, diff_map, depth_2d = self.manager_thread.fitness_calculator.calculate_single_fitness_and_maps(chromosome)
                self._push_analytic_maps(diff_map, depth_2d)

    def _handle_fit_func_maps(self, data: Dict[str, Optional[np.ndarray]]):
        if data.get('saliency') is not None:
            sal_bgr = cv2.applyColorMap(data['saliency'], cv2.COLORMAP_VIRIDIS)
            sal_rgb = cv2.cvtColor(sal_bgr, cv2.COLOR_BGR2RGB)
            self.render_results_queue.put(('fit_saliency', Image.fromarray(sal_rgb)))

    def _handle_radon_map(self, sinogram_data: np.ndarray):
        if sinogram_data is None:
            return
        sino_u8 = np.clip(sinogram_data * 255.0, 0, 255).astype(np.uint8)
        sino_resized = cv2.resize(sino_u8, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        sino_color = cv2.cvtColor(cv2.applyColorMap(sino_resized, cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)
        self.render_results_queue.put(('radon_sinogram', Image.fromarray(sino_color)))

    def _handle_migration(self, _: Any):
        self.state.migration_count += 1

    def _handle_done(self, data: str):
        self._finalize_run()
        removed = self.prune_chromosome(silent=True)
        msg = f"{data} (Нитей: {max(0, len(self.state.overall_best_chromosome or []) - 1)}, удалено петель: {removed})"
        self.ui_manager.widgets['status_label'].config(text=msg)

    def _send_status(self, message: str):
        self.queues['status_queue'].put({'type': 'status', 'data': message})

    def request_canvas_update(self, key: Any, data: Any, render_params: Dict[str, Any]):
        if not self.canvas_manager:
            return
        canvas = self.ui_manager.result_canvases.get(key)
        w = canvas.winfo_width() if canvas else 550
        h = canvas.winfo_height() if canvas else 550
        out_size = max(320, min(w, h) - 10) if (w > 20 and h > 20) else 550
        params = dict(render_params)
        params['output_size'] = out_size
        try:
            params['line_weight'] = self.state.vars['line_weight'].get()
            params['max_step'] = self.state.vars['step_scrubber'].get() if key == 'overall' else None
        except Exception:
            params['line_weight'] = 72
            params['max_step'] = None
        self.render_queue.put((key, data, params))

    def _calculate_thread_length(self, chromosome: Optional[List[int]]) -> float:
        if not chromosome or not self.canvas_manager or len(chromosome) < 2:
            return 0.0
        coords = self.canvas_manager.pin_coords
        idx = np.asarray(chromosome, dtype=np.int64)
        diffs = coords[idx[1:]] - coords[idx[:-1]]
        total_px = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
        scale_factor = 10.0 / float(self.canvas_manager.size)  # для базового диаметра 10 см
        return (total_px * scale_factor) / 100.0

    def _update_progress_stats(self):
        if not self.state.is_running:
            return
        self.ui_manager.update_stats_display(self.state)
        self.stats_update_job = self.root.after(1000, self._update_progress_stats)

    def prune_chromosome(self, silent: bool = False) -> int:
        if not self.state.overall_best_chromosome or len(self.state.overall_best_chromosome) < 3:
            return 0
        chromo = self.state.overall_best_chromosome
        pruned = [chromo[0]]
        i = 1
        while i < len(chromo) - 1:
            if pruned and chromo[i + 1] == pruned[-1]:
                i += 2
            elif chromo[i] == pruned[-1]:
                i += 1
            else:
                pruned.append(chromo[i])
                i += 1
        if i == len(chromo) - 1 and chromo[-1] != pruned[-1]:
            pruned.append(chromo[-1])

        removed = len(chromo) - len(pruned)
        if removed > 0:
            self.state.overall_best_chromosome = pruned
            self.best_chromosomes_by_key['overall'] = pruned
            self.state.vars['step_scrubber'].set(max(10, len(pruned) - 1))
            config = self.state.get_config()
            self.request_canvas_update('overall', (pruned, config['pins'], config['thread_thickness']), {'force': True})
            self.ui_manager.update_thread_length_display(self._calculate_thread_length(pruned))
            if not silent:
                messagebox.showinfo("Оптимизация пути", f"Удалено холостых возвратных петель: {removed}\nИтого четких нитей: {len(pruned)-1}")
        elif not silent:
            messagebox.showinfo("Оптимизация пути", "Холостых петель не обнаружено, путь уже оптимален!")
        return removed

    def export_txt(self):
        if not self.state.overall_best_chromosome:
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if not filepath:
            return
        steps_limit = min(len(self.state.overall_best_chromosome), self.state.vars['step_scrubber'].get() + 1)
        active_chromo = self.state.overall_best_chromosome[:steps_limit]
        total_m = self._calculate_thread_length(active_chromo)
        config = self.state.get_config()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=====================================================\n")
            f.write("   СХЕМА ПЛЕТЕНИЯ STRING ART (v8.6 Crystal Clear)\n")
            f.write("=====================================================\n")
            f.write(f"Форма рамы: {config['canvas_shape']} | Всего гвоздей: {config['pins']}\n")
            f.write(f"Количество переходов (нитей): {len(active_chromo) - 1}\n")
            f.write(f"Расход нити: ~{total_m:.1f} м (для Ø10см) | ~{total_m*3:.0f} м (для Ø30см) | ~{total_m*5:.0f} м (для Ø50см)\n")
            f.write("-----------------------------------------------------\n")
            for i, pin in enumerate(active_chromo):
                f.write(f"Шаг {i + 1:4d}: Гвоздь {pin}\n")
        messagebox.showinfo("Экспорт TXT", f"Схема гвоздей успешно сохранена:\n{filepath}")

    def export_png(self):
        if not self.state.overall_best_chromosome:
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png")])
        if not filepath:
            return
        self._send_status("Экспорт Ultra HD PNG (2400x2400px с субпиксельным сглаживанием)...")
        config = self.state.get_config()
        cm = CanvasManager(IMAGE_SIZE, config['pins'], config['canvas_shape'])
        img = cm.render_chromosome_to_pil(
            self.state.overall_best_chromosome,
            thickness=config['thread_thickness'],
            line_weight_percent=config['line_weight'],
            max_step=self.state.vars['step_scrubber'].get(),
            output_size=2400
        )
        img.save(filepath)
        self._send_status("Экспорт PNG 2400x2400 завершен.")
        messagebox.showinfo("Экспорт PNG", f"Изображение высокого разрешения сохранено в:\n{filepath}")


    def export_html(self):
        if not self.state.overall_best_chromosome:
            messagebox.showwarning("Внимание", "Сначала необходимо запустить генерацию.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML5 Web Assistant", "*.html")],
                                         title="Сохранить автономный веб-ассистент плетения")
        if not p:
            return
        try:
            chromosome = self.state.overall_best_chromosome
            coords = self.canvas_manager.pin_coords
            pins_json = json.dumps([[round(float(x), 2), round(float(y), 2)] for x, y in coords])
            layers_json = json.dumps([{"name": "Черная нить", "color": "#111111", "path": [int(x) for x in chromosome]}])
            html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>String Art Studio — Веб-ассистент</title>
<style>
body {{ background: #f6f8fa; color: #1f2328; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; flex-direction: column; align-items: center; padding: 12px; margin: 0; }}
h1 {{ font-size: 20px; color: #0969da; margin: 8px; }}
.card {{ background: #ffffff; border: 1px solid #d0d7de; border-radius: 10px; padding: 10px 20px; margin: 6px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.06); width: 90%; max-width: 500px; }}
.pins {{ font-size: 28px; font-weight: bold; color: #0969da; }}
#cwrap {{ width: min(85vw, 500px); height: min(85vw, 500px); background: #ffffff; border-radius: 50%; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: 10px; overflow: hidden; }}
canvas {{ width: 100%; height: 100%; }}
.ctrls {{ display: flex; gap: 8px; margin: 10px; }}
button {{ background: #f6f8fa; border: 1px solid #d0d7de; color: #1f2328; padding: 10px 16px; border-radius: 8px; font-size: 15px; cursor: pointer; }}
button.prim {{ background: #0969da; color: #ffffff; font-weight: bold; border: none; }}
</style>
</head>
<body>
<h1>String Art Studio — Ассистент плетения</h1>
<div class="card">
  <div style="font-size:12px;color:#57606a;">ТЕКУЩИЙ ХОД</div>
  <div class="pins" id="pinfo">0 &rarr; 0</div>
  <div id="sinfo" style="font-size:13px;color:#57606a;">Шаг 1 из {len(chromosome)-1}</div>
</div>
<div id="cwrap"><canvas id="cv" width="1000" height="1000"></canvas></div>
<div class="ctrls">
  <button id="bp">&larr; Назад (A)</button>
  <button id="bn" class="prim">Вперед (D / Пробел) &rarr;</button>
  <button id="bv">&#128266; Голос: ВКЛ</button>
</div>
<script>
const PINS = {pins_json};
const PATH = {layers_json}[0].path;
let step = 1, voice = true;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
function draw() {{
  document.getElementById('pinfo').innerHTML = PATH[step-1] + ' &rarr; ' + PATH[step];
  document.getElementById('sinfo').innerText = 'Шаг ' + step + ' из ' + (PATH.length - 1);
  ctx.clearRect(0, 0, 1000, 1000);
  const sc = 2.0;
  ctx.strokeStyle = '#111111'; ctx.lineWidth = 1.2 * sc; ctx.globalAlpha = 0.35; ctx.beginPath();
  for (let i = 0; i < step; i++) {{
    ctx.moveTo(PINS[PATH[i]][0]*sc, PINS[PATH[i]][1]*sc);
    ctx.lineTo(PINS[PATH[i+1]][0]*sc, PINS[PATH[i+1]][1]*sc);
  }}
  ctx.stroke();
  ctx.globalAlpha = 1.0; ctx.strokeStyle = '#0969da'; ctx.lineWidth = 3.5 * sc;
  ctx.beginPath(); ctx.moveTo(PINS[PATH[step-1]][0]*sc, PINS[PATH[step-1]][1]*sc);
  ctx.lineTo(PINS[PATH[step]][0]*sc, PINS[PATH[step]][1]*sc); ctx.stroke();
  if (voice && 'speechSynthesis' in window) {{
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance('на ' + PATH[step]); u.lang = 'ru-RU'; window.speechSynthesis.speak(u);
  }}
}}
function nxt(d) {{ step = Math.max(1, Math.min(PATH.length-1, step+d)); draw(); }}
document.getElementById('bn').onclick = () => nxt(1);
document.getElementById('bp').onclick = () => nxt(-1);
document.getElementById('bv').onclick = (e) => {{ voice = !voice; e.target.innerText = voice ? '&#128266; Голос: ВКЛ' : '&#128263; Голос: ВЫКЛ'; }};
window.onkeydown = (e) => {{ if (e.key === ' ' || e.key === 'ArrowRight' || e.key === 'd') nxt(1); else if (e.key === 'ArrowLeft' || e.key === 'a') nxt(-1); }};
draw();
</script>
</body>
</html>"""
            with open(p, 'w', encoding='utf-8') as out_f:
                out_f.write(html)
            messagebox.showinfo("Успех", f"Автономный веб-ассистент успешно сохранен в:\n{p}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось экспортировать HTML:\n{e}")

    def export_svg(self):
        if not self.state.overall_best_chromosome:
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG vector", "*.svg")])
        if not filepath:
            return
        config = self.state.get_config()
        SVG_SIZE = 1200
        cm = CanvasManager(SVG_SIZE, config['pins'], config['canvas_shape'])
        coords = cm.pin_coords
        steps_limit = min(len(self.state.overall_best_chromosome), self.state.vars['step_scrubber'].get() + 1)
        active_chromo = self.state.overall_best_chromosome[:steps_limit]

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f'<svg width="{SVG_SIZE}" height="{SVG_SIZE}" viewBox="0 0 {SVG_SIZE} {SVG_SIZE}" xmlns="http://www.w3.org/2000/svg">\n')
            f.write('  <rect width="100%" height="100%" fill="white"/>\n')
            path_data = "M " + " L ".join([f"{coords[p][0]:.2f},{coords[p][1]:.2f}" for p in active_chromo])
            f.write(f'  <path d="{path_data}" stroke="black" stroke-opacity="0.78" stroke-width="{config["thread_thickness"]*0.65:.2f}" fill="none" stroke-linejoin="round" stroke-linecap="round"/>\n')
            f.write('</svg>\n')
        messagebox.showinfo("Экспорт SVG", f"Векторный файл SVG успешно сохранен:\n{filepath}")


# ----------------------------------------------------------------------
# РАЗДЕЛ 6: ГЛАВНОЕ ОКНО ПРИЛОЖЕНИЯ
# ----------------------------------------------------------------------

class StringArtApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("1520x940")
        self.state = AppState()
        self.queues = {'status_queue': queue.Queue()}
        self.ui_manager = UIManager(self, self.state)
        self.coordinator = EvolutionCoordinator(self, self.state, self.ui_manager, self.queues)
        self.ui_manager.coordinator = self.coordinator

        self.ui_manager.create_widgets()
        self.ui_manager.update_title()
        self.ui_manager.widgets['load_image_button'].config(command=self.load_image)
        self.ui_manager.setup_for_new_run(self.state.vars['num_islands'].get())

        self.after(80, self.coordinator.process_queues)
        self.after(800, self._periodic_ui_update)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        if self.state.is_running and messagebox.askokcancel("Выход", "Генерация еще выполняется. Остановить и выйти?"):
            self.coordinator.stop_generation()
            self.destroy()
        elif not self.state.is_running:
            self.destroy()

    def _periodic_ui_update(self):
        if self.ui_manager.chart_needs_update:
            self.ui_manager.update_chart()
        self.after(800, self._periodic_ui_update)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp")])
        if not file_path:
            return
        try:
            self.state.original_image = Image.open(file_path)
            self.state.original_image_path = file_path
            self.ui_manager.update_title()
            self.apply_auto_recommendations()
            self.ui_manager.widgets['status_label'].config(text="Изображение загружено и параметры подобраны.")
        except Exception as e:
            logging.error("Не удалось загрузить изображение", exc_info=True)
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение:\n{e}")

    def apply_auto_recommendations(self):
        if not self.state.original_image:
            return
        shape = self.state.vars['canvas_shape'].get()
        rec = auto_recommend_parameters(self.state.original_image, shape)
        for k in ['pins', 'lines', 'line_weight', 'clahe_clip', 'gamma', 'fearless_factor', 'contour_weight', 'radon_influence', 'stages']:
            if k in self.state.vars:
                self.state.vars[k].set(rec[k])
        self.state.vars['step_scrubber'].set(rec['lines'])
        self.ui_manager.widgets['analysis_label'].config(text=rec['summary'], foreground='#0f5132')
        self.redraw_target_canvas()

    def redraw_target_canvas(self, event=None):
        if not self.state.original_image:
            return
        canvas = self.ui_manager.widgets['target_canvas']
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 20 or h < 20:
            return
        display_size = min(w, h) - 10
        target_np, _, _, _ = prepare_target_image(
            self.state.original_image,
            display_size,
            clahe_clip=self.state.vars['clahe_clip'].get(),
            gamma=self.state.vars['gamma'].get(),
            shape=self.state.vars['canvas_shape'].get()
        )
        self.state.preprocessed_target_np = target_np
        img_pil = Image.fromarray(target_np)
        self.state.target_image_tk = ImageTk.PhotoImage(img_pil)
        canvas.delete("all")
        canvas.create_image(w // 2, h // 2, anchor=tk.CENTER, image=self.state.target_image_tk)


if __name__ == "__main__":
    try:
        if getattr(sys, 'frozen', False):
            os.environ["OPENCV_SALIENCY_PATH"] = os.path.join(sys._MEIPASS, 'opencv-saliency')
        app = StringArtApp()
        app.mainloop()
    except Exception:
        logging.critical("Критическая ошибка при запуске приложения", exc_info=True)
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Критическая ошибка", f"Не удалось запустить приложение:\n\n{traceback.format_exc()}")