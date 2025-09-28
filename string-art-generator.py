# -*- coding: utf-8 -*-
"""
Генератор изображений в технике String Art с использованием гибридного алгоритма.
Версия 8.6:
"""

# ----------------------------------------------------------------------
# РАЗДЕЛ 1: ИМПОРТЫ И ГЛОБАЛЬНЫЕ НАСТРОЙКИ
# ----------------------------------------------------------------------
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, font
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
import json
from typing import Dict, Any, List, Optional, Tuple, Callable

# --- Библиотеки для работы с изображениями и построения графиков ---
from PIL import Image, ImageTk
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# --- Опциональные библиотеки для ускорения и улучшения вычислений ---
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
    import torch
    GPU_ENABLED_FLAG = torch.cuda.is_available()
except ImportError:
    torch, GPU_ENABLED_FLAG = None, False
try:
    from lpips import LPIPS
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS, LPIPS_AVAILABLE = None, False
try:
    from scipy.ndimage import distance_transform_edt
    SCIPY_AVAILABLE = True
except ImportError:
    distance_transform_edt, SCIPY_AVAILABLE = None, False
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    njit = lambda *args, **kwargs: (lambda f: f) # Dummy decorator
    NUMBA_AVAILABLE = False


# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# --- Глобальные константы ---
IMAGE_SIZE = 400
GREEDY_SUBTRACTION_VALUE = 20
RENDER_ALPHA_BASE = 1600
RENDER_ALPHA_MAX = 0.2
PATIENCE_BASE = 20
PATIENCE_STAGE_FACTOR = 15
STAGNATION_MUTATION_FACTOR = 2.0

TOOLTIPS = {
    'load_image': "Открыть файл изображения (JPG, PNG).\nРекомендуются контрастные портреты с четким фоном.",
    'pins': "Количество точек (гвоздей) по периметру.\nБольше = выше детализация. Рекомендуется: 256-350.",
    'lines': "Общее количество линий (нитей).\nБольше = темнее и плотнее. Рекомендуется: 2000-4000.",
    'thread_thickness': "Толщина нитей для отображения и экспорта.\nНе влияет на скорость генерации.",
    'num_islands': "Количество независимых популяций (потоков).\nРекомендуется ставить равным количеству ядер вашего CPU.",
    'population_size': "Количество вариантов на каждом 'острове'.\nБольше = лучше поиск, но медленнее. Рекомендуется: 50-100.",
    'stages': "Количество стадий оптимизации.\nПозволяет алгоритму уточнять результат. Рекомендуется: 4-8.",
    'mutation_rate': "Базовая вероятность случайных изменений (%)\nПомогает избегать 'застревания'. Рекомендуется: 10-20%.",
    'migration_interval': "Частота обмена лучшими вариантами между островами.\nРекомендуется: 20-30 поколений.",
    'contour_weight': "Насколько сильно алгоритм фокусируется на контурах.\nВыше значение = четче края. Рекомендуется: 20-40%.",
    'memetic_enabled': "Включить локальный поиск (Меметический алгоритм).\n'Дополировывает' лучшие решения, ускоряя сходимость. Рекомендуется включить.",
    'memetic_intensity': "Процент лучших особей для 'полировки'.\nРекомендуется: 10-20%.",
    'memetic_depth': "Количество итераций 'полировки' для каждой особи.\nРекомендуется: 20-40.",
    'radon_enabled': "ГИБРИД: Использовать преобразование Радона для направления поиска.\nУскоряет сходимость на глобальном уровне, находя важные структуры.",
    'radon_influence': "ГИБРИД: Сила влияния Радона (%).\nОпределяет, насколько сильно Радон смещает выбор линий.",
    'start': "Начать процесс генерации.",
    'pause': "Приостановить/возобновить генерацию.",
    'stop': "Остановить генерацию и сохранить текущий лучший результат.",
    'export_txt': "Сохранить последовательность номеров гвоздей для ручной работы.",
    'export_png': "Сохранить итоговую картинку в высоком разрешении (2000x2000px).",
    'export_svg': "Сохранить результат в векторном формате SVG,\nидеальном для масштабирования и печати.",
}


# ----------------------------------------------------------------------
# РАЗДЕЛ 2: НИЗКОУРОВНЕВЫЕ УТИЛИТЫ
# ----------------------------------------------------------------------

def prepare_target_image(original_image: Optional[Image.Image], target_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if original_image is None:
        empty_img = np.full((target_size, target_size), 255, dtype=np.uint8)
        empty_maps = np.zeros((target_size, target_size), dtype=np.uint8)
        return empty_img, empty_maps, empty_maps

    img_gray = np.array(original_image.convert('L'))

    background = np.full((target_size, target_size), 255, dtype=np.uint8)
    h, w = img_gray.shape
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    background[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    blurred = cv2.GaussianBlur(background, (3, 3), 0)
    gradient_magnitude = np.sqrt(cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)**2 + cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)**2)
    edges = (gradient_magnitude / np.max(gradient_magnitude) * 255).astype(np.uint8) if np.max(gradient_magnitude) > 0 else np.zeros_like(background, dtype=np.uint8)

    saliency_map = np.zeros_like(background, dtype=np.uint8)
    try:
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        _, saliency_map_raw = saliency.computeSaliency(cv2.cvtColor(background, cv2.COLOR_GRAY2BGR))
        saliency_map = (saliency_map_raw * 255).astype(np.uint8)
    except (cv2.error, AttributeError):
        logging.warning("Advanced saliency algorithm not available. Falling back to Canny.")
        saliency_map = cv2.Canny(blurred, 50, 150)

    return background, edges, saliency_map


def _calculate_line_pixels_numpy(x1: float, y1: float, x2: float, y2: float, size: int) -> np.ndarray:
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0: return np.empty(0, dtype=np.int64)

    x = np.linspace(x1, x2, length)
    y = np.linspace(y1, y2, length)
    cc = np.round(x).astype(np.int64)
    rr = np.round(y).astype(np.int64)

    valid_indices = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
    return rr[valid_indices] * size + cc[valid_indices]


# ----------------------------------------------------------------------
# РАЗДЕЛ 2.5: КЛАСС ДЛЯ РАБОТЫ С ПРЕОБРАЗОВАНИЕМ РАДОНА
# ----------------------------------------------------------------------

class RadonGuide:
    def __init__(self, target_image_np: np.ndarray, canvas_manager: 'CanvasManager', config: Dict[str, Any]):
        self.config = config
        self.canvas_manager = canvas_manager
        self.sinogram: Optional[np.ndarray] = None
        self.line_weights: Dict[Tuple[int, int], float] = {}
        
        if self.config.get('radon_enabled', False) and radon is not None:
            self._compute_sinogram_and_weights(target_image_np)

    def _compute_sinogram_and_weights(self, target_image: np.ndarray):
        logging.info("HYBRID: Calculating Radon transform...")
        image_for_radon = 255 - target_image
        theta = np.linspace(0., 180., max(image_for_radon.shape), endpoint=False)
        self.sinogram = radon(image_for_radon, theta=theta, circle=True)
        
        if self.sinogram.max() > 0:
            self.sinogram = (self.sinogram - self.sinogram.min()) / (self.sinogram.max() - self.sinogram.min())
            self.sinogram = np.power(self.sinogram, 2)
        
        center = self.canvas_manager.size / 2.0
        num_pins, coords = self.canvas_manager.num_pins, self.canvas_manager.pin_coords

        for i in range(num_pins):
            for j in range(i + 1, num_pins):
                p1, p2 = coords[i], coords[j]
                mid_point = ((p1[0] + p2[0]) / 2 - center, (p1[1] + p2[1]) / 2 - center)
                
                rho = mid_point[0] * math.cos(math.atan2(mid_point[1], mid_point[0])) + \
                      mid_point[1] * math.sin(math.atan2(mid_point[1], mid_point[0]))
                angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 180
                
                rho_idx = int(rho + self.sinogram.shape[0] / 2)
                theta_idx = int(angle)
                
                if 0 <= rho_idx < self.sinogram.shape[0] and 0 <= theta_idx < self.sinogram.shape[1]:
                    self.line_weights[tuple(sorted((i, j)))] = self.sinogram[rho_idx, theta_idx]
        logging.info(f"HYBRID: Radon weight map created for {len(self.line_weights)} lines.")

    def get_weighted_candidates(self, current_pin: int, num_candidates: int) -> List[int]:
        if not self.line_weights:
            return [p for p in range(self.canvas_manager.num_pins) if p != current_pin]

        pins, weights = [], []
        for p_idx in range(self.canvas_manager.num_pins):
            if p_idx == current_pin: continue
            key = tuple(sorted((current_pin, p_idx)))
            pins.append(p_idx)
            weights.append(self.line_weights.get(key, 0.01))
        
        influence = self.config.get('radon_influence', 50) / 100.0
        total_weight = sum(weights)
        if total_weight > 0:
            probabilities = [(w / total_weight) * influence + (1.0 / len(pins)) * (1.0 - influence) for w in weights]
        else:
            probabilities = None
            
        return random.choices(pins, weights=probabilities, k=min(num_candidates, len(pins)))


# ----------------------------------------------------------------------
# РАЗДЕЛ 3: КЛАССЫ-ИНСТРУМЕНТЫ ЯДРА АЛГОРИТМА
# ----------------------------------------------------------------------

class CanvasManager:
    def __init__(self, size: int, num_pins: int, shape: str = 'Круг'):
        self.size, self.num_pins, self.shape = size, num_pins, shape
        self.pin_coords = self._calculate_pin_coords()
        self.pixel_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self.mask_cache: Dict[Tuple[Any, ...], np.ndarray] = {}
        self.mask_cache_lock = threading.Lock()

    def _calculate_pin_coords(self) -> np.ndarray:
        return self._calculate_square_coords() if self.shape == 'Квадрат' else self._calculate_circular_coords()

    def _calculate_circular_coords(self) -> np.ndarray:
        coords, radius, center = [], self.size / 2 - (self.size / 40), self.size / 2
        for i in range(self.num_pins):
            angle = (i / self.num_pins) * 2 * math.pi
            coords.append((center + radius * math.cos(angle), center + radius * math.sin(angle)))
        return np.array(coords)

    def _calculate_square_coords(self) -> np.ndarray:
        coords, margin = [], self.size / 40
        side, perimeter = self.size - 2 * margin, (self.size - 2 * margin) * 4
        if self.num_pins == 0: return np.array([])
        dist_between = perimeter / self.num_pins
        for i in range(self.num_pins):
            dist = i * dist_between
            if dist <= side: x, y = margin + dist, margin
            elif dist <= 2 * side: x, y = self.size - margin, margin + (dist - side)
            elif dist <= 3 * side: x, y = (self.size - margin) - (dist - 2 * side), self.size - margin
            else: x, y = margin, (self.size - margin) - (dist - 3 * side)
            coords.append((x, y))
        return np.array(coords)

    def get_line_pixel_indices(self, p1_idx: int, p2_idx: int) -> np.ndarray:
        key = tuple(sorted((p1_idx, p2_idx)))
        if key in self.pixel_cache: return self.pixel_cache[key]
        pixels = _calculate_line_pixels_numpy(*self.pin_coords[p1_idx], *self.pin_coords[p2_idx], self.size)
        self.pixel_cache[key] = pixels
        return pixels

    def get_line_mask_cpu(self, p1_idx: int, p2_idx: int, thickness: int) -> np.ndarray:
        key = (tuple(sorted((p1_idx, p2_idx))), self.size, thickness)
        if key in self.mask_cache: return self.mask_cache[key]
        with self.mask_cache_lock:
            if key in self.mask_cache: return self.mask_cache[key]
            mask = np.zeros((self.size, self.size), dtype=np.float32)
            p1, p2 = tuple(self.pin_coords[p1_idx].astype(int)), tuple(self.pin_coords[p2_idx].astype(int))
            cv2.line(mask, p1, p2, 1.0, thickness, cv2.LINE_AA)
            self.mask_cache[key] = mask
            return mask

    def render_chromosome_to_numpy(self, chromosome: List[int], thickness: int = 1) -> np.ndarray:
        if not chromosome or len(chromosome) < 2:
            return np.full((self.size, self.size), 255, dtype=np.uint8)
        darkness_map = np.zeros((self.size, self.size), dtype=np.float32)
        num_lines = len(chromosome) - 1
        alpha = min(RENDER_ALPHA_MAX, RENDER_ALPHA_BASE / (num_lines + 1)) if num_lines > 0 else RENDER_ALPHA_MAX
        for i in range(num_lines):
            line_mask = self.get_line_mask_cpu(chromosome[i], chromosome[i+1], thickness)
            darkness_map += line_mask * alpha
        return np.clip((1.0 - darkness_map) * 255, 0, 255).astype(np.uint8)

    def render_chromosome_to_pil(self, chromosome: List[int], thickness: int = 1) -> Image.Image:
        return Image.fromarray(self.render_chromosome_to_numpy(chromosome, thickness))

    def precompute_line_data_for_gpu(self, device: Any) -> Dict[Tuple[int, int], Any]:
        line_pixels_map_gpu = {}
        for i in range(self.num_pins):
            for j in range(i + 1, self.num_pins):
                pixel_indices_np = self.get_line_pixel_indices(i, j)
                if len(pixel_indices_np) > 0:
                    key = tuple(sorted((i, j)))
                    line_pixels_map_gpu[key] = torch.from_numpy(pixel_indices_np).long().to(device)
        return line_pixels_map_gpu


class FitnessCalculator:
    def __init__(self, target_2d: np.ndarray, target_edges_2d: np.ndarray, saliency_map: np.ndarray,
                 canvas_manager: CanvasManager, config: Dict[str, Any], queues: Dict[str, queue.Queue]):
        self.canvas_manager, self.queues, self.config = canvas_manager, queues, config
        self.line_pixels_gpu_map: Optional[Dict[Tuple[int, int], Any]] = None
        self.target_2d = target_2d
        self.saliency_map = (saliency_map.astype(np.float32) / 255.0 + 0.1) / 1.1
        self.distance_transform_map = None
        if SCIPY_AVAILABLE:
            dtm = distance_transform_edt(255 - target_edges_2d)
            if dtm.max() > 0: self.distance_transform_map = dtm / dtm.max()
        self.lpips_model, self.device = None, None
        if GPU_ENABLED_FLAG and self.config['thread_thickness'] == 1:
            self.device = torch.device("cuda")
            self.target_gpu = torch.from_numpy(self.target_2d.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(self.device)
            self.saliency_gpu = torch.from_numpy(self.saliency_map).unsqueeze(0).unsqueeze(0).to(self.device)
            if LPIPS_AVAILABLE:
                try: self.lpips_model = LPIPS(net='alex').to(self.device)
                except Exception: logging.error("Failed to load LPIPS model.", exc_info=True)
            if self.distance_transform_map is not None:
                self.dt_map_gpu = torch.from_numpy(self.distance_transform_map).float().to(self.device)
            self.queues['status_queue'].put({'type': 'gpu_status', 'data': True})
        if not self.device: self.queues['status_queue'].put({'type': 'gpu_status', 'data': False})

    def enable_gpu_rendering(self, line_pixels_gpu_map: Dict[Tuple[int, int], Any]):
        if self.device: self.line_pixels_gpu_map = line_pixels_gpu_map

    def _render_chromosome_batch_gpu(self, chromosomes: List[List[int]]) -> torch.Tensor:
        batch_size, img_size = len(chromosomes), self.canvas_manager.size
        flat_size = img_size * img_size
        max_retries, chunk_size = 3, 16
        for _ in range(max_retries):
            try:
                batch_img_flat = torch.zeros(batch_size * flat_size, dtype=torch.float32, device=self.device)
                for chunk_start in range(0, batch_size, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, batch_size)
                    all_indices, all_values = [], []
                    for i in range(chunk_start, chunk_end):
                        chromo = chromosomes[i]
                        if len(chromo) <= 1: continue
                        alpha = min(RENDER_ALPHA_MAX, RENDER_ALPHA_BASE / len(chromo))
                        for j in range(len(chromo) - 1):
                            key = tuple(sorted((chromo[j], chromo[j + 1])))
                            if self.line_pixels_gpu_map and key in self.line_pixels_gpu_map:
                                indices = self.line_pixels_gpu_map[key] + i * flat_size
                                all_indices.append(indices)
                                all_values.append(torch.full((len(indices),), alpha, device=self.device, dtype=torch.float32))
                    if all_indices:
                        batch_img_flat.index_add_(0, torch.cat(all_indices), torch.cat(all_values))
                return torch.clamp(batch_img_flat.view(batch_size, 1, img_size, img_size), 0, 1)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and chunk_size > 1:
                    logging.warning(f"OOM with chunk_size={chunk_size}. Retrying with {chunk_size // 2}.")
                    torch.cuda.empty_cache()
                    chunk_size //= 2
                else: raise
        raise RuntimeError("GPU rendering failed even with minimal chunk size.")

    def _calculate_fitness_gpu(self, rendered_batch_darkness: torch.Tensor) -> List[float]:
        batch_size = rendered_batch_darkness.size(0)
        rendered_light = 1.0 - rendered_batch_darkness
        perceptual = torch.zeros(batch_size, device=self.device)
        if self.lpips_model is not None:
            rendered_lpips, target_lpips = rendered_light.repeat(1,3,1,1) * 2 - 1, self.target_gpu.repeat(1,3,1,1) * 2 - 1
            with torch.no_grad(): perceptual = 1.0 - self.lpips_model(rendered_lpips, target_lpips, normalize=True).squeeze()
        else:
            weighted_mse = torch.mean(self.saliency_gpu * (self.target_gpu - rendered_light) ** 2, dim=[1, 2, 3])
            perceptual = 1.0 - weighted_mse
        structural = torch.zeros(batch_size, device=self.device)
        if hasattr(self, 'dt_map_gpu'):
            line_mask = (rendered_batch_darkness > 0.01).float()
            penalty = self.saliency_gpu * self.dt_map_gpu * line_mask
            num_pixels = torch.sum(line_mask, dim=[1,2,3]).clamp(min=1.0)
            structural = 1.0 - (torch.sum(penalty, dim=[1,2,3]) / num_pixels)
        weight = self.config['contour_weight'] / 100.0
        return ((1.0 - weight) * perceptual + weight * structural).cpu().tolist()

    def calculate_fitness_scores(self, chromosomes: List[List[int]]) -> List[float]:
        if not chromosomes: return []
        if not (self.device and self.line_pixels_gpu_map):
            return [self.calculate_single_fitness(c) for c in chromosomes]
        try:
            return self._calculate_fitness_gpu(self._render_chromosome_batch_gpu(chromosomes))
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logging.warning(f"GPU error ({e}), falling back to CPU for fitness batch.")
            return [self.calculate_single_fitness(c) for c in chromosomes]

    def calculate_batch_fitness(self, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not population: return []
        fitness_scores = self.calculate_fitness_scores([ind['chromosome'] for ind in population])
        for i, ind in enumerate(population): ind['fitness'] = fitness_scores[i]
        population.sort(key=lambda x: x['fitness'], reverse=True)
        return population

    def calculate_single_fitness(self, chromosome: List[int]) -> float:
        rendered_np = self.canvas_manager.render_chromosome_to_numpy(chromosome)
        perceptual = 0.0
        if ssim_cpu:
            perceptual, _ = ssim_cpu(self.target_2d, rendered_np, data_range=255, full=True, K1=0.01, K2=0.03)
        else:
            mse = np.mean(self.saliency_map * ((self.target_2d.astype(np.float32) - rendered_np.astype(np.float32)) ** 2))
            perceptual = 1.0 - (mse / 255.0**2)
        structural = 0.0
        if self.distance_transform_map is not None:
            line_mask = (rendered_np < 250).astype(np.float32)
            penalty = np.sum(self.saliency_map * self.distance_transform_map * line_mask)
            num_pixels = np.sum(line_mask)
            structural = 1.0 - (penalty / num_pixels) if num_pixels > 0 else 1.0
        weight = self.config['contour_weight'] / 100.0
        return (1.0 - weight) * perceptual + weight * structural

    def calculate_single_fitness_and_map(self, chromosome: List[int]) -> Tuple[float, Optional[np.ndarray]]:
        rendered_np = self.canvas_manager.render_chromosome_to_numpy(chromosome)
        diff_map = self.target_2d.astype(np.float32) - rendered_np.astype(np.float32)
        return self.calculate_single_fitness(chromosome), diff_map


# ----------------------------------------------------------------------
# РАЗДЕЛ 4: КЛАССЫ ГЕНЕТИЧЕСКОГО АЛГОРИТМА
# ----------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def sum_line_pixels(residual: np.ndarray, indices: np.ndarray) -> float:
    return np.sum(residual.take(indices))

@njit(fastmath=True, cache=True)
def subtract_line(residual: np.ndarray, indices: np.ndarray, value: float):
    residual[indices] = np.maximum(0.0, residual[indices] - value)

class GreedyInitializer:
    def __init__(self, canvas_manager: CanvasManager, target_flat: np.ndarray, radon_guide: Optional[RadonGuide] = None):
        self.canvas_manager, self.target_flat = canvas_manager, target_flat.astype(np.float32)
        self.num_pins, self.radon_guide = canvas_manager.num_pins, radon_guide

    def run(self, num_lines: int, update_callback: Callable[[str, List[int]], None]) -> List[int]:
        if self.num_pins == 0: return []
        path = [np.random.randint(0, self.num_pins)]
        current_pin = path[0]
        residual = self.target_flat.copy()
        for i in range(num_lines - 1):
            best_score, best_next = -1e12, -1
            candidate_pins = self.radon_guide.get_weighted_candidates(current_pin, self.num_pins // 4) if self.radon_guide else range(self.num_pins)
            for next_pin in candidate_pins:
                if next_pin == current_pin or (len(path) > 1 and next_pin == path[-2]): continue
                line_pixels = self.canvas_manager.get_line_pixel_indices(current_pin, next_pin)
                if len(line_pixels) == 0: continue
                score = sum_line_pixels(residual, line_pixels)
                if score > best_score: best_score, best_next = score, next_pin
            if best_next == -1: break
            best_pixels = self.canvas_manager.get_line_pixel_indices(current_pin, best_next)
            if len(best_pixels) > 0: subtract_line(residual, best_pixels, GREEDY_SUBTRACTION_VALUE)
            path.append(best_next)
            current_pin = best_next
            if i % 10 == 0:
                update_callback(f"Набросок... {int(((i + 1) / num_lines) * 100)}%", path)
        update_callback("Набросок... 100%", path)
        return path


class IslandThread(threading.Thread):
    def __init__(self, island_id: int, config: Dict[str, Any], fitness_calculator: FitnessCalculator,
                 queues: Dict[str, queue.Queue], base_chromosome: List[int], radon_guide: Optional[RadonGuide] = None):
        super().__init__()
        self.island_id, self.config, self.fitness_calculator = island_id, config, fitness_calculator
        self.queues, self.base_chromosome, self.radon_guide = queues, base_chromosome, radon_guide
        self.status_queue, self.stop_event, self.pause_event = queues['status_queue'], threading.Event(), threading.Event()
        self.pause_event.set()
        self.population: List[Dict[str, Any]] = []
        self.best_on_island: Optional[Dict[str, Any]] = None
        self.migrant_queue: queue.Queue = queue.Queue()
        self.mutation_operators = {"reverse": self._reverse, "swap": self._swap, "insert": self._insert,
                                   "scramble": self._scramble, "radon_guided_swap": self._radon_swap}
        self.mutation_stats = {name: {"success": 1, "attempts": 1} for name in self.mutation_operators.keys()}

    def stop(self): self.stop_event.set()
    def pause(self): self.pause_event.clear()
    def resume(self): self.pause_event.set()
    def get_best_individual(self) -> Optional[Dict[str, Any]]: return self.best_on_island
    def accept_migrant(self, migrant: Dict[str, Any]): self.migrant_queue.put(migrant)

    def _reverse(self, c, s, e):
        s, e = max(0, min(s, len(c) - 1)), max(s, min(e, len(c)))
        if e - s < 2: return c
        c[s:e] = c[s:e][::-1]
        return c

    def _swap(self, c, s, e):
        s, e = max(0, min(s, len(c) - 1)), max(s, min(e, len(c)))
        if e - s < 2: return c
        i1, i2 = random.sample(range(s, e), 2)
        c[i1], c[i2] = c[i2], c[i1]
        return c

    def _insert(self, c, s, e):
        s, e = max(0, min(s, len(c) - 1)), max(s, min(e, len(c)))
        if e - s < 2: return c
        p = c.pop(random.randint(s, e - 1))
        c.insert(random.randint(s, e - 1), p)
        return c

    def _scramble(self, c, s, e):
        s, e = max(0, min(s, len(c) - 1)), max(s, min(e, len(c)))
        if e - s < 2: return c
        seg = c[s:e]
        random.shuffle(seg)
        c[s:e] = seg
        return c

    def _radon_swap(self, c, s, e):
        s, e = max(0, min(s, len(c) - 1)), max(s, min(e, len(c)))
        if e - s < 1 or not self.radon_guide or not self.radon_guide.line_weights:
            return c
        
        idx_to_replace = random.randint(s, e - 1)
        ref_pin = c[idx_to_replace - 1] if idx_to_replace > 0 else c[0]
        candidates = self.radon_guide.get_weighted_candidates(ref_pin, 10)
        
        if candidates:
            c[idx_to_replace] = random.choice(candidates)
            
        return c

    def run(self):
        try:
            self._initialize_population()
            lines_per_stage = self.config['lines'] // self.config['stages']
            for stage in range(self.config['stages']):
                if self.stop_event.is_set(): break
                start, end = stage * lines_per_stage, min((stage + 1) * lines_per_stage, self.config['lines'])
                self._run_stage(stage, start, end)
        except Exception:
            logging.error(f"Critical error on island {self.island_id}", exc_info=True)
            self.status_queue.put({'type': 'error', 'data': f"Critical error on island {self.island_id}"})

    def _initialize_population(self):
        logging.info(f"Island {self.island_id}: Initializing diverse population...")
        base_fitness, _ = self.fitness_calculator.calculate_single_fitness_and_map(self.base_chromosome)
        self.population = [{'chromosome': list(self.base_chromosome), 'fitness': base_fitness}]
        chromos_to_create, num_genes = [], len(self.base_chromosome)
        for _ in range(self.config['population_size'] - 1):
            mutated_chromo = list(self.base_chromosome)
            for _ in range(max(1, int(num_genes * 0.02))):
                idx1, idx2 = random.sample(range(num_genes), 2)
                mutated_chromo[idx1], mutated_chromo[idx2] = mutated_chromo[idx2], mutated_chromo[idx1]
            chromos_to_create.append({'chromosome': mutated_chromo})
        if chromos_to_create:
            evaluated_chromos = self.fitness_calculator.calculate_batch_fitness(chromos_to_create)
            self.population.extend(evaluated_chromos)
        self.population.sort(key=lambda x: x.get('fitness', -1), reverse=True)
        self.best_on_island = self.population[0]
        self._send_island_update()
        logging.info(f"Island {self.island_id}: Init complete. Best fitness: {self.best_on_island['fitness']:.5f}")


    def _run_stage(self, stage: int, start_line: int, end_line: int):
        patience, limit = 0, PATIENCE_BASE + (stage * PATIENCE_STAGE_FACTOR)
        last_best_fit = self.best_on_island['fitness'] if self.best_on_island else -1
        while patience < limit:
            self.pause_event.wait()
            if self.stop_event.is_set(): break
            try: self._evolve_one_generation(stage, start_line, end_line, patience, limit)
            except Exception: logging.error(f"Generation error on island {self.island_id}", exc_info=True); patience += 1; continue
            if self.best_on_island and self.best_on_island['fitness'] > last_best_fit + 1e-6:
                last_best_fit = self.best_on_island['fitness']; patience = 0
            else: patience += 1
            
    def _calculate_population_diversity(self) -> float:
        if len(self.population) < 2:
            return 0.0
        
        sample_size = min(len(self.population), 20)
        sample = random.sample(self.population, sample_size)
        
        total_distance, num_comparisons = 0, 0
        
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                c1, c2 = sample[i]['chromosome'], sample[j]['chromosome']
                distance = sum(1 for k in range(len(c1)) if c1[k] != c2[k]) / len(c1)
                total_distance += distance
                num_comparisons += 1
        
        return total_distance / num_comparisons if num_comparisons > 0 else 0.0

    def _evolve_one_generation(self, stage, start, end, patience, limit):
        self._send_generation_tick(stage, patience, limit)
        self._process_migrant()

        stagnation = 1.0 + (patience / limit) * STAGNATION_MUTATION_FACTOR
        diversity = self._calculate_population_diversity()
        diversity_factor = 1.0 + (0.5 - min(diversity, 0.5)) * 2.0
        base_mutation_rate = self.config['mutation_rate'] / 100.0
        dynamic_rate = base_mutation_rate * stagnation * diversity_factor
        mut_rate = min(dynamic_rate, 0.9)
        
        self.population = self._create_new_population(start, end, mut_rate)
        if self.config.get('memetic_enabled', False): self._apply_local_search(start, end)
        if self.population and (not self.best_on_island or self.population[0]['fitness'] > self.best_on_island['fitness']):
            self.best_on_island = self.population[0].copy()
            self._send_island_update()

    def _choose_mutation_strategy(self) -> Tuple[str, Callable]:
        if not (self.radon_guide and self.radon_guide.line_weights) and "radon_guided_swap" in self.mutation_stats:
            del self.mutation_stats["radon_guided_swap"]
        valid_ops = [name for name in self.mutation_operators if name in self.mutation_stats]
        weights = [self.mutation_stats[n]["success"] / self.mutation_stats[n]["attempts"] for n in valid_ops]
        chosen = random.choices(valid_ops, weights=weights if sum(weights) > 0 else None, k=1)[0]
        return chosen, self.mutation_operators[chosen]

    def _create_new_population(self, start, end, mut_rate) -> List[Dict[str, Any]]:
        if not self.population: return []
        new_pop_data = self.population[:1]
        chromos_to_eval = []
        while len(new_pop_data) + len(chromos_to_eval) < self.config['population_size']:
            p1, p2 = self._tournament_selection(), self._tournament_selection()
            if p1 is None or p2 is None: continue
            child = self._order_crossover(p1['chromosome'], p2['chromosome'], start, end)
            if random.random() < mut_rate:
                mut_name, mut_func = self._choose_mutation_strategy()
                mutated = mut_func(list(child), start, end)
                chromos_to_eval.append({'chromosome': mutated, 'mutation_info': (mut_name, p1['fitness'])})
            else:
                chromos_to_eval.append({'chromosome': child})
        
        if chromos_to_eval:
            evaluated_pop = self.fitness_calculator.calculate_batch_fitness(chromos_to_eval)
            new_pop_data.extend(evaluated_pop)

        for ind_data in new_pop_data:
            if 'mutation_info' in ind_data:
                mut_name, parent_fitness = ind_data.pop('mutation_info')
                self.mutation_stats[mut_name]["attempts"] += 1
                if ind_data['fitness'] > parent_fitness: self.mutation_stats[mut_name]["success"] += 1
        new_pop_data.sort(key=lambda x: x.get('fitness', -1), reverse=True)
        return new_pop_data

    def _tournament_selection(self) -> Optional[Dict[str, Any]]:
        if not self.population: return None
        return max(random.sample(self.population, min(5, len(self.population))), key=lambda x: x.get('fitness', -1))

    def _order_crossover(self, p1: List[int], p2: List[int], start: int, end: int) -> List[int]:
        if start >= end -1: return list(p1)
        child = list(p1)
        i, j = sorted(random.sample(range(start, end), 2))
        segment, segment_set = p1[i:j], set(p1[i:j])
        genes_from_p2 = [g for g in p2 if g not in segment_set]
        p2_idx = 0
        for idx in range(len(child)):
            if not (i <= idx < j):
                if p2_idx < len(genes_from_p2): child[idx] = genes_from_p2[p2_idx]; p2_idx += 1
        return child

    def _apply_local_search(self, start, end):
        if not self.population: return
        num_to_improve = int(len(self.population) * (self.config.get('memetic_intensity', 10) / 100.0))
        for i in range(min(num_to_improve, len(self.population))):
            ind = self.population[i]
            improved = self._local_search(ind['chromosome'], start, end)
            if improved is not ind['chromosome']:
                ind['chromosome'] = improved
                ind['fitness'] = self.fitness_calculator.calculate_single_fitness(improved)
        self.population.sort(key=lambda x: x.get('fitness', -1), reverse=True)

    def _local_search(self, chromo, start, end):
        best_c, best_f = list(chromo), self.fitness_calculator.calculate_single_fitness(chromo)
        for _ in range(self.config.get('memetic_depth', 20)):
            if end - start < 2: continue
            temp_c = list(best_c)
            idx = random.randint(start, end - 2)
            temp_c[idx], temp_c[idx + 1] = temp_c[idx + 1], temp_c[idx]
            new_f = self.fitness_calculator.calculate_single_fitness(temp_c)
            if new_f > best_f: best_f, best_c = new_f, temp_c
        return best_c

    def _process_migrant(self):
        try:
            migrant = self.migrant_queue.get_nowait()
            if not self.population or migrant['fitness'] > self.population[-1]['fitness']:
                self.population[-1] = migrant
                self.population.sort(key=lambda x: x.get('fitness', -1), reverse=True)
        except (queue.Empty, IndexError): pass

    def _send_island_update(self):
        if self.best_on_island:
            update_data = self.best_on_island.copy()
            update_data['id'] = self.island_id
            self.status_queue.put({'type': 'island_update', 'data': update_data})

    def _send_generation_tick(self, s, p, l): self.status_queue.put({'type': 'generation_tick', 'data': (self.island_id, s, p, l)})


class IslandManagerThread(threading.Thread):
    def __init__(self, config: Dict[str, Any], queues: Dict[str, queue.Queue],
                 fitness_calculator: FitnessCalculator, target_image_np: np.ndarray, 
                 radon_guide: Optional[RadonGuide] = None):
        super().__init__()
        self.config, self.queues = config, queues
        self.fitness_calculator = fitness_calculator
        self.target_image_np = target_image_np
        self.radon_guide = radon_guide
        self.islands: List[IslandThread] = []
        self.stop_event = threading.Event()
        self.status_queue = queues['status_queue']

    def stop(self): self.stop_event.set(); [i.stop() for i in self.islands]
    def pause(self): [i.pause() for i in self.islands]
    def resume(self): [i.resume() for i in self.islands]

    def _initializer_worker(self, island_id: int, results_dict: Dict[int, List[int]]):
        """Worker function to run GreedyInitializer for one island."""
        try:
            logging.info(f"Initializer-{island_id}: Starting independent greedy initialization...")
            target_flat = (255 - self.target_image_np).flatten()
            greedy = GreedyInitializer(self.fitness_calculator.canvas_manager, target_flat, self.radon_guide)

            def greedy_callback(status: str, path: List[int]):
                self.queues['status_queue'].put({'type': 'greedy_update_island', 'data': {'id': island_id, 'text': status, 'path': path}})

            initial_chromosome = greedy.run(self.config['lines'], greedy_callback)
            results_dict[island_id] = initial_chromosome
            logging.info(f"Initializer-{island_id}: Independent initialization finished.")
        except Exception:
            logging.error(f"Error in initializer worker {island_id}", exc_info=True)
            results_dict[island_id] = [] # Indicate failure

    def run(self):
        try:
            self.status_queue.put({'type': 'status', 'data': "Независимая инициализация островов..."})

            initial_chromosomes: Dict[int, List[int]] = {}
            initializer_threads = []
            for i in range(self.config['num_islands']):
                thread = threading.Thread(target=self._initializer_worker, args=(i, initial_chromosomes), name=f"Initializer-{i}")
                initializer_threads.append(thread)
                thread.start()

            for thread in initializer_threads:
                thread.join()

            if len(initial_chromosomes) != self.config['num_islands'] or any(not path for path in initial_chromosomes.values()):
                raise RuntimeError("Не все острова смогли инициализироваться.")

            self.status_queue.put({'type': 'status', 'data': "Запуск эволюции..."})
            
            self.islands = [IslandThread(i, self.config, self.fitness_calculator, self.queues, initial_chromosomes[i], self.radon_guide) 
                            for i in range(self.config['num_islands'])]
            
            for island in self.islands:
                island.start()
            
            gen_counter = 0
            while any(i.is_alive() for i in self.islands):
                if self.stop_event.is_set(): break
                time.sleep(0.1); gen_counter += 1
                if gen_counter % self.config.get('migration_interval', 20) == 0: self._perform_migration()
            
            for island in self.islands:
                island.join(timeout=1.0)
                
            msg = 'Процесс остановлен.' if self.stop_event.is_set() else 'Генерация завершена!'
            self.status_queue.put({'type': 'done', 'data': msg})
        except Exception as e:
            logging.error("Error in manager thread", exc_info=True)
            self.status_queue.put({'type': 'error', 'data': f"Ошибка в управляющем потоке: {e}"})
    
    def _perform_migration(self):
        self.status_queue.put({'type': 'migration'})
        migrants = [i.get_best_individual() for i in self.islands]
        if all(m is not None for m in migrants):
            for i, island in enumerate(self.islands):
                island.accept_migrant(migrants[(i + 1) % len(self.islands)])


# ----------------------------------------------------------------------
# РАЗДЕЛ 5: КЛАССЫ УПРАВЛЕНИЯ ИНТЕРФЕЙСОМ И СОСТОЯНИЕМ
# ----------------------------------------------------------------------
class AppState:
    def __init__(self):
        self.vars: Dict[str, tk.Variable] = {
            'lines': tk.IntVar(value=3000), 'population_size': tk.IntVar(value=50),
            'stages': tk.IntVar(value=6), 'mutation_rate': tk.IntVar(value=15),
            'num_islands': tk.IntVar(value=max(2, os.cpu_count() or 4)),
            'migration_interval': tk.IntVar(value=25), 'pins': tk.IntVar(value=256),
            'thread_thickness': tk.IntVar(value=1), 'canvas_shape': tk.StringVar(value='Круг'),
            'contour_weight': tk.IntVar(value=30),
            'memetic_enabled': tk.BooleanVar(value=True), 'memetic_intensity': tk.IntVar(value=15),
            'memetic_depth': tk.IntVar(value=30), 'radon_enabled': tk.BooleanVar(value=True),
            'radon_influence': tk.IntVar(value=50),
        }
        self.original_image: Optional[Image.Image] = None
        self.original_image_path: Optional[str] = None
        self.target_image_tk: Optional[Any] = None
        self.overall_best_chromosome: Optional[List[int]] = None
        self.overall_best_fitness: float = -1.0
        self.is_running: bool = False; self.is_paused: bool = False
        self.start_time: float = 0; self.migration_count: int = 0
        self.generation_counters: Dict[int, int] = {}
        self.chart_data_by_island: Dict[int, List[Tuple[int, float]]] = {}
        self.island_best_fitnesses: Dict[int, float] = {}
        self.island_progress_data: Dict[int, Any] = {}

    def get_config(self) -> Dict[str, Any]: return {k: v.get() for k, v in self.vars.items()}
    
    def save_config(self, filepath: str) -> bool:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.get_config(), f, indent=4)
            return True
        except Exception:
            logging.error(f"Failed to save config to {filepath}", exc_info=True)
            return False

    def load_config(self, filepath: str) -> bool:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            for key, value in config.items():
                if key in self.vars:
                    self.vars[key].set(value)
            return True
        except Exception:
            logging.error(f"Failed to load config from {filepath}", exc_info=True)
            return False

    def reset_for_new_run(self):
        self.overall_best_chromosome, self.overall_best_fitness = None, -1.0
        self.migration_count, self.is_paused, self.start_time = 0, False, time.time()
        num_islands = self.vars['num_islands'].get()
        self.chart_data_by_island = {i: [] for i in range(num_islands)}
        self.island_best_fitnesses = {i: -1.0 for i in range(num_islands)}
        self.generation_counters = {i: 0 for i in range(num_islands)}
        self.island_progress_data = {i: (0, 1) for i in range(num_islands)}


class ToolTip:
    def __init__(self, widget, text):
        self.widget, self.text, self.tipwindow = widget, text, None
        self.widget.bind("<Enter>", self.showtip)
        self.widget.bind("<Leave>", self.hidetip)
    def showtip(self, event=None):
        if self.tipwindow or not self.text: return
        x, y, _, _ = self.widget.bbox("insert")
        x, y = x + self.widget.winfo_rootx() + 25, y + self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True); tw.wm_geometry(f"+{int(x)}+{int(y)}")
        tk.Label(tw, text=self.text, justify=tk.LEFT, bg="#ffffe0", relief=tk.SOLID, borderwidth=1, font=("Segoe UI", 8)).pack(ipadx=4)
    def hidetip(self, event=None):
        if self.tipwindow: self.tipwindow.destroy()
        self.tipwindow = None


class UIManager:
    def __init__(self, root: 'StringArtApp', state: AppState):
        self.root, self.state, self.coordinator = root, state, None
        self.widgets: Dict[str, tk.Widget] = {}
        self.result_canvases: Dict[Any, tk.Canvas] = {}
        self.result_photo_refs: Dict[Any, Any] = {}
        self.main_island_progress_bars: Dict[int, ttk.Progressbar] = {}
        self.main_island_labels: Dict[int, ttk.Label] = {}
        self.chart_needs_update: bool = False
        self.ax: Optional[plt.Axes] = None
        self.chart_canvas: Optional[FigureCanvasTkAgg] = None

    def create_widgets(self):
        self.root.configure(bg=self.root.theme['bg_primary'])
        main_pane = ttk.PanedWindow(self.root, orient=tk.VERTICAL, style='Custom.Vertical.TPanedwindow')
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))
        
        top_area = ttk.PanedWindow(main_pane, orient=tk.HORIZONTAL, style='Custom.Horizontal.TPanedwindow')
        main_pane.add(top_area, weight=1)

        self._create_control_panel(top_area)
        self._create_display_area(top_area)
        
        progress_frame = ttk.LabelFrame(main_pane, text=" Прогресс Эволюции ", style='Modern.TLabelframe')
        main_pane.add(progress_frame, weight=1)
        self._create_stats_and_chart_area(progress_frame)

    def _create_control_panel(self, parent: ttk.PanedWindow):
        container = ttk.Frame(parent, width=400, style='Controls.TFrame')
        parent.add(container, weight=0)
        
        left = ttk.Frame(container, style='Controls.TFrame')
        left.pack(side=tk.LEFT, fill=tk.Y, anchor='n', padx=(10, 5), pady=10, expand=True)
        right = ttk.Frame(container, style='Controls.TFrame')
        right.pack(side=tk.LEFT, fill=tk.Y, anchor='n', padx=(5, 10), pady=10, expand=True)

        self._create_left_column(left)
        self._create_right_column(right)

    def _create_left_column(self, parent: tk.Widget):
        frame1 = self._create_section(parent, "1. Холст и Изображение")
        
        btn = ttk.Button(frame1, text="Загрузить изображение", command=self.root.load_image, style='Accent.TButton')
        btn.pack(fill=tk.X, pady=(0, 10)); ToolTip(btn, TOOLTIPS['load_image'])
        self.widgets['load_image_button'] = btn

        f_shape = ttk.Frame(frame1, style='Controls.TFrame'); f_shape.pack(fill=tk.X, pady=4)
        ttk.Label(f_shape, text="Форма холста:", width=15, style='Controls.TLabel').pack(side=tk.LEFT)
        combo = ttk.Combobox(f_shape, textvariable=self.state.vars['canvas_shape'], values=['Круг', 'Квадрат'], state='readonly')
        combo.pack(fill=tk.X, expand=True); ToolTip(combo, "Выберите форму, по которой будут располагаться гвозди.")
        
        self._create_slider_row(frame1, 'pins', "Гвозди", (100, 800))
        self._create_slider_row(frame1, 'lines', "Нити", (500, 8000))
        self._create_slider_row(frame1, 'thread_thickness', "Толщина нити", (1, 5))

        frame2 = self._create_section(parent, "2. Параметры Эволюции")
        self._create_slider_row(frame2, 'num_islands', "Острова", (1, max(1, os.cpu_count() or 1) * 2))
        self._create_slider_row(frame2, 'population_size', "Популяция", (10, 200))
        self._create_slider_row(frame2, 'stages', "Этапы", (2, 25))
        self._create_slider_row(frame2, 'migration_interval', "Миграция", (5, 50))

    def _create_right_column(self, parent: tk.Widget):
        frame3 = self._create_section(parent, "3. Продвинутая оптимизация")
        
        self._create_slider_row(frame3, 'contour_weight', "Вес структуры (%)", (0, 100))
        self._create_slider_row(frame3, 'mutation_rate', "Мутация (%)", (0, 100))
        
        ttk.Separator(frame3, orient='horizontal').pack(fill='x', pady=15)
        
        check_mem = ttk.Checkbutton(frame3, text="Локальный поиск (меметика)", variable=self.state.vars['memetic_enabled'], style='Modern.TCheckbutton')
        check_mem.pack(anchor='w', pady=(0, 5)); ToolTip(check_mem, TOOLTIPS['memetic_enabled'])
        self._create_slider_row(frame3, 'memetic_intensity', "Интенсивность (%)", (1, 50))
        self._create_slider_row(frame3, 'memetic_depth', "Глубина", (5, 100))

        ttk.Separator(frame3, orient='horizontal').pack(fill='x', pady=15)

        check_radon = ttk.Checkbutton(frame3, text="Помощь Радона (гибрид)", variable=self.state.vars['radon_enabled'], style='Modern.TCheckbutton')
        if not SKIMAGE_AVAILABLE: check_radon.config(state=tk.DISABLED)
        check_radon.pack(anchor='w', pady=(0, 5)); ToolTip(check_radon, TOOLTIPS['radon_enabled'])
        self._create_slider_row(frame3, 'radon_influence', "Влияние Радона (%)", (0, 100))

        frame4 = self._create_section(parent, "4. Управление")
        btn_frame = ttk.Frame(frame4, style='Controls.TFrame'); btn_frame.pack(fill=tk.X, pady=5)
        btns = [('start', "▶ Запуск", self.coordinator.start_generation, 'Accent.TButton'),
                ('pause', "❚❚ Пауза", self.coordinator.toggle_pause, 'TButton'),
                ('stop', "■ Стоп", self.coordinator.stop_generation, 'TButton')]
        for n, t, c, s in btns:
            btn = ttk.Button(btn_frame, text=t, command=c, style=s, state=tk.DISABLED if n != 'start' else tk.NORMAL)
            btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=3); ToolTip(btn, TOOLTIPS[n])
            self.widgets[f'{n}_button'] = btn
        
        ttk.Separator(frame4, orient='horizontal').pack(fill='x', pady=15)
        
        export_frame = ttk.Frame(frame4, style='Controls.TFrame')
        export_frame.pack(fill=tk.X, pady=3)
        
        exp_btns = [('export_txt', ".txt", self.coordinator.export_txt),
                    ('export_png', ".png", self.coordinator.export_png),
                    ('export_svg', ".svg", self.coordinator.export_svg)]
        for n, t, c in exp_btns:
            btn = ttk.Button(export_frame, text=t, command=c, state=tk.DISABLED, style='Secondary.TButton')
            btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
            self.widgets[f'{n}_button'] = btn
            ToolTip(btn, TOOLTIPS.get(n, ""))

    def _create_display_area(self, parent: ttk.PanedWindow):
        display_area = ttk.Frame(parent, style='Controls.TFrame')
        parent.add(display_area, weight=1)
        
        target_frame = ttk.LabelFrame(display_area, text=" Исходное изображение ", style='Modern.TLabelframe')
        target_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5), pady=10)
        canvas = tk.Canvas(target_frame, bg=self.root.theme['bg_secondary'], relief='flat', highlightthickness=1, highlightbackground=self.root.theme['border'])
        canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        canvas.bind("<Configure>", self.root.redraw_target_canvas)
        self.widgets['target_canvas'] = canvas
        
        results_area = ttk.LabelFrame(display_area, text=" Результаты ", style='Modern.TLabelframe')
        results_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=10)
        results_notebook = ttk.Notebook(results_area, style='Modern.TNotebook')
        results_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.widgets['results_notebook'] = results_notebook

    def _create_section(self, p, title): return ttk.LabelFrame(p, text=f" {title} ", style='Modern.TLabelframe').pack(fill=tk.X, pady=(0,15), ipadx=5, ipady=5) or p.winfo_children()[-1]

    def _create_stats_and_chart_area(self, p):
        p.configure(style='Controls.TFrame')
        bottom = ttk.Frame(p, style='Controls.TFrame'); bottom.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5,10))
        
        left = ttk.Frame(bottom, style='Controls.TFrame'); left.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        self.widgets['status_label'] = ttk.Label(left, text="Ожидание...", style='Status.TLabel'); self.widgets['status_label'].pack(anchor='w')
        self.widgets['fitness_label'] = ttk.Label(left, text="", style='Secondary.TLabel'); self.widgets['fitness_label'].pack(anchor='w', pady=(0, 5))
        self.widgets['compute_mode_label'] = ttk.Label(left, text="Режим: --", font=self.root.small_font); self.widgets['compute_mode_label'].pack(anchor='w', pady=(0, 15))
        for k, t in [('time', "Время:"), ('eta', "ETA:"), ('gens', "Поколений:"), ('mig', "Миграций:"), ('len', "Длина нити:")]:
            self.widgets[k + '_label'] = self._create_stat_row(left, t)
        
        self.widgets['main_progress_bars_frame'] = ttk.Frame(left, style='Controls.TFrame'); self.widgets['main_progress_bars_frame'].pack(fill=tk.X, expand=True, pady=(15,0))
        
        chart_frame = ttk.Frame(bottom, style='Controls.TFrame')
        chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        fig = Figure(facecolor=self.root.theme['bg_secondary'], dpi=100)
        self.ax = fig.add_subplot(111)
        self.chart_canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        self.chart_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def _create_slider_row(self, p, name, text, limits):
        f = ttk.Frame(p, style='Controls.TFrame'); f.pack(fill=tk.X, pady=4)
        ttk.Label(f, text=f"{text}:", width=15, style='Controls.TLabel').pack(side=tk.LEFT)
        var = self.state.vars[name]
        entry = ttk.Entry(f, textvariable=var, width=5, font=self.root.base_font); entry.pack(side=tk.RIGHT)
        scale = ttk.Scale(f, from_=limits[0], to=limits[1], variable=var, orient=tk.HORIZONTAL, command=lambda v: var.set(int(float(v))))
        scale.pack(fill=tk.X, expand=True, padx=5); ToolTip(entry, TOOLTIPS.get(name, "")); ToolTip(scale, TOOLTIPS.get(name, ""))

    def _create_stat_row(self, p, text):
        f = ttk.Frame(p, style='Controls.TFrame'); f.pack(fill='x'); ttk.Label(f, text=text, font=self.root.bold_font, width=12).pack(side=tk.LEFT)
        lbl = ttk.Label(f, text="--", font=self.root.base_font); lbl.pack(side=tk.LEFT); return lbl

    def on_result_canvas_resize(self, event, key):
        canvas = self.result_canvases.get(key)
        if not (canvas and canvas.winfo_width() > 10 and self.coordinator): return
        if chromosome := self.coordinator.best_chromosomes_by_key.get(key):
            render_data = (chromosome, self.state.get_config()['pins'], self.state.vars['thread_thickness'].get())
            self.coordinator.request_canvas_update(key, render_data, {})

    def setup_for_new_run(self, num_islands):
        self.setup_notebook_tabs(num_islands); self.setup_progress_bars(num_islands)
        self.chart_needs_update = True; self.update_chart()

    def setup_notebook_tabs(self, num_islands):
        nb = self.widgets['results_notebook']
        for i in nb.tabs(): nb.forget(i)
        self.result_canvases.clear(); self.result_photo_refs.clear()
        self._create_notebook_tab('overall', "Лучший", nb)
        for i in range(num_islands): self._create_notebook_tab(i, f"Остров {i+1}", nb)
        self._create_analysis_canvas_tab(nb, 'diff_map', "Расхождения")
        self._create_analysis_canvas_tab(nb, 'fit_saliency', "Значимость")
        self._create_analysis_canvas_tab(nb, 'radon_sinogram', "Синограмма")
        if self.state.overall_best_chromosome: self.root.after(100, lambda: self.on_result_canvas_resize(None, 'overall'))

    def _create_analysis_canvas_tab(self, nb, key, title):
        frame = ttk.Frame(nb); nb.add(frame, text=title)
        canvas = tk.Canvas(frame, bg='black', highlightthickness=0); canvas.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        self.result_canvases[key] = canvas

    def _create_notebook_tab(self, key, text, nb):
        tab = ttk.Frame(nb, style='Modern.TNotebook.Tab'); nb.add(tab, text=text)
        canvas = tk.Canvas(tab, bg=self.root.theme['bg_secondary'], highlightthickness=0); canvas.pack(fill=tk.BOTH, expand=True)
        canvas.bind("<Configure>", lambda e, k=key: self.on_result_canvas_resize(e, k))
        self.result_canvases[key] = canvas

    def setup_progress_bars(self, num_islands):
        pb_frame = self.widgets['main_progress_bars_frame']
        for w in pb_frame.winfo_children(): w.destroy()
        self.main_island_progress_bars.clear(); self.main_island_labels.clear()
        for i in range(num_islands):
            f = ttk.Frame(pb_frame, style='Controls.TFrame'); f.pack(fill=tk.X, pady=2)
            lbl = ttk.Label(f, text="Ожидание...", width=20, font=self.root.small_font, style='Secondary.TLabel'); lbl.pack(side=tk.LEFT, padx=(0, 5))
            pb = ttk.Progressbar(f, orient='horizontal', mode='determinate'); pb.pack(fill=tk.X, expand=True)
            self.main_island_labels[i], self.main_island_progress_bars[i] = lbl, pb

    def update_chart(self):
        if not self.ax or not self.chart_canvas: return
        self.ax.clear()
        num_islands = self.state.get_config()['num_islands']
        colors = plt.cm.hsv(np.linspace(0, 1, num_islands + 1))[:-1] if num_islands > 0 else []
        for i in range(num_islands):
            if data := self.state.chart_data_by_island.get(i, []):
                self.ax.plot(*zip(*data), label=f'Остров {i + 1}', color=colors[i])

        self.ax.set_title("Эволюция фитнеса", color=self.root.theme['text_primary'], fontdict={'family': 'Segoe UI', 'size': 10})
        self.ax.set_xlabel("Поколение", color=self.root.theme['text_secondary'], fontdict={'family': 'Segoe UI', 'size': 9})
        self.ax.set_ylabel("Фитнес", color=self.root.theme['text_secondary'], fontdict={'family': 'Segoe UI', 'size': 9})
        
        if any(self.state.chart_data_by_island.values()): 
            legend = self.ax.legend(loc='upper left', fontsize=8)
            legend.get_frame().set_facecolor(self.root.theme['bg_secondary'])
            for text in legend.get_texts():
                text.set_color(self.root.theme['text_primary'])

        self.ax.grid(True, linestyle='--', alpha=0.3, color=self.root.theme['border'])
        self.ax.set_facecolor(self.root.theme['bg_secondary'])
        for spine in self.ax.spines.values():
            spine.set_edgecolor(self.root.theme['border'])
        self.ax.tick_params(axis='x', colors=self.root.theme['text_secondary'])
        self.ax.tick_params(axis='y', colors=self.root.theme['text_secondary'])
        
        self.chart_canvas.figure.tight_layout(pad=2.0)
        self.chart_canvas.draw()
        self.chart_needs_update = False

    def toggle_ui_state(self, is_running: bool):
        self.state.is_running = is_running
        state = tk.DISABLED if is_running else tk.NORMAL
        
        self.widgets['start_button'].config(state=state)
        self.widgets['stop_button'].config(state=tk.NORMAL if is_running else tk.DISABLED)
        self.widgets['pause_button'].config(state=tk.NORMAL if is_running else tk.DISABLED)
        
        export_state = tk.NORMAL if not is_running and self.state.overall_best_chromosome else tk.DISABLED
        for key in ['export_txt_button', 'export_png_button', 'export_svg_button']:
            if key in self.widgets: self.widgets[key].config(state=export_state)
            
        if not is_running:
            self.widgets['pause_button'].config(text="❚❚ Пауза")
            for i, pb in self.main_island_progress_bars.items():
                pb['value'] = 0
                if lbl := self.main_island_labels.get(i): lbl.config(text="Завершено")

    def update_island_progress(self, island_id, stage, patience, limit):
        if island_id in self.main_island_labels:
            p_pct = int((patience / limit) * 100) if limit > 0 else 0
            stages = self.state.get_config().get('stages', 1)
            self.main_island_labels[island_id].config(text=f"Этап {stage + 1}/{stages}: {p_pct}%")
        if island_id in self.main_island_progress_bars:
            self.main_island_progress_bars[island_id].config(maximum=limit, value=patience)

    def update_best_fitness_display(self, fitness: float): self.widgets['fitness_label'].config(text=f"Лучший фитнес: {fitness:.6f}")
    def update_thread_length_display(self, length: float): self.widgets['len_label'].config(text=f"{length:.2f} м (для 10см)")

    def update_stats_display(self, state: AppState):
        elapsed = time.time() - state.start_time
        self.widgets['time_label'].config(text=time.strftime('%H:%M:%S', time.gmtime(elapsed)))
        self.widgets['gens_label'].config(text=f"{sum(state.generation_counters.values())}")
        self.widgets['mig_label'].config(text=f"{state.migration_count}")
        total_prog, islands, stages = 0, state.get_config()['num_islands'], state.get_config()['stages']
        if not (islands and stages): return
        for i in range(islands):
            stage, limit = state.island_progress_data.get(i, (0, 1))
            pb = self.main_island_progress_bars.get(i)
            patience = pb['value'] if pb else 0
            prog_per_stage = 1.0 / stages
            stage_prog = (patience / limit if limit > 0 else 0) * prog_per_stage
            total_prog += (stage * prog_per_stage) + stage_prog
        avg_prog = total_prog / islands if islands > 0 else 0
        if avg_prog > 0.01:
            eta = (elapsed / avg_prog) * (1 - avg_prog) if avg_prog < 1 else 0
            self.widgets['eta_label'].config(text=time.strftime('%H:%M:%S', time.gmtime(eta)))
        else: self.widgets['eta_label'].config(text="Расчет...")

    def update_canvas_image(self, key: Any, img_pil: Image.Image):
        canvas = self.result_canvases.get(key)
        if not canvas: return
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 2 or h < 2: return self.root.after(100, lambda: self.update_canvas_image(key, img_pil))
        img_copy = img_pil.copy(); img_copy.thumbnail((w, h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img_copy)
        self.result_photo_refs[key] = photo
        canvas.delete("all"); canvas.create_image(w // 2, h // 2, image=photo)

    def update_title(self):
        title = "String Art Generator v8.5"
        if self.state.original_image_path: title += f" - [{os.path.basename(self.state.original_image_path)}]"
        self.root.title(title)

    def clear_analysis_tabs(self):
        for key in ['diff_map', 'fit_saliency', 'radon_sinogram']:
            if canvas := self.result_canvases.get(key):
                if canvas.winfo_width() > 1 and canvas.winfo_height() > 1:
                    black = Image.new('RGB', (canvas.winfo_width(), canvas.winfo_height()), 'black')
                    self.root.after(10, lambda k=key, i=black: self.update_canvas_image(k, i))


class EvolutionCoordinator:
    def __init__(self, root: 'StringArtApp', state: AppState, ui: UIManager, queues: Dict[str, queue.Queue]):
        self.root, self.state, self.ui_manager, self.queues = root, state, ui, queues
        self.manager_thread: Optional[IslandManagerThread] = None
        self.renderer_thread: Optional[threading.Thread] = None
        self.render_queue, self.render_results_queue = queue.Queue(), queue.Queue()
        self.best_chromosomes_by_key: Dict[Any, List[int]] = {}
        self.stats_update_job: Optional[str] = None
        self.canvas_manager: Optional[CanvasManager] = None
        self.render_cache: Dict[Tuple, Image.Image] = {}
        self._start_renderer_thread()

    def _start_renderer_thread(self):
        self.renderer_thread = threading.Thread(target=self._renderer_worker, name="RendererThread", daemon=True)
        self.renderer_thread.start()

    def _renderer_worker(self):
        while True:
            try:
                key, data, _ = self.render_queue.get()
                if self.canvas_manager is None: continue
                chromosome, _, thickness = data
                img = self.canvas_manager.render_chromosome_to_pil(chromosome, thickness)
                self.render_cache[(tuple(chromosome), self.canvas_manager.size, thickness)] = img
                self.render_results_queue.put((key, img))
            except Exception: logging.error("Error in renderer thread", exc_info=True)

    def start_generation(self):
        if self.state.original_image is None: return messagebox.showwarning("Внимание", "Загрузите изображение.")
        config = self.state.get_config()
        if not self._validate_config(config): return
        self.state.reset_for_new_run(); self.ui_manager.toggle_ui_state(True)
        self.best_chromosomes_by_key.clear(); self.render_cache.clear()
        self.ui_manager.setup_for_new_run(config['num_islands'])
        threading.Thread(target=self._prepare_and_run_evolution, name="PreparationThread", daemon=True).start()

    def _validate_config(self, config):
        if config['pins'] < 3: return messagebox.showerror("Ошибка", "Нужно минимум 3 гвоздя.") or False
        if config['lines'] < 10: return messagebox.showerror("Ошибка", "Нужно минимум 10 линий.") or False
        if config.get('radon_enabled') and not SKIMAGE_AVAILABLE:
            return messagebox.showerror("Ошибка", "scikit-image не найден. Помощь Радона недоступна.") or False
        return True

    def _prepare_and_run_evolution(self):
        try:
            config = self.state.get_config()
            self._send_status("Подготовка изображения...")
            target_np, edges_np, saliency = prepare_target_image(self.state.original_image, IMAGE_SIZE)
            self.canvas_manager = CanvasManager(IMAGE_SIZE, config['pins'], config['canvas_shape'])
            radon_guide = None
            if config.get('radon_enabled', False):
                self._send_status("Анализ Радона..."); radon_guide = RadonGuide(target_np, self.canvas_manager, config)
                if radon_guide.sinogram is not None: self.queues['status_queue'].put({'type': 'radon_map', 'data': radon_guide.sinogram})
            fc = FitnessCalculator(target_np, edges_np, saliency, self.canvas_manager, config, self.queues)
            self._send_status("Визуализация карт анализа...")
            self.queues['status_queue'].put({'type': 'fit_func_maps', 'data': {'saliency': saliency, 'dt': fc.distance_transform_map}})
            if fc.device: self._prepare_gpu_data(fc)
            self._start_island_manager(config, fc, target_np, radon_guide)
        except Exception:
            logging.error("Preparation thread error", exc_info=True)
            self.queues['status_queue'].put({'type': 'error', 'data': "Критическая ошибка при подготовке."})
            self.ui_manager.toggle_ui_state(False)

    def _prepare_gpu_data(self, fc): self._send_status("Кэширование линий для GPU..."); fc.enable_gpu_rendering(fc.canvas_manager.precompute_line_data_for_gpu(fc.device))
    
    def _start_island_manager(self, config, fc, target_np, radon_guide):
        self.manager_thread = IslandManagerThread(config, self.queues, fc, target_np, radon_guide)
        self.manager_thread.start()
        self.stats_update_job = self.root.after(1000, self._update_progress_stats)

    def stop_generation(self):
        if self.stats_update_job: self.root.after_cancel(self.stats_update_job)
        self.stats_update_job = None
        if self.manager_thread and self.manager_thread.is_alive():
            self.manager_thread.stop(); self._send_status("Остановка...")
        self.root.after(200, self._finalize_run)

    def _finalize_run(self):
        if pruned := self.prune_chromosome():
            messagebox.showinfo("Оптимизация", f"Путь оптимизирован! Удалено линий: {pruned}")
        self.ui_manager.toggle_ui_state(False)

    def toggle_pause(self):
        if not (self.manager_thread and self.manager_thread.is_alive()): return
        self.state.is_paused = not self.state.is_paused
        if self.state.is_paused:
            if self.stats_update_job: self.root.after_cancel(self.stats_update_job); self.stats_update_job = None
            self.manager_thread.pause(); self.ui_manager.widgets['pause_button'].config(text="► Возобновить")
            self._send_status("Пауза...")
        else:
            self.manager_thread.resume(); self.ui_manager.widgets['pause_button'].config(text="❚❚ Пауза")
            self.stats_update_job = self.root.after(1000, self._update_progress_stats)
            
    def process_queues(self):
        try:
            while not self.queues['status_queue'].empty():
                msg = self.queues['status_queue'].get_nowait()
                self._get_message_handler(msg.get('type'))(msg.get('data'))
            while not self.render_results_queue.empty():
                key, img = self.render_results_queue.get_nowait()
                self.ui_manager.update_canvas_image(key, img)
        except queue.Empty: pass
        except Exception as e: logging.error(f"Queue processing error: {e}", exc_info=True)
        finally: self.root.after(100, self.process_queues)

    def _get_message_handler(self, msg_type):
        return { 'greedy_update_island': self._handle_greedy_update_island, 'status': self._handle_status, 'error': self._handle_error,
                 'gpu_status': self._handle_gpu_status, 'generation_tick': self._handle_generation_tick,
                 'island_update': self._handle_island_update, 'done': self._handle_done, 'migration': self._handle_migration,
                 'fit_func_maps': self._handle_fit_func_maps, 'radon_map': self._handle_radon_map,
               }.get(msg_type, lambda data: None)
    
    def _handle_greedy_update_island(self, data: Dict[str, Any]):
        island_id, path, text = data['id'], data['path'], data['text']
        self.ui_manager.widgets['status_label'].config(text=f"Остров {island_id+1}: {text}")
        config = self.state.get_config()
        render_data = (path, config['pins'], config['thread_thickness'])
        self.request_canvas_update(island_id, render_data, {})
        if not self.state.overall_best_chromosome:
             self.request_canvas_update('overall', render_data, {})

    def _handle_status(self, data): self.ui_manager.widgets['status_label'].config(text=data)
    def _handle_error(self, data):
        messagebox.showerror("Ошибка в потоке", data); self.ui_manager.widgets['status_label'].config(text="Ошибка!")
        self.ui_manager.toggle_ui_state(False)
    def _handle_gpu_status(self, data): 
        text = f"Режим: {'GPU (CUDA)' if data else 'CPU'}"
        color = self.root.theme['text_success'] if data else self.root.theme['text_warning']
        self.ui_manager.widgets['compute_mode_label'].config(text=text, foreground=color)
    def _handle_generation_tick(self, data):
        island_id, stage, patience, limit = data
        self.ui_manager.widgets['status_label'].config(text=f"Остров {island_id+1}, Этап {stage+1}/{self.state.get_config()['stages']}...")
        self.state.generation_counters[island_id] = self.state.generation_counters.get(island_id, 0) + 1
        self.state.island_progress_data[island_id] = (stage, limit)
        self.ui_manager.update_island_progress(island_id, stage, patience, limit)
        if (self.state.generation_counters[island_id]) % 10 == 0:
            fit = self.state.island_best_fitnesses.get(island_id, 0)
            self.state.chart_data_by_island.setdefault(island_id, []).append((sum(self.state.generation_counters.values()), fit))
            self.ui_manager.chart_needs_update = True
    def _handle_island_update(self, data):
        i, c, f = data['id'], data['chromosome'], data['fitness']
        self.state.island_best_fitnesses[i], self.best_chromosomes_by_key[i] = f, c
        config = self.state.get_config()
        self.request_canvas_update(i, (c, config['pins'], config['thread_thickness']), {})
        if f > self.state.overall_best_fitness:
            self.state.overall_best_fitness, self.state.overall_best_chromosome = f, list(c)
            self.best_chromosomes_by_key['overall'] = list(c)
            self.ui_manager.update_best_fitness_display(f)
            self.ui_manager.update_thread_length_display(self._calculate_thread_length(c))
            self.request_canvas_update('overall', (c, config['pins'], config['thread_thickness']), {})
            if self.manager_thread and (fc := self.manager_thread.fitness_calculator):
                _, diff_map = fc.calculate_single_fitness_and_map(c)
                if diff_map is not None:
                    self.render_results_queue.put(('diff_map', self._create_diff_pil_image(diff_map)))
    def _handle_fit_func_maps(self, data):
        if 'saliency' in data and data['saliency'] is not None: self.render_results_queue.put(('fit_saliency', Image.fromarray(data['saliency']).convert('L')))
        if 'dt' in data and data['dt'] is not None: self.render_results_queue.put(('fit_dt', Image.fromarray((data['dt'] * 255).astype(np.uint8)).convert('L')))
    def _handle_radon_map(self, sino):
        if sino is None: return
        sino_vis = (255 * (sino - sino.min()) / (sino.max() - sino.min())).astype(np.uint8)
        self.render_results_queue.put(('radon_sinogram', Image.fromarray(sino_vis).convert('L')))
    def _handle_migration(self, _): self.state.migration_count += 1
    def _handle_done(self, data):
        self._finalize_run()
        status = data;
        if pruned := self.prune_chromosome(): status += f" Путь оптимизирован ({pruned} линий удалено)."
        self.ui_manager.widgets['status_label'].config(text=status)

    def _send_status(self, msg): self.queues['status_queue'].put({'type': 'status', 'data': msg})
    def request_canvas_update(self, key, data, params={}):
        if not (self.canvas_manager and self.ui_manager.result_canvases.get(key)): return
        cache_key = (tuple(data[0]), self.canvas_manager.size, data[2])
        if img := self.render_cache.get(cache_key): self.render_results_queue.put((key, img)); return
        self.render_queue.put((key, data, params))

    def _create_diff_pil_image(self, diff_map):
        max_abs = np.max(np.abs(diff_map))
        if max_abs == 0: return Image.new('RGB', diff_map.shape[::-1], 'black')
        fig = plt.figure(figsize=(5, 5), dpi=100); ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(diff_map, cmap='seismic', vmin=-max_abs, vmax=max_abs); ax.axis('off')
        buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0); img = Image.open(buf); plt.close(fig); return img

    def _calculate_thread_length(self, chromo):
        if not (chromo and self.canvas_manager): return 0.0
        coords = self.canvas_manager.pin_coords
        length = sum(np.linalg.norm(coords[chromo[i]] - coords[chromo[i+1]]) for i in range(len(chromo) - 1))
        return (length * (10.0 / self.canvas_manager.size)) / 100.0

    def _update_progress_stats(self):
        if not (self.state.is_running and self.manager_thread and self.manager_thread.is_alive()):
            if self.stats_update_job: self.root.after_cancel(self.stats_update_job); self.stats_update_job = None
            return
        self.ui_manager.update_stats_display(self.state)
        self.stats_update_job = self.root.after(1000, self._update_progress_stats)

    def prune_chromosome(self) -> int:
        if not self.state.overall_best_chromosome: return 0
        chromo = self.state.overall_best_chromosome
        pruned = [chromo[0]]
        i = 1
        while i < len(chromo) - 1:
            if chromo[i - 1] == chromo[i + 1]: i += 2
            else: pruned.append(chromo[i]); i += 1
        pruned.append(chromo[-1])
        removed = len(chromo) - len(pruned)
        if removed > 0:
            self.state.overall_best_chromosome = pruned
            self.best_chromosomes_by_key['overall'] = pruned
            cfg = self.state.get_config()
            self.request_canvas_update('overall', (pruned, cfg['pins'], cfg['thread_thickness']), {})
            self.ui_manager.update_thread_length_display(self._calculate_thread_length(pruned))
        return removed

    def export_txt(self):
        if not self.state.overall_best_chromosome: return
        if not (filepath := filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])): return
        length = self._calculate_thread_length(self.state.overall_best_chromosome)
        info = f"Примерная длина нити для холста 10 см: {length:.2f} м\n"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Последовательность гвоздей для String Art\n"
                    f"Всего гвоздей: {self.state.get_config()['pins']}\n"
                    f"Всего линий: {len(self.state.overall_best_chromosome) - 1}\n{info}\n"
                    f"Последовательность (нумерация с 0):\n" +
                    "\n".join([f"Шаг {i+1}: {p}" for i, p in enumerate(self.state.overall_best_chromosome)]))
        messagebox.showinfo("Экспорт", f"Путь сохранен в {os.path.basename(filepath)}")

    def export_png(self):
        if not self.state.overall_best_chromosome: return
        if not (filepath := filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG images", "*.png")])): return
        self._send_status("Экспорт PNG (2000px)...")
        config = self.state.get_config()
        cm = CanvasManager(2000, config['pins'], config['canvas_shape'])
        img = cm.render_chromosome_to_pil(self.state.overall_best_chromosome, config['thread_thickness'])
        img.save(filepath); self._send_status("Экспорт PNG завершен.")
        messagebox.showinfo("Экспорт", f"Изображение сохранено в {os.path.basename(filepath)}")

    def export_svg(self):
        if not self.state.overall_best_chromosome: return
        if not (filepath := filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG", "*.svg")])): return
        self._send_status("Экспорт SVG...")
        config, size = self.state.get_config(), 1000
        cm = CanvasManager(size, config['pins'], config['canvas_shape'])
        coords = cm.pin_coords
        path_data = "M " + " L ".join([f"{c[0]:.2f},{c[1]:.2f}" for c in [coords[p] for p in self.state.overall_best_chromosome]])
        with open(filepath, 'w') as f:
            f.write(f'<svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">\n'
                    f'  <rect width="100%" height="100%" fill="white"/>\n'
                    f'  <path d="{path_data}" stroke="black" stroke-width="{config["thread_thickness"]}" fill="none"/>\n'
                    '</svg>\n')
        self._send_status("Экспорт SVG завершен.")
        messagebox.showinfo("Экспорт", f"SVG сохранен в {os.path.basename(filepath)}")


# ----------------------------------------------------------------------
# РАЗДЕЛ 6: ГЛАВНЫЙ КЛАСС ПРИЛОЖЕНИЯ И ТОЧКА ВХОДА
# ----------------------------------------------------------------------

class StringArtApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("1500x950")
        
        self.theme = {
            'bg_primary': '#F0F2F5', 'bg_secondary': '#FFFFFF',
            'controls_bg': '#F0F2F5', 'text_primary': '#212121',
            'text_secondary': '#616161', 'accent': '#0078D4',
            'accent_fg': '#FFFFFF', 'border': '#E0E0E0',
            'text_success': '#2E7D32', 'text_warning': '#FF8F00'
        }
        
        self.base_font = font.Font(family="Segoe UI", size=9)
        self.bold_font = font.Font(family="Segoe UI", size=9, weight="bold")
        self.small_font = font.Font(family="Segoe UI", size=8)
        self.title_font = font.Font(family="Segoe UI", size=10, weight="bold")
        
        self._setup_styles()
        
        self.state = AppState()
        self.queues = {'status_queue': queue.Queue()}
        self.ui_manager = UIManager(self, self.state)
        self.coordinator = EvolutionCoordinator(self, self.state, self.ui_manager, self.queues)
        self.ui_manager.coordinator = self.coordinator
        
        self.ui_manager.create_widgets()
        self._load_last_config()
        self.ui_manager.update_title()

        self.after(100, self.coordinator.process_queues)
        self.after(1000, self._periodic_ui_update)
        self.after(500, self._periodic_state_check)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use('clam')

        # --- General ---
        style.configure('.', background=self.theme['bg_primary'], foreground=self.theme['text_primary'], font=self.base_font, bordercolor=self.theme['border'])
        style.configure('TFrame', background=self.theme['bg_primary'])
        style.configure('Controls.TFrame', background=self.theme['controls_bg'])
        
        # --- Labels ---
        style.configure('TLabel', background=self.theme['bg_primary'], foreground=self.theme['text_primary'], font=self.base_font)
        style.configure('Controls.TLabel', background=self.theme['controls_bg'])
        style.configure('Secondary.TLabel', foreground=self.theme['text_secondary'], background=self.theme['controls_bg'])
        style.configure('Status.TLabel', font=self.bold_font, background=self.theme['controls_bg'])

        # --- LabelFrame ---
        style.configure('Modern.TLabelframe', background=self.theme['controls_bg'], bordercolor=self.theme['border'], lightcolor=self.theme['controls_bg'], darkcolor=self.theme['controls_bg'])
        style.configure('Modern.TLabelframe.Label', font=self.title_font, foreground=self.theme['text_primary'], background=self.theme['controls_bg'])

        # --- PanedWindow ---
        style.configure('Custom.Vertical.TPanedwindow', background=self.theme['bg_primary'])
        style.configure('Custom.Horizontal.TPanedwindow', background=self.theme['bg_primary'])

        # --- Buttons ---
        style.configure('TButton', font=self.bold_font, padding=(10, 8), relief='flat', background='#B0BEC5', foreground=self.theme['text_primary'])
        style.map('TButton', background=[('active', '#90A4AE')])
        style.configure('Accent.TButton', background=self.theme['accent'], foreground=self.theme['accent_fg'])
        style.map('Accent.TButton', background=[('active', '#005A9E')])
        style.configure('Secondary.TButton', background=self.theme['bg_secondary'], foreground=self.theme['text_primary'], bordercolor=self.theme['border'])
        style.map('Secondary.TButton', background=[('active', '#E0E0E0')])

        # --- Other ---
        style.configure('TScale', background=self.theme['controls_bg'])
        style.configure('Modern.TCheckbutton', background=self.theme['controls_bg'])
        style.configure('TProgressbar', troughcolor=self.theme['border'], background=self.theme['accent'])
        
        # --- Notebook ---
        style.configure('Modern.TNotebook', background=self.theme['bg_primary'], bordercolor=self.theme['border'])
        style.configure('Modern.TNotebook.Tab', padding=(10, 5), font=self.base_font, background=self.theme['bg_primary'], foreground=self.theme['text_secondary'])
        style.map('Modern.TNotebook.Tab', background=[('selected', self.theme['bg_secondary'])], foreground=[('selected', self.theme['accent'])])

    def _get_config_path(self) -> str:
        home = os.path.expanduser("~")
        config_dir = os.path.join(home, ".string-art-generator")
        os.makedirs(config_dir, exist_ok=True)
        return os.path.join(config_dir, "last_config.json")
        
    def _load_last_config(self):
        config_path = self._get_config_path()
        if os.path.exists(config_path):
            self.state.load_config(config_path)

    def on_closing(self):
        config_path = self._get_config_path()
        self.state.save_config(config_path)
        if self.state.is_running and messagebox.askokcancel("Выход", "Процесс активен. Выйти?"):
            self.coordinator.stop_generation(); self.destroy()
        elif not self.state.is_running: self.destroy()

    def _periodic_ui_update(self):
        if self.ui_manager.chart_needs_update: self.ui_manager.update_chart()
        self.after(1000, self._periodic_ui_update)

    def _periodic_state_check(self):
        try:
            if self.coordinator and self.coordinator.manager_thread:
                is_alive = self.coordinator.manager_thread.is_alive()
                if is_alive != self.state.is_running and not self.coordinator.stop_event.is_set():
                    logging.warning(f"UI state mismatch detected (is_running={self.state.is_running}, is_alive={is_alive}). Correcting.")
                    if self.coordinator.stats_update_job and not is_alive:
                        self.after_cancel(self.coordinator.stats_update_job); self.coordinator.stats_update_job = None
                    self.ui_manager.toggle_ui_state(is_alive)
        except Exception as e: logging.error(f"Error in state check: {e}")
        finally: self.after(500, self._periodic_state_check)

    def load_image(self):
        if not (file_path := filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])): return
        try:
            self.state.original_image = Image.open(file_path)
            self.state.original_image_path = file_path
            self.ui_manager.update_title(); self.redraw_target_canvas()
            self.ui_manager.widgets['status_label'].config(text="Изображение загружено.")
            self.ui_manager.clear_analysis_tabs()
        except Exception:
            logging.error("Failed to load image.", exc_info=True)
            messagebox.showerror("Ошибка", "Не удалось загрузить изображение.")

    def redraw_target_canvas(self, event=None):
        if not self.state.original_image: return
        canvas = self.ui_manager.widgets['target_canvas']
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 10 or h < 10: return
        img_copy = self.state.original_image.copy()
        img_copy.thumbnail((w, h), Image.Resampling.LANCZOS)
        bg = Image.new('RGB', (w, h), self.theme['bg_secondary'])
        bg.paste(img_copy, ((w - img_copy.width) // 2, (h - img_copy.height) // 2))
        self.state.target_image_tk = ImageTk.PhotoImage(bg)
        canvas.delete("all"); canvas.create_image(w // 2, h // 2, image=self.state.target_image_tk)


if __name__ == "__main__":
    try:
        if getattr(sys, 'frozen', False):
            os.environ["OPENCV_SALIENCY_PATH"] = os.path.join(sys._MEIPASS, 'opencv-saliency')
        app = StringArtApp()
        app.mainloop()
    except Exception:
        logging.critical("Fatal error on startup", exc_info=True)
        root = tk.Tk(); root.withdraw()
        messagebox.showerror("Критическая ошибка", f"Не удалось запустить приложение.\n\n{traceback.format_exc()}")

