import numpy as np
import queue
from typing import Dict, Any, List, Optional, Tuple

try:
    from skimage.metrics import structural_similarity as ssim_cpu
except ImportError:
    ssim_cpu = None

try:
    import torch
    import torch.nn.functional as F
    GPU_ENABLED_FLAG = torch.cuda.is_available()
except ImportError:
    torch, F, GPU_ENABLED_FLAG = None, None, False

if False:
    from .canvas import CanvasManager


class FitnessCalculator:
    """
    Управляет вычислением приспособленности (fitness) хромосом.
    Использует CPU или GPU в зависимости от доступности и настроек.
    """
    def __init__(self, target_2d: np.ndarray, canvas_manager: 'CanvasManager',
                 thread_thickness: int, queues: Dict[str, queue.Queue]):
        self.canvas_manager = canvas_manager
        self.queues = queues
        self.line_pixels_gpu_map: Optional[Dict[Tuple[int, int], Any]] = None

        # --- Подготовка целевого изображения ---
        # Алгоритм будет пытаться воспроизвести негатив исходного изображения.
        # Светлые участки оригинала должны стать темными от нитей, и наоборот.
        self.target_2d = target_2d
        self.target_2d_inverted = 255 - target_2d

        # Нормализуем целевое изображение (значения от 0.0 до 1.0) для расчетов.
        self.target_normalized = self.target_2d_inverted.astype(np.float32) / 255.0

        # --- Выбор устройства для вычислений (CPU или GPU) ---
        # ИСПРАВЛЕНО: GPU используется только если толщина нити = 1, т.к. GPU-рендер
        # не поддерживает линии разной толщины.
        if GPU_ENABLED_FLAG and thread_thickness == 1:
            self.device = torch.device("cuda")
            # Переносим целевое изображение в память GPU.
            self.target_gpu = torch.from_numpy(self.target_normalized).unsqueeze(0).unsqueeze(0).to(self.device)
            self.queues['status_queue'].put({'type': 'gpu_status', 'data': True})
        else:
            self.device = None
            self.queues['status_queue'].put({'type': 'gpu_status', 'data': False})

    def enable_gpu_rendering(self, line_pixels_gpu_map: Dict[Tuple[int, int], Any]):
        """Включает GPU-режим, передавая предрассчитанные данные о линиях."""
        if self.device:
            self.line_pixels_gpu_map = line_pixels_gpu_map

    def _render_chromosome_batch_gpu(self, chromosomes: List[List[int]]) -> torch.Tensor:
        """Рендерит пакет хромосом на GPU, используя векторизацию."""
        batch_size = len(chromosomes)
        image_size = self.canvas_manager.size
        batch_image_flat = torch.zeros(batch_size, image_size * image_size, dtype=torch.float32, device=self.device)
        
        for i, chromo in enumerate(chromosomes):
            num_lines = len(chromo) - 1
            if num_lines <= 0: continue
            alpha = min(0.1, 800 / (num_lines + 1))
            for j in range(num_lines):
                key = tuple(sorted((chromo[j], chromo[j+1])))
                if self.line_pixels_gpu_map and key in self.line_pixels_gpu_map:
                    indices = self.line_pixels_gpu_map[key]
                    batch_image_flat[i].index_add_(0, indices, torch.full_like(indices, fill_value=alpha, dtype=torch.float32))
        
        batch_image = batch_image_flat.view(batch_size, 1, image_size, image_size)
        # Инвертируем и нормализуем, чтобы получить изображение (0=черный, 1=белый).
        rendered_normalized = torch.clamp((1.0 - batch_image), 0, 1)
        return rendered_normalized

    def _calculate_fitness_gpu(self, rendered_batch: torch.Tensor) -> List[float]:
        """
        Вычисляет fitness для батча на GPU.
        Используется MSE (Mean Squared Error) — среднеквадратичная ошибка.
        Чем меньше ошибка, тем лучше.
        """
        target_batch = self.target_gpu.expand_as(rendered_batch)
        
        # Считаем MSE.
        mse = torch.mean((target_batch - rendered_batch) ** 2, dim=[1, 2, 3])
        
        # Преобразуем ошибку в "фитнес": 1.0 - MSE.
        # Идеальное совпадение (MSE=0) даст фитнес 1.0.
        fitness = 1.0 - mse
        
        return fitness.cpu().tolist()

    def calculate_batch_fitness(self, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Оценивает всю популяцию. Использует GPU, если возможно, иначе CPU.
        """
        chromosomes = [ind['chromosome'] for ind in population]
        
        if self.device:
            try:
                rendered_batch_gpu = self._render_chromosome_batch_gpu(chromosomes)
                fitness_scores = self._calculate_fitness_gpu(rendered_batch_gpu)
                for i, ind in enumerate(population):
                    ind['fitness'] = fitness_scores[i]
            except torch.cuda.OutOfMemoryError:
                # Если на GPU не хватило памяти, переключаемся на CPU.
                return self._calculate_batch_fitness_cpu(population)
        else:
            return self._calculate_batch_fitness_cpu(population)
            
        population.sort(key=lambda x: x['fitness'], reverse=True)
        return population

    def _calculate_batch_fitness_cpu(self, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Оценивает популяцию на CPU."""
        for ind in population:
            rendered_np = self.canvas_manager.render_chromosome_to_numpy(ind['chromosome'])
            rendered_normalized = (255 - rendered_np).astype(np.float32) / 255.0
            
            mse = np.mean((self.target_normalized - rendered_normalized) ** 2)
            ind['fitness'] = 1.0 - mse
        
        population.sort(key=lambda x: x['fitness'], reverse=True)
        return population

    def calculate_single_fitness_and_map(self, chromosome: List[int]) -> Tuple[float, Optional[np.ndarray]]:
        """
        Вычисляет точный fitness для одной хромосомы на CPU (используя SSIM, если доступно)
        и создает карту расхождений для визуализации.
        """
        rendered_np = self.canvas_manager.render_chromosome_to_numpy(chromosome)
        
        # Карта расхождений = целевое_изображение - сгенерированное
        # Положительные значения: цель светлее, чем рендер (ниток слишком много)
        # Отрицательные значения: цель темнее, чем рендер (ниток не хватает)
        difference_map = self.target_2d.astype(np.float32) - rendered_np.astype(np.float32)
        
        if not ssim_cpu:
            # Если scikit-image не установлена, используем наш MSE-фитнес.
            rendered_normalized = (255 - rendered_np).astype(np.float32) / 255.0
            mse = np.mean((self.target_normalized - rendered_normalized) ** 2)
            fitness = 1.0 - mse
            return fitness, difference_map

        # SSIM (Structural Similarity Index) - более качественная метрика,
        # чем MSE, учитывает структуру изображения.
        score, _ = ssim_cpu(self.target_2d_inverted, (255 - rendered_np), data_range=255, full=True)
        
        return score, difference_map
