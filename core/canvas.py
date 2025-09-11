import numpy as np
import cv2
import math
from PIL import Image
import threading
from typing import Tuple, Dict, Any, List

# Константа, определяющая внутренний размер холста для вычислений.
IMAGE_SIZE = 400

def _calculate_line_pixels_numpy(x1: float, y1: float, x2: float, y2: float, size: int) -> np.ndarray:
    """
    Вычисляет 1D-индексы всех пикселей, лежащих на линии между двумя точками.
    Используется для быстрой отрисовки на GPU.
    """
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0:
        return np.empty(0, dtype=np.int64)

    # Генерируем координаты пикселей вдоль линии.
    x = np.linspace(x1, x2, length)
    y = np.linspace(y1, y2, length)

    cc = np.round(x).astype(np.int64) # Колонки
    rr = np.round(y).astype(np.int64) # Ряды

    # Отсеиваем пиксели, которые выходят за пределы холста.
    valid_indices = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
    rr_valid, cc_valid = rr[valid_indices], cc[valid_indices]

    # Преобразуем 2D координаты (ряд, колонка) в 1D индекс для плоского массива.
    return rr_valid * size + cc_valid


class CanvasManager:
    """
    Управляет холстом: вычисляет координаты гвоздей, кэширует геометрию
    линий и рендерит итоговое изображение по заданной хромосоме.
    """
    def __init__(self, size: int, num_pins: int):
        self.size = size
        self.num_pins = num_pins
        self.pin_coords = self._calculate_pin_coords()
        
        # Кэши для хранения уже рассчитанных линий, чтобы не повторять вычисления.
        # Ключ - отсортированная пара индексов гвоздей.
        self.pixel_cache: Dict[Tuple[int, int], np.ndarray] = {} # Для GPU
        self.mask_cache: Dict[Tuple[Any, ...], np.ndarray] = {}   # Для CPU
        self.mask_cache_lock = threading.Lock() # Блокировка для потокобезопасного доступа к кэшу.

    def _calculate_pin_coords(self) -> np.ndarray:
        """Рассчитывает и возвращает 2D-координаты всех гвоздей, расположенных по кругу."""
        coords = []
        # Гвозди располагаются на окружности, которая немного меньше размера холста.
        radius = self.size / 2 - (self.size / 40)
        center = self.size / 2
        for i in range(self.num_pins):
            angle = (i / self.num_pins) * 2 * math.pi
            x = center + radius * math.cos(angle)
            y = center + radius * math.sin(angle)
            coords.append((x, y))
        return np.array(coords)

    def get_line_pixel_indices(self, p1_idx: int, p2_idx: int) -> np.ndarray:
        """Возвращает кэшированные или вычисляет новые 1D-индексы пикселей для линии."""
        key = tuple(sorted((p1_idx, p2_idx)))
        if key in self.pixel_cache:
            return self.pixel_cache[key]
        
        p1 = self.pin_coords[p1_idx]
        p2 = self.pin_coords[p2_idx]
        
        pixels = _calculate_line_pixels_numpy(p1[0], p1[1], p2[0], p2[1], self.size)
        self.pixel_cache[key] = pixels
        return pixels

    def get_line_mask_cpu(self, p1_idx: int, p2_idx: int, thickness: int) -> np.ndarray:
        """Возвращает кэшированную или создает новую 2D-маску линии для рендеринга на CPU."""
        key = (tuple(sorted((p1_idx, p2_idx))), self.size, thickness)
        
        if key in self.mask_cache:
            return self.mask_cache[key]
            
        with self.mask_cache_lock:
            # Двойная проверка на случай, если другой поток уже создал маску, пока мы ждали блокировку.
            if key in self.mask_cache: return self.mask_cache[key]
            
            mask = np.zeros((self.size, self.size), dtype=np.float32)
            p1 = tuple(self.pin_coords[p1_idx].astype(int))
            p2 = tuple(self.pin_coords[p2_idx].astype(int))
            
            # Рисуем сглаженную линию на маске с помощью OpenCV.
            cv2.line(mask, p1, p2, color=1.0, thickness=thickness, lineType=cv2.LINE_AA)
            
            self.mask_cache[key] = mask
            return mask

    def render_chromosome_to_numpy(self, chromosome: List[int], thickness: int = 1) -> np.ndarray:
        """
        Рендерит полную хромосому (последовательность линий) в изображение NumPy.
        Использует CPU.
        """
        if not chromosome or len(chromosome) < 2:
            # Если хромосома пуста, возвращаем белый холст.
            return np.full((self.size, self.size), 255, dtype=np.uint8)

        # Карта "темноты": чем больше нитей проходит через пиксель, тем выше значение.
        darkness_map = np.zeros((self.size, self.size), dtype=np.float32)
        num_lines = len(chromosome) - 1
    
        # alpha - это прозрачность одной нити. Она динамически подбирается так,
        # чтобы при большом количестве нитей изображение не становилось абсолютно черным.
        alpha = min(0.2, 1600 / (num_lines + 1))
    
        # Последовательно "накладываем" маски всех линий из хромосомы на карту темноты.
        for i in range(num_lines):
            line_mask = self.get_line_mask_cpu(chromosome[i], chromosome[i+1], thickness)
            darkness_map += line_mask * alpha
        
        # Инвертируем карту темноты (1.0 - darkness), чтобы получить
        # изображение "темные нити на белом фоне".
        image_float = 1.0 - darkness_map
        # Преобразуем в 8-битный формат (0-255) для отображения или сохранения.
        final_image = np.clip(image_float * 255, 0, 255).astype(np.uint8)
        return final_image

    def render_chromosome_to_pil(self, chromosome: List[int], thickness: int = 1) -> Image.Image:
        """Рендерит хромосому и возвращает результат в виде объекта PIL Image."""
        numpy_image = self.render_chromosome_to_numpy(chromosome, thickness)
        return Image.fromarray(numpy_image)
    
    def precompute_line_data_for_gpu(self, device: Any) -> Dict[Tuple[int, int], Any]:
        """
        Заранее рассчитывает пиксельные индексы для всех возможных линий
        и загружает их на GPU в виде тензоров PyTorch.
        """
        import torch
        line_pixels_map_gpu = {}
        for i in range(self.num_pins):
            for j in range(i + 1, self.num_pins):
                pixel_indices_np = self.get_line_pixel_indices(i, j)
                if len(pixel_indices_np) > 0:
                    key = tuple(sorted((i, j)))
                    line_pixels_map_gpu[key] = torch.from_numpy(pixel_indices_np).long().to(device)
        return line_pixels_map_gpu
