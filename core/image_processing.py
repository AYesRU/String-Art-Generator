import numpy as np
import cv2
from PIL import Image
from typing import Optional

def prepare_target_image(original_image: Optional[Image.Image], target_size: int) -> np.ndarray:
    """
    Подготовка исходного изображения для алгоритма генерации строкового искусства.
    
    Параметры:
        original_image: Optional[Image.Image] - Исходное изображение
        target_size: int - Целевой размер изображения
        
    Возвращает:
        np.ndarray: Подготовленное изображение в градациях серого
    """
    if original_image is None:
        # Возврат белого холста если изображение не загружено
        return np.full((target_size, target_size), 255, dtype=np.uint8)

    # Конвертация в градации серого
    img_gray = np.array(original_image.convert('L'))
    
    # Нормализация контраста для однотонных изображений
    if img_gray.max() - img_gray.min() < 10:
        if img_gray.max() == img_gray.min():
            # Создание средне-серого изображения
            img_gray = np.full_like(img_gray, 128)
        else:
            # Растягивание гистограммы яркости
            img_gray = ((img_gray - img_gray.min()) * 255 / 
                       (img_gray.max() - img_gray.min())).astype(np.uint8)

    h, w = img_gray.shape
    # Вычисление коэффициента масштабирования
    scale = min(target_size / w, target_size / h)
    
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Изменение размера с сохранением пропорций
    resized = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Создание белого фона
    background = np.full((target_size, target_size), 255, dtype=np.uint8)

    # Центрирование изображения на фоне
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2

    background[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return background