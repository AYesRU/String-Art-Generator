import numpy as np
import cv2
from PIL import Image
from typing import Optional

def prepare_target_image(original_image: Optional[Image.Image], target_size: int) -> np.ndarray:
    """
    Подготавливает исходное изображение для генетического алгоритма.
    1. Конвертирует в оттенки серого.
    2. Нормализует контраст, если он слишком низкий.
    3. Масштабирует изображение до нужного размера с сохранением пропорций.
    4. Помещает отмасштабированное изображение в центр белого квадрата.
    """
    if original_image is None:
        # Если изображение не загружено, возвращаем просто белый холст.
        return np.full((target_size, target_size), 255, dtype=np.uint8)

    # Конвертируем изображение в Ч/Б (градации серого)
    img_gray = np.array(original_image.convert('L'))
    
    # Если изображение почти однотонное (например, полностью черное или белое),
    # применяем нормализацию, чтобы расширить диапазон яркости до 0-255.
    # Это помогает алгоритму лучше "видеть" детали.
    if img_gray.max() - img_gray.min() < 10:
        if img_gray.max() == img_gray.min():
            # Если все пиксели одинаковые, делаем их средне-серыми.
            img_gray = np.full_like(img_gray, 128)
        else:
            # Растягиваем гистограмму яркости.
            img_gray = ((img_gray - img_gray.min()) * 255 / (img_gray.max() - img_gray.min())).astype(np.uint8)

    h, w = img_gray.shape
    # Вычисляем коэффициент масштабирования, чтобы изображение вписалось в квадрат
    # target_size x target_size, сохраняя свои пропорции.
    scale = min(target_size / w, target_size / h)
    
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Изменяем размер изображения с помощью интерполяции INTER_AREA,
    # которая хорошо подходит для уменьшения изображений.
    resized = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Создаем белый фон (холст) нужного размера.
    background = np.full((target_size, target_size), 255, dtype=np.uint8)

    # Вычисляем координаты для центрирования изображения на холсте.
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2

    # "Вклеиваем" отмасштабированное изображение в центр белого фона.
    background[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return background
