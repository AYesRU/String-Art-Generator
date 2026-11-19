# -*- coding: utf-8 -*-
"""
Предобработка изображений, маскирование под форму рамы, выделение контуров
и значимости (Sobel + DoG + FineGrained Saliency), цветоделение CMYK и авто-подбор параметров.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .geometry import CanvasManager


def _composite_onto_bg(pil_img: Image.Image, bg_rgb: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Корректно преобразует прозрачные RGBA/P изображения на заданный сплошной фон."""
    if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
        rgba = pil_img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (*bg_rgb, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return pil_img.convert("RGB")


def _composite_onto_white(pil_img: Image.Image) -> Image.Image:
    """Корректно преобразует прозрачные RGBA/P изображения на чистый белый фон."""
    return _composite_onto_bg(pil_img, (255, 255, 255))


def _fit_image_to_frame(
    pil_img: Image.Image, target_size: int, shape: str, invert_board: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Вписывает изображение в ограничивающий прямоугольник активной формы гвоздей
    и возвращает (rgb_fitted, gray_fitted, polygon_mask).
    """
    cm = CanvasManager(target_size, 180, shape=shape)
    mask = cm.get_polygon_mask(dilate_px=2)
    x0, y0, x1, y1 = cm.get_content_bbox()
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)

    fill_val = 0 if invert_board else 255
    rgb_src = np.array(_composite_onto_bg(pil_img, (fill_val, fill_val, fill_val)), dtype=np.uint8)
    src_h, src_w = rgb_src.shape[:2]
    scale = min(box_w / max(1, src_w), box_h / max(1, src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    resized_rgb = cv2.resize(rgb_src, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas_rgb = np.full((target_size, target_size, 3), fill_val, dtype=np.uint8)
    off_x = x0 + (box_w - new_w) // 2
    off_y = y0 + (box_h - new_h) // 2
    canvas_rgb[off_y : off_y + new_h, off_x : off_x + new_w] = resized_rgb

    # Зануляем область за пределами рамы гвоздей (чистый белый или черный фон доски)
    canvas_rgb = np.where(mask[:, :, None] == 255, canvas_rgb, fill_val)
    canvas_gray = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2GRAY)
    return canvas_rgb, canvas_gray, mask


def _is_dark_board_mode(color_mode: str) -> bool:
    return "Белая нить" in color_mode or "Темная" in color_mode


def prepare_target_image_bundle(
    original_image: Optional[Image.Image],
    target_size: int,
    clahe_clip: float = 1.3,
    gamma: float = 0.80,
    shape: str = "Круг",
    color_mode: str = "Монохром (Черная нить)",
) -> Dict[str, Any]:
    """
    Полный конвейер предобработки: вписывание в геометрию рамы, очистка фоновых зон,
    построение карт контуров (Sobel), значимости (Saliency + DoG) и каналов (Mono / Dark Board / CMYK).
    """
    invert_board = _is_dark_board_mode(color_mode)

    if original_image is None:
        fill_val = 18 if invert_board else 255
        empty_rgb = np.full((target_size, target_size, 3), fill_val, dtype=np.uint8)
        empty_gray = np.full((target_size, target_size), fill_val, dtype=np.uint8)
        empty_maps = np.zeros((target_size, target_size), dtype=np.uint8)
        empty_imp = np.ones((target_size, target_size), dtype=np.float32)
        return {
            "rgb": empty_rgb,
            "gray": empty_gray,
            "edges": empty_maps,
            "saliency": empty_maps,
            "importance": empty_imp,
            "mask": np.full((target_size, target_size), 255, dtype=np.uint8),
            "invert_board": invert_board,
            "channels": [
                {
                    "name": "Белая нить" if invert_board else "Черная нить",
                    "color_hex": "#f5f5f5" if invert_board else "#111111",
                    "darkness": np.zeros((target_size, target_size), dtype=np.float32),
                    "importance": empty_imp,
                    "edges": empty_maps,
                }
            ],
        }

    canvas_rgb, background, mask = _fit_image_to_frame(original_image, target_size, shape, invert_board=invert_board)

    # Мягкий билатеральный фильтр сохраняет зрачки/контуры и убирает шум кожи
    background = cv2.bilateralFilter(background, d=5, sigmaColor=22, sigmaSpace=22)

    if clahe_clip > 0.1:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
        background = clahe.apply(background)

    # Отсечка фоновой зоны внутри маски, чтобы не плести "кашу" на пустом фоне
    inside_vals = background[mask == 255]
    if inside_vals.size > 0:
        p_bounds = (16, 96) if invert_board else (4, 84)
        p_low, p_high = np.percentile(inside_vals, p_bounds)
        if p_high - p_low > 20:
            norm_float = (background.astype(np.float32) - p_low) / (p_high - p_low)
            background = np.clip(norm_float * 255.0, 0.0, 255.0).astype(np.uint8)

    if abs(gamma - 1.0) > 0.03:
        inv_gamma = float(gamma) if invert_board else (1.0 / max(0.1, float(gamma)))
        table = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)], dtype=np.uint8)
        background = cv2.LUT(background, table)

    exterior_val = 0 if invert_board else 255
    background = np.where(mask == 255, background, exterior_val)

    # Карта контуров Собеля
    blurred = cv2.GaussianBlur(background, (3, 3), 0)
    sobelx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(sobelx, sobely)
    grad_mag = np.where(mask == 255, grad_mag, 0.0)
    max_g = float(np.max(grad_mag))
    edges = (grad_mag / max_g * 255.0).astype(np.uint8) if max_g > 0 else np.zeros_like(background, dtype=np.uint8)

    # Визуальная значимость (FineGrained Saliency + DoG)
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
    dog = np.where(mask == 255, dog, 0.0)
    max_dog = float(dog.max())
    if max_dog > 0:
        dog /= max_dog

    combined_saliency = np.clip(0.35 * saliency_map + 0.65 * dog, 0.0, 1.0)
    combined_saliency = np.where(mask == 255, combined_saliency, 0.0)

    saliency_uint8 = (combined_saliency * 255.0).astype(np.uint8)
    importance = (combined_saliency * 3.5 + 0.6).astype(np.float32)
    importance = np.where(mask == 255, importance, 0.15).astype(np.float32)

    channels = extract_channels(
        canvas_rgb, background, importance, mask, color_mode, gamma=gamma, edges=edges
    )

    return {
        "rgb": canvas_rgb,
        "gray": background,
        "edges": edges,
        "saliency": saliency_uint8,
        "importance": importance,
        "mask": mask,
        "invert_board": invert_board,
        "channels": channels,
    }


def _compute_channel_edges_and_importance(
    ch_density: np.ndarray,
    base_importance: np.ndarray,
    mask: np.ndarray,
    boost: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Строит индивидуальную карту контуров Собеля и значимости для отдельного CMYK-канала."""
    blur = cv2.GaussianBlur(ch_density.astype(np.float32), (3, 3), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    grad = np.where(mask == 255, grad, 0.0)
    max_g = float(np.max(grad))
    if max_g > 1e-5:
        grad_norm = grad / max_g
    else:
        grad_norm = np.zeros_like(ch_density, dtype=np.float32)

    ch_edges = (grad_norm * 255.0).astype(np.uint8)
    ch_imp = (0.55 * base_importance + 0.45 * (grad_norm * 3.5 + 0.6)) * float(boost)
    ch_imp = np.where(mask == 255, ch_imp, 0.15).astype(np.float32)
    return ch_edges, ch_imp


def extract_channels(
    canvas_rgb: np.ndarray,
    gray_target: np.ndarray,
    importance: np.ndarray,
    mask: np.ndarray,
    color_mode: str,
    gamma: float = 0.85,
    edges: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Формирует список целевых слоев плотности (Черная нить, Белая нить на темном или 4 слоя CMYK)."""
    if edges is None:
        edges = np.zeros_like(gray_target, dtype=np.uint8)

    if _is_dark_board_mode(color_mode):
        white_density = gray_target.astype(np.float32) / 255.0
        white_density = np.where(mask == 255, white_density, 0.0).astype(np.float32)
        return [
            {
                "name": "Белая нить",
                "color_hex": "#f5f5f5",
                "darkness": white_density,
                "importance": importance,
                "edges": edges,
            }
        ]

    if "CMYK" not in color_mode:
        mono_dark = (255.0 - gray_target.astype(np.float32)) / 255.0
        mono_dark = np.where(mask == 255, mono_dark, 0.0).astype(np.float32)
        return [
            {
                "name": "Черная нить",
                "color_hex": "#111111",
                "darkness": mono_dark,
                "importance": importance,
                "edges": edges,
            }
        ]

    # Для CMYK слегка сглаживаем и применяем гамму + защиту светлого фона
    rgb_smooth = cv2.bilateralFilter(canvas_rgb, d=5, sigmaColor=18, sigmaSpace=18)
    if abs(gamma - 1.0) > 0.03:
        inv_gamma = 1.0 / max(0.1, float(gamma))
        table = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)], dtype=np.uint8)
        rgb_smooth = cv2.LUT(rgb_smooth, table)

    rgb_norm = np.clip(rgb_smooth.astype(np.float32) / 255.0, 0.0, 1.0)
    r, g, b = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]

    k = 1.0 - np.maximum(np.maximum(r, g), b)
    denom = np.maximum(1.0 - k, 1e-5)
    c = np.clip((1.0 - r - k) / denom, 0.0, 1.0)
    m = np.clip((1.0 - g - k) / denom, 0.0, 1.0)
    y = np.clip((1.0 - b - k) / denom, 0.0, 1.0)
    k = np.clip(k, 0.0, 1.0)

    # Гасим паразитные слабые оттенки на почти белой бумаге
    for ch_arr in (c, m, y, k):
        ch_arr[mask == 0] = 0.0
        ch_arr[ch_arr < 0.06] = 0.0

    c_edges, c_imp = _compute_channel_edges_and_importance(c, importance, mask, boost=1.0)
    m_edges, m_imp = _compute_channel_edges_and_importance(m, importance, mask, boost=1.0)
    y_edges, y_imp = _compute_channel_edges_and_importance(y, importance, mask, boost=1.0)
    k_edges, k_imp = _compute_channel_edges_and_importance(k, importance, mask, boost=1.15)

    return [
        {"name": "Циан (Cyan)", "color_hex": "#00a8e8", "darkness": c.astype(np.float32), "importance": c_imp, "edges": c_edges},
        {"name": "Маджента (Magenta)", "color_hex": "#e6007a", "darkness": m.astype(np.float32), "importance": m_imp, "edges": m_edges},
        {"name": "Желтый (Yellow)", "color_hex": "#f5c400", "darkness": y.astype(np.float32), "importance": y_imp, "edges": y_edges},
        {"name": "Черный (Black)", "color_hex": "#111111", "darkness": k.astype(np.float32), "importance": k_imp, "edges": k_edges},
    ]


def prepare_target_image(
    original_image: Optional[Image.Image],
    target_size: int,
    clahe_clip: float = 1.3,
    gamma: float = 0.80,
    shape: str = "Круг",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Совместимая обертка, возвращающая (background, edges, saliency_uint8, importance)."""
    bundle = prepare_target_image_bundle(
        original_image,
        target_size,
        clahe_clip=clahe_clip,
        gamma=gamma,
        shape=shape,
        color_mode="Монохром (Черная нить)",
    )
    return bundle["gray"], bundle["edges"], bundle["saliency"], bundle["importance"]


def auto_recommend_parameters(original_image: Image.Image, shape: str = "Круг") -> Dict[str, Any]:
    """
    Интеллектуальный расчет параметров под изображение для четкого плетения без "каши".
    """
    rgb_clean = _composite_onto_white(original_image)
    img_gray = np.array(rgb_clean.convert("L"))
    h, w = img_gray.shape
    scale = min(400.0 / max(1, w), 400.0 / max(1, h))
    resized = cv2.resize(
        img_gray,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )

    p5, p95 = np.percentile(resized, (5, 95))
    contrast_range = float(p95 - p5)
    mean_lum = float(np.mean(resized))

    clahe_clip = 1.5 if contrast_range < 135 else 1.1
    gamma = 0.74 if mean_lum < 115 else (0.90 if mean_lum > 175 else 0.82)

    temp_img, edges, _, _ = prepare_target_image(original_image, 400, clahe_clip, gamma, shape)

    edge_density = float(np.mean(edges > 45))
    lap_std = float(np.std(cv2.Laplacian(temp_img, cv2.CV_64F)))
    complexity_score = float(np.clip((edge_density - 0.04) / 0.18 * 0.65 + (lap_std - 8.0) / 38.0 * 0.35, 0.0, 1.0))
    darkness_ratio = float(np.mean(np.maximum(0.0, (215.0 - temp_img) / 215.0)))

    raw_pins = 240 + complexity_score * 64
    recommended_pins = int(np.clip(round(raw_pins / 4.0) * 4, 220, 320))

    raw_lines = 950 + darkness_ratio * 650 + complexity_score * 250
    recommended_lines = int(np.clip(round(raw_lines / 50.0) * 50, 950, 1850))

    line_weight = int(np.clip(round(75 - (recommended_lines - 1100) / 45.0), 62, 85))
    fearless_factor = int(np.clip(round(94 - complexity_score * 6), 86, 98))
    contour_weight = int(np.clip(round(50 + complexity_score * 20), 40, 75))
    radon_influence = int(np.clip(round(30 + complexity_score * 15), 20, 50))
    stages = 5 if recommended_lines <= 1500 else 6

    summary = (
        f"Детали: {int(complexity_score * 100)}% | Тень: {int(darkness_ratio * 100)}% → "
        f"{recommended_pins} гвоздей, {recommended_lines} нитей"
    )

    return {
        "pins": recommended_pins,
        "lines": recommended_lines,
        "line_weight": line_weight,
        "clahe_clip": round(clahe_clip, 1),
        "gamma": round(gamma, 2),
        "fearless_factor": fearless_factor,
        "contour_weight": contour_weight,
        "radon_influence": radon_influence,
        "stages": stages,
        "summary": summary,
    }
