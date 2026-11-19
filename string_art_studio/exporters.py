# -*- coding: utf-8 -*-
"""
Модуль экспорта результатов и сохранения проектов String Art Studio Pro:
- Пошаговая текстовая карта (.txt) с расчетом метража и катушек
- Таблица переходов (.csv)
- Программа для ЧПУ-станков (.gcode) в декартовых координатах (Y вверх)
- Масштабируемый вектор (.svg) по слоям (поддержка светлой и темной доски)
- Чертеж шаблона сверления гвоздей 1:1 в мм (.svg) с контуром и контрольной шкалой 100 мм
- Ultra-HD мастер-растр (.png, 2400x2400px+)
- Автономный интерактивный веб-ассистент (.html) с Web Speech API, изоляцией localStorage, переходом к шагу и темной темой
- Сохранение и загрузка проекта (.sart.json)
"""

import hashlib
import html as html_lib
import json
import math
from typing import Any, Dict, List, Optional

import numpy as np

from . import APP_TITLE
from .geometry import CanvasManager
from .i18n import normalize_lang, to_display_shape, translate_layer_name


def _slice_layer_path(
    path: List[int],
    l_idx: int,
    max_step: Optional[int],
    active_layer_idx: int,
) -> List[int]:
    if not path:
        return []
    if max_step is None:
        return path
    if l_idx < active_layer_idx:
        return path
    if l_idx == active_layer_idx:
        limit = min(len(path), max(2, int(max_step) + 1))
        return path[:limit]
    return []


def export_instructions_txt(
    filepath: str,
    layers: List[Dict[str, Any]],
    canvas_manager: CanvasManager,
    max_step: Optional[int] = None,
    active_layer_idx: int = 0,
    lang: str = "ru",
) -> float:
    """Сохраняет пошаговую инструкцию плетения в .txt и возвращает общий расход в метрах."""
    code = normalize_lang(lang)
    is_en = code == "en"
    disp_shape = to_display_shape(canvas_manager.shape, code)

    if is_en:
        lines = [
            "=" * 78,
            f"   {APP_TITLE.upper()} — STRING ART WEAVING SCHEME",
            "=" * 78,
            f"Frame shape: {disp_shape} | Total pins: {canvas_manager.num_pins}",
            f"Physical size (Ø/width): {canvas_manager.physical_cm:.1f} cm",
            "=" * 78,
            "",
        ]
    else:
        lines = [
            "=" * 78,
            f"   {APP_TITLE.upper()} — СХЕМА ПЛЕТЕНИЯ НИТЯМИ",
            "=" * 78,
            f"Форма рамы: {disp_shape} | Всего гвоздей: {canvas_manager.num_pins}",
            f"Физический размер (Ø/ширина): {canvas_manager.physical_cm:.1f} см",
            "=" * 78,
            "",
        ]
    grand_total_m = 0.0
    m_per_px = (canvas_manager.physical_cm / 100.0) / max(1.0, float(canvas_manager.size * 0.93))

    for l_idx, layer in enumerate(layers):
        p = _slice_layer_path(layer.get("path", []), l_idx, max_step, active_layer_idx)
        lname = translate_layer_name(str(layer.get("name", "Основной")), code)
        if len(p) < 2:
            if max_step is None or l_idx <= active_layer_idx:
                if is_en:
                    lines.append(f"--- LAYER {l_idx + 1}: {lname} ({layer.get('color_hex', '#111111')}) ---")
                    lines.append("Threads in layer: 0 (color absent / not required)")
                else:
                    lines.append(f"--- СЛОЙ {l_idx + 1}: {lname} ({layer.get('color_hex', '#111111')}) ---")
                    lines.append("Нитей в слое: 0 (цвет отсутствует / не требуется)")
                lines.append("")
            continue
        if is_en:
            lines.append(f"--- LAYER {l_idx + 1}: {lname} ({layer.get('color_hex', '#111111')}) ---")
            lines.append(f"Threads in layer: {len(p) - 1}")
            lines.append("Step      From Pin     To Pin      Length (m)   Cumulative (m)")
        else:
            lines.append(f"--- СЛОЙ {l_idx + 1}: {lname} ({layer.get('color_hex', '#111111')}) ---")
            lines.append(f"Нитей в слое: {len(p) - 1}")
            lines.append("Шаг       От гвоздя    К гвоздю    Длина (м)    Нарастающий итог (м)")
        lines.append("-" * 74)

        cum_m = 0.0
        for i in range(len(p) - 1):
            p1, p2 = p[i], p[i + 1]
            c1 = canvas_manager.pin_coords[p1]
            c2 = canvas_manager.pin_coords[p2]
            chord_m = float(np.hypot(c2[0] - c1[0], c2[1] - c1[1])) * m_per_px
            cum_m += chord_m
            lines.append(f"{i + 1:<10}{p1:<13}{p2:<12}{chord_m:<13.3f}{cum_m:<.2f}")

        grand_total_m += cum_m
        spools = max(1, math.ceil(cum_m / 500.0))
        lines.append("")
        if is_en:
            lines.append(f"Layer '{lname}' total: {cum_m:.2f} m (~{spools} x 500 m spools)")
        else:
            lines.append(f"Итого по слою «{lname}»: {cum_m:.2f} м (~{spools} катушек по 500 м)")
        lines.append("")

    lines.append("=" * 78)
    if is_en:
        lines.append(f"TOTAL THREAD LENGTH: {grand_total_m:.2f} m")
        lines.append("Happy crafting!")
    else:
        lines.append(f"ОБЩИЙ РАСХОД НИТИ: {grand_total_m:.2f} м")
        lines.append("Успешного творчества!")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return grand_total_m


def export_csv(
    filepath: str,
    layers: List[Dict[str, Any]],
    canvas_manager: CanvasManager,
    lang: str = "ru",
) -> None:
    """Сохраняет переходы гвоздей и длину каждого сегмента в CSV (UTF-8 with BOM для Excel)."""
    code = normalize_lang(lang)
    m_per_px = (canvas_manager.physical_cm / 100.0) / max(1.0, float(canvas_manager.size * 0.93))
    header = "Layer,Color,Step,From_Pin,To_Pin,Length_m\n" if code == "en" else "Слой,Цвет,Шаг,От_гвоздя,К_гвоздю,Длина_м\n"
    with open(filepath, "w", encoding="utf-8-sig") as f:
        f.write(header)
        for layer in layers:
            path = layer.get("path", [])
            name = translate_layer_name(str(layer.get("name", "Основной")), code).replace(",", " ")
            color = layer.get("color_hex", "#111111")
            for i in range(len(path) - 1):
                p1, p2 = path[i], path[i + 1]
                c1 = canvas_manager.pin_coords[p1]
                c2 = canvas_manager.pin_coords[p2]
                seg_m = float(np.hypot(c2[0] - c1[0], c2[1] - c1[1])) * m_per_px
                f.write(f"{name},{color},{i + 1},{p1},{p2},{seg_m:.4f}\n")


def export_gcode_cnc(
    filepath: str,
    layers: List[Dict[str, Any]],
    canvas_manager: CanvasManager,
    feed_rate: int = 1500,
    z_safe: float = 6.0,
    z_work: float = 0.5,
) -> None:
    """Генерирует программу G-Code для ЧПУ-станка плетения нитями (мм, декартова ось Y вверх)."""
    scale = (canvas_manager.physical_cm * 10.0) / float(canvas_manager.size)
    size_f = float(canvas_manager.size)
    lines = [
        f"; Generated by {APP_TITLE}",
        f"; Shape: {canvas_manager.shape}, Pins: {canvas_manager.num_pins}, Size: {canvas_manager.physical_cm:.1f} cm",
        "; Coordinate system: Cartesian Y-up (X right, Y up, 0,0 at bottom-left)",
        "G21 ; Millimeters",
        "G90 ; Absolute positioning",
        f"G0 Z{z_safe:.2f} F{feed_rate}",
    ]
    for l_idx, layer in enumerate(layers):
        p = layer.get("path", [])
        if len(p) < 2:
            continue
        lines.append(f"\n; --- Layer {l_idx + 1}: {layer.get('name', 'Main')} ---")
        c0 = canvas_manager.pin_coords[p[0]]
        x0 = float(c0[0]) * scale
        y0 = (size_f - float(c0[1])) * scale
        lines.append(f"G0 X{x0:.2f} Y{y0:.2f}")
        lines.append(f"G1 Z{z_work:.2f} F{max(100, feed_rate // 2)}")
        for idx in p[1:]:
            pt = canvas_manager.pin_coords[idx]
            x_mm = float(pt[0]) * scale
            y_mm = (size_f - float(pt[1])) * scale
            lines.append(f"G1 X{x_mm:.2f} Y{y_mm:.2f} F{feed_rate}")
        lines.append(f"G0 Z{z_safe:.2f}")

    lines.append("M5 ; Tool stop")
    lines.append("G0 X0.00 Y0.00 ; Return home")
    lines.append("M2 ; End of program")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def export_svg_vector(
    filepath: str,
    layers: List[Dict[str, Any]],
    canvas_manager: CanvasManager,
    thread_thickness: int = 1,
    max_step: Optional[int] = None,
    active_layer_idx: int = 0,
    invert_board: bool = False,
) -> None:
    """Экспортирует многослойный векторный SVG-чертеж со всеми нитями и гвоздями."""
    svg_size = 1200
    cm = CanvasManager(svg_size, canvas_manager.num_pins, canvas_manager.shape, canvas_manager.physical_cm)
    coords = cm.pin_coords
    stroke_w = max(0.35, float(thread_thickness) * 0.55)
    dark_mode = bool(invert_board and len(layers) <= 1)
    bg_hex = "#121418" if dark_mode else "#ffffff"
    pin_hex = "#b8bec8" if dark_mode else "#333333"

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_size}" height="{svg_size}" viewBox="0 0 {svg_size} {svg_size}">',
        f'  <rect width="100%" height="100%" fill="{bg_hex}"/>',
    ]
    for l_idx, layer in enumerate(layers):
        active_path = _slice_layer_path(layer.get("path", []), l_idx, max_step, active_layer_idx)
        if len(active_path) < 2:
            continue
        color = layer.get("color_hex", "#f5f5f5" if dark_mode else "#111111")
        opacity = "0.48" if len(layers) > 1 else "0.72"
        path_data = "M " + " L ".join(f"{coords[p][0]:.2f},{coords[p][1]:.2f}" for p in active_path)
        lines.append(
            f'  <g id="layer_{l_idx}">\n'
            f'    <path d="{path_data}" stroke="{color}" stroke-opacity="{opacity}" '
            f'stroke-width="{stroke_w:.2f}" fill="none" stroke-linejoin="round" stroke-linecap="round"/>\n'
            f"  </g>"
        )

    lines.append(f'  <g id="pins" fill="{pin_hex}">')
    for pt in coords:
        lines.append(f'    <circle cx="{pt[0]:.2f}" cy="{pt[1]:.2f}" r="1.4"/>')
    lines.append("  </g>")
    lines.append("</svg>")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def export_drill_template_svg(filepath: str, canvas_manager: CanvasManager, lang: str = "ru") -> None:
    """
    Экспортирует шаблон сверления 1:1 в реальных миллиметрах:
    - контур рамы и центральное перекрестие,
    - метки и нумерация всех гвоздей,
    - контрольная калибровочная линейка 100 мм (или 50 мм для малых досок) для проверки масштаба печати.
    """
    code = normalize_lang(lang)
    is_en = code == "en"
    disp_shape = to_display_shape(canvas_manager.shape, code)
    dim_mm = canvas_manager.physical_cm * 10.0
    scale = dim_mm / float(canvas_manager.size)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {dim_mm:.2f} {dim_mm:.2f}" width="{dim_mm:.2f}mm" height="{dim_mm:.2f}mm">',
        '  <rect width="100%" height="100%" fill="#ffffff" stroke="#cccccc" stroke-width="0.4"/>',
    ]

    # Контур рамы по гвоздям
    if len(canvas_manager.pin_coords) >= 3:
        poly_pts = " ".join(f"{float(pt[0]) * scale:.2f},{float(pt[1]) * scale:.2f}" for pt in canvas_manager.pin_coords)
        lines.append(
            f'  <polygon points="{poly_pts}" fill="none" stroke="#999999" stroke-width="0.3" stroke-dasharray="2.5,2.0"/>'
        )

    # Центральное перекрестие
    cx_mm = dim_mm / 2.0
    lines.append(
        f'  <line x1="{cx_mm - 5:.2f}" y1="{cx_mm:.2f}" x2="{cx_mm + 5:.2f}" y2="{cx_mm:.2f}" stroke="#0969da" stroke-width="0.25"/>'
    )
    lines.append(
        f'  <line x1="{cx_mm:.2f}" y1="{cx_mm - 5:.2f}" x2="{cx_mm:.2f}" y2="{cx_mm + 5:.2f}" stroke="#0969da" stroke-width="0.25"/>'
    )

    # Заголовок / спецификация
    legend_raw = (
        f"{APP_TITLE} | {disp_shape} | {canvas_manager.num_pins} pins | {canvas_manager.physical_cm:.1f} cm (1:1)"
        if is_en
        else f"{APP_TITLE} | {disp_shape} | {canvas_manager.num_pins} гвоздей | {canvas_manager.physical_cm:.1f} см (1:1)"
    )
    legend_text = html_lib.escape(legend_raw)
    lines.append(
        f'  <text x="6.00" y="7.50" font-size="3.2" font-family="Arial, sans-serif" fill="#1f2328">{legend_text}</text>'
    )

    # Калибровочная линейка 100 мм (или 50 мм для компактных рам < 140 мм)
    bar_mm = 100.0 if dim_mm >= 140.0 else 50.0
    bx0 = 8.0
    bx1 = bx0 + bar_mm
    by = dim_mm - 7.0
    bar_lbl = f"{int(bar_mm)} mm (1:1 scale check)" if is_en else f"{int(bar_mm)} mm (проверка масштаба 1:1)"
    lines.append('  <g id="calibration_bar" stroke="#0969da" stroke-width="0.35">')
    lines.append(f'    <line x1="{bx0:.2f}" y1="{by:.2f}" x2="{bx1:.2f}" y2="{by:.2f}"/>')
    lines.append(f'    <line x1="{bx0:.2f}" y1="{by - 2.2:.2f}" x2="{bx0:.2f}" y2="{by + 2.2:.2f}"/>')
    lines.append(f'    <line x1="{(bx0 + bx1) / 2.0:.2f}" y1="{by - 1.4:.2f}" x2="{(bx0 + bx1) / 2.0:.2f}" y2="{by + 1.4:.2f}"/>')
    lines.append(f'    <line x1="{bx1:.2f}" y1="{by - 2.2:.2f}" x2="{bx1:.2f}" y2="{by + 2.2:.2f}"/>')
    lines.append(
        f'    <text x="{(bx0 + bx1) / 2.0:.2f}" y="{by - 2.6:.2f}" font-size="3.0" text-anchor="middle" '
        f'font-family="Arial, sans-serif" fill="#0969da" stroke="none">{bar_lbl}</text>'
    )
    lines.append("  </g>")

    for idx, pt in enumerate(canvas_manager.pin_coords):
        x_mm = float(pt[0]) * scale
        y_mm = float(pt[1]) * scale
        lines.append(
            f'  <line x1="{x_mm - 1.5:.2f}" y1="{y_mm:.2f}" x2="{x_mm + 1.5:.2f}" y2="{y_mm:.2f}" stroke="#111111" stroke-width="0.25"/>'
        )
        lines.append(
            f'  <line x1="{x_mm:.2f}" y1="{y_mm - 1.5:.2f}" x2="{x_mm:.2f}" y2="{y_mm + 1.5:.2f}" stroke="#111111" stroke-width="0.25"/>'
        )
        lines.append(
            f'  <circle cx="{x_mm:.2f}" cy="{y_mm:.2f}" r="0.8" fill="none" stroke="#d1242f" stroke-width="0.2"/>'
        )
        if idx % 5 == 0 or idx == canvas_manager.num_pins - 1:
            lines.append(
                f'  <text x="{x_mm:.2f}" y="{y_mm - 2.2:.2f}" font-size="2.6" text-anchor="middle" font-family="Arial, sans-serif" fill="#1f2328">{idx}</text>'
            )

    lines.append("</svg>")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def export_png_ultra_hd(
    filepath: str,
    layers: List[Dict[str, Any]],
    canvas_manager: CanvasManager,
    thickness: int = 1,
    line_weight_percent: int = 72,
    output_size: int = 2400,
    max_step: Optional[int] = None,
    active_layer_idx: int = 0,
    invert_board: bool = False,
) -> None:
    """Рендерит и сохраняет Ultra-HD PNG без потери оптической плотности нити."""
    img = canvas_manager.render_layers_to_pil(
        layers,
        thickness=thickness,
        line_weight_percent=line_weight_percent,
        max_step=max_step,
        output_size=output_size,
        active_layer_idx=active_layer_idx,
        invert_board=invert_board,
    )
    img.save(filepath, "PNG")


def build_interactive_html5_assistant(
    title: str,
    canvas_manager: CanvasManager,
    layers: List[Dict[str, Any]],
    invert_board: bool = False,
    lang: str = "ru",
) -> str:
    """
    Генерирует автономный однофайловый HTML5-ассистент ручного плетения (RU / EN):
    - изолированный ключ localStorage на каждый проект,
    - предвычисленная таблица кумулятивного метража O(1),
    - переход к произвольному шагу, выбор интервала автоплея и переключатель светлой/темной темы.
    """
    code = normalize_lang(lang)
    is_en = code == "en"
    safe_title = html_lib.escape(title)
    disp_shape = html_lib.escape(to_display_shape(canvas_manager.shape, code))
    pins_list = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in canvas_manager.pin_coords]
    pins_json = json.dumps(pins_list)

    dark_board = bool(invert_board and len(layers) <= 1)
    default_thread_hex = "#f5f5f5" if dark_board else "#111111"

    clean_layers = []
    for layer in layers:
        path = layer.get("path", [])
        if len(path) >= 2:
            clean_layers.append({
                "name": translate_layer_name(str(layer.get("name", "Основной")), code),
                "color": layer.get("color_hex", default_thread_hex),
                "path": [int(x) for x in path],
            })
    if not clean_layers:
        clean_layers.append({"name": "Main" if is_en else "Основной", "color": default_thread_hex, "path": [0, 0]})

    layers_json = json.dumps(clean_layers, ensure_ascii=False)

    # Уникальный детерминированный ключ проекта для изоляции прогресса в localStorage
    sig_payload = f"{title}|{canvas_manager.shape}|{canvas_manager.num_pins}|{len(clean_layers)}|" + "|".join(
        f"{len(l['path'])}:{l['path'][:8]}:{l['path'][-8:]}" for l in clean_layers
    )
    project_hash = hashlib.sha1(sig_payload.encode("utf-8")).hexdigest()[:10]
    project_key = f"stringart_v96_{project_hash}_"

    br_rad = "50%" if canvas_manager.shape in ("Круг", "Овал") else "12px"
    board_bg = "#121418" if dark_board else "#ffffff"
    pin_dot_color = "#b8bec8" if dark_board else "#57606a"
    src_size = float(canvas_manager.size)
    physical_cm = float(canvas_manager.physical_cm)

    ui = (
        {
            "html_lang": "en",
            "page_suffix": "Interactive Weaving Assistant",
            "subtitle": f"Interactive step-by-step weaving assistant ({disp_shape}, {canvas_manager.num_pins} pins)",
            "lbl_move": "Current Move",
            "lbl_layer_init": "Layer: Main",
            "lbl_layer_prefix": "Layer: ",
            "lbl_step": "Step",
            "lbl_len": f"Length (Ø{physical_cm:.0f} cm)",
            "unit_m": " m",
            "lbl_rem": "Left: ~",
            "btn_prev": "&larr; Prev (A)",
            "btn_play": "&#9654; Auto",
            "btn_pause": "⏸ Pause",
            "btn_play_js": "▶ Auto",
            "btn_next": "Next (D / Space) &rarr;",
            "jump_ph": "Step #",
            "jump_title": "Jump to step",
            "btn_jump": "Go",
            "spd_title": "Auto-play interval",
            "spd_1": "Auto: 1.5 s",
            "spd_2": "Auto: 2.5 s",
            "spd_3": "Auto: 3.5 s",
            "spd_4": "Auto: 5.0 s",
            "voice_on": "🔊 Voice: ON",
            "voice_off": "🔇 Voice: OFF",
            "theme_light": "☀ Light Theme",
            "theme_dark": "🌙 Dark Theme",
            "btn_reset": "↺ Reset",
            "speech_prefix": "to ",
            "speech_lang": "en-US",
        }
        if is_en
        else {
            "html_lang": "ru",
            "page_suffix": "Интерактивный Ассистент Плетения",
            "subtitle": f"Интерактивный пошаговый ассистент ручного плетения ({disp_shape}, {canvas_manager.num_pins} гвоздей)",
            "lbl_move": "Текущий ход",
            "lbl_layer_init": "Слой: Основной",
            "lbl_layer_prefix": "Слой: ",
            "lbl_step": "Шаг",
            "lbl_len": f"Метраж (Ø{physical_cm:.0f} см)",
            "unit_m": " м",
            "lbl_rem": "Осталось: ~",
            "btn_prev": "&larr; Назад (A)",
            "btn_play": "&#9654; Авто",
            "btn_pause": "⏸ Пауза",
            "btn_play_js": "▶ Авто",
            "btn_next": "Вперед (D / Пробел) &rarr;",
            "jump_ph": "Шаг №",
            "jump_title": "Перейти к шагу",
            "btn_jump": "Перейти",
            "spd_title": "Интервал автоплея",
            "spd_1": "Авто: 1.5 с",
            "spd_2": "Авто: 2.5 с",
            "spd_3": "Авто: 3.5 с",
            "spd_4": "Авто: 5.0 с",
            "voice_on": "🔊 Голос: ВКЛ",
            "voice_off": "🔇 Голос: ВЫКЛ",
            "theme_light": "☀ Светлая тема",
            "theme_dark": "🌙 Темная тема",
            "btn_reset": "↺ Сброс",
            "speech_prefix": "на ",
            "speech_lang": "ru-RU",
        }
    )

    html = f"""<!DOCTYPE html>
<html lang="{ui['html_lang']}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>{safe_title} — {ui['page_suffix']}</title>
<style>
  :root {{
    --bg: #f6f8fa;
    --card: #ffffff;
    --border: #d0d7de;
    --text: #1f2328;
    --muted: #57606a;
    --accent: #0969da;
    --btn-hover: #eaeef2;
    --active: #00b8d9;
    --warn: #cf222e;
  }}
  body.dark-ui {{
    --bg: #0d1117;
    --card: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #58a6ff;
    --btn-hover: #21262d;
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
    transition: background 0.2s, color 0.2s;
  }}
  header {{ text-align: center; margin-bottom: 10px; }}
  h1 {{ font-size: 20px; color: var(--accent); }}
  .subtitle {{ font-size: 13px; color: var(--muted); }}
  .layer-selector {{ display: flex; gap: 6px; margin-bottom: 8px; flex-wrap: wrap; justify-content: center; }}
  .layer-btn {{ padding: 6px 12px; font-size: 13px; border-radius: 6px; }}
  .layer-btn.active {{ border-color: var(--accent); background: rgba(9,105,218,0.14); color: var(--accent); font-weight: 700; }}
  .dashboard {{
    display: flex;
    gap: 10px;
    justify-content: center;
    width: 100%;
    max-width: 780px;
    margin-bottom: 12px;
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 14px;
    flex: 1;
    text-align: center;
    box-shadow: 0 2px 6px rgba(31,35,40,0.05);
  }}
  .card-label {{ font-size: 11px; text-transform: uppercase; color: var(--muted); letter-spacing: 0.4px; }}
  .card-value {{ font-size: 24px; font-weight: 700; color: var(--accent); margin-top: 2px; }}
  .card-meta {{ font-size: 12px; color: var(--muted); }}
  #canvas-wrapper {{
    position: relative;
    width: min(88vw, 520px);
    height: min(88vw, 520px);
    background: {board_bg};
    border: 1px solid var(--border);
    border-radius: {br_rad};
    box-shadow: 0 4px 18px rgba(31,35,40,0.12);
    overflow: hidden;
  }}
  canvas {{ display: block; width: 100%; height: 100%; }}
  .slider-container {{
    width: 100%;
    max-width: 640px;
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 10px 0 4px;
  }}
  input[type=range] {{ flex: 1; accent-color: var(--accent); height: 6px; cursor: pointer; }}
  .controls-bar {{
    display: flex;
    gap: 8px;
    justify-content: center;
    align-items: center;
    width: 100%;
    max-width: 760px;
    margin: 6px 0;
    flex-wrap: wrap;
  }}
  button, select, input[type=number] {{
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
  }}
  button, select {{ cursor: pointer; user-select: none; }}
  input[type=number] {{ width: 92px; text-align: center; }}
  button:hover {{ background: var(--btn-hover); border-color: var(--accent); }}
  button.primary {{ background: var(--accent); color: #ffffff; font-weight: 700; border-color: var(--accent); }}
  button.primary:hover {{ filter: brightness(0.92); }}
</style>
</head>
<body class="{'dark-ui' if dark_board else ''}">
<header>
  <h1>{safe_title}</h1>
  <div class="subtitle">{ui['subtitle']}</div>
</header>
<div id="layer-tabs" class="layer-selector"></div>
<div class="dashboard">
  <div class="card">
    <div class="card-label">{ui['lbl_move']}</div>
    <div class="card-value" id="pin-info">0 &rarr; 0</div>
    <div class="card-meta" id="layer-name">{ui['lbl_layer_init']}</div>
  </div>
  <div class="card">
    <div class="card-label">{ui['lbl_step']}</div>
    <div class="card-value" id="step-counter">0 / 0</div>
    <div class="card-meta" id="percent-info">0.0%</div>
  </div>
  <div class="card">
    <div class="card-label">{ui['lbl_len']}</div>
    <div class="card-value" id="length-info">0.0{ui['unit_m']}</div>
    <div class="card-meta" id="remaining-info">{ui['lbl_rem']}0{ui['unit_m']}</div>
  </div>
</div>
<div id="canvas-wrapper">
  <canvas id="board-canvas" width="1040" height="1040"></canvas>
</div>
<div class="slider-container">
  <span style="font-size:12px;color:var(--muted);">1</span>
  <input type="range" id="step-slider" min="1" max="100" value="1">
  <span style="font-size:12px;color:var(--muted);" id="max-step-display">100</span>
</div>
<div class="controls-bar">
  <button id="btn-prev-10">&laquo; 10</button>
  <button id="btn-prev">{ui['btn_prev']}</button>
  <button id="btn-play" class="primary">{ui['btn_play']}</button>
  <button id="btn-next" class="primary">{ui['btn_next']}</button>
  <button id="btn-next-10">10 &raquo;</button>
</div>
<div class="controls-bar">
  <input type="number" id="jump-input" min="1" placeholder="{ui['jump_ph']}" title="{ui['jump_title']}">
  <button id="btn-jump">{ui['btn_jump']}</button>
  <select id="speed-select" title="{ui['spd_title']}">
    <option value="1500">{ui['spd_1']}</option>
    <option value="2500" selected>{ui['spd_2']}</option>
    <option value="3500">{ui['spd_3']}</option>
    <option value="5000">{ui['spd_4']}</option>
  </select>
  <button id="btn-voice">{ui['voice_on']}</button>
  <button id="btn-theme">{ui['theme_light'] if dark_board else ui['theme_dark']}</button>
  <button id="btn-reset">{ui['btn_reset']}</button>
</div>
<script>
  const PINS = {pins_json};
  const LAYERS = {layers_json};
  const SRC_SIZE = {src_size};
  const DIAMETER_CM = {physical_cm};
  const PROJECT_KEY = {json.dumps(project_key)};
  const BOARD_BG = {json.dumps(board_bg)};
  const PIN_COLOR = {json.dumps(pin_dot_color)};
  const TXT = {json.dumps(ui, ensure_ascii=False)};
  let curLayer = 0;
  let curStep = 1;
  let isPlaying = false;
  let playInterval = null;
  let voiceEnabled = true;

  const M_PER_PX = (DIAMETER_CM / 100.0) / (SRC_SIZE * 0.93);
  const CUM_LEN = LAYERS.map((layer) => {{
    const p = layer.path;
    const arr = new Float32Array(p.length);
    let acc = 0;
    for (let i = 0; i < p.length - 1; i++) {{
      const a = PINS[p[i]], b = PINS[p[i + 1]];
      if (a && b) acc += Math.hypot(b[0] - a[0], b[1] - a[1]) * M_PER_PX;
      arr[i + 1] = acc;
    }}
    return arr;
  }});

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
  const btnTheme = document.getElementById('btn-theme');
  const jumpInput = document.getElementById('jump-input');
  const speedSelect = document.getElementById('speed-select');

  function getPath() {{ return LAYERS[curLayer].path; }}
  function getTotal() {{ return Math.max(1, getPath().length - 1); }}

  const layerTabsBar = document.getElementById('layer-tabs');
  if (LAYERS.length > 1) {{
    LAYERS.forEach((l, idx) => {{
      const b = document.createElement('button');
      b.className = 'layer-btn' + (idx === 0 ? ' active' : '');
      b.textContent = l.name;
      b.onclick = () => switchLayer(idx);
      layerTabsBar.appendChild(b);
    }});
  }}

  function switchLayer(idx) {{
    curLayer = idx;
    try {{
      const saved = localStorage.getItem(PROJECT_KEY + curLayer);
      curStep = saved ? parseInt(saved, 10) : 1;
    }} catch (e) {{ curStep = 1; }}
    document.querySelectorAll('.layer-btn').forEach((b, i) => b.classList.toggle('active', i === idx));
    update();
  }}

  function speakPin(targetPin) {{
    if (!voiceEnabled || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(TXT.speech_prefix + targetPin);
    u.lang = TXT.speech_lang;
    u.rate = 1.15;
    window.speechSynthesis.speak(u);
  }}

  function update() {{
    const path = getPath();
    const total = getTotal();
    if (curStep > total) curStep = total;
    if (curStep < 1) curStep = 1;

    slider.max = total;
    slider.value = curStep;
    jumpInput.max = total;
    maxStepDisp.textContent = total;

    const fromP = path[curStep - 1] ?? 0;
    const toP = path[curStep] ?? 0;
    pinInfo.innerHTML = fromP + ' &rarr; ' + toP;
    stepCounter.textContent = curStep + ' / ' + total;
    percentInfo.textContent = ((curStep / total) * 100).toFixed(1) + '%';
    layerName.textContent = TXT.lbl_layer_prefix + LAYERS[curLayer].name;

    const cum = CUM_LEN[curLayer];
    const curLen = cum[Math.min(curStep, cum.length - 1)] || 0;
    const totalLen = cum[cum.length - 1] || 0;
    lengthInfo.textContent = curLen.toFixed(1) + TXT.unit_m;
    remainingInfo.textContent = TXT.lbl_rem + Math.max(0, totalLen - curLen).toFixed(1) + TXT.unit_m;

    ctx.globalAlpha = 1.0;
    ctx.fillStyle = BOARD_BG;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const scale = canvas.width / SRC_SIZE;

    for (let l = 0; l < curLayer; l++) {{
      const lp = LAYERS[l].path;
      ctx.strokeStyle = LAYERS[l].color || '#111111';
      ctx.lineWidth = 1.2 * scale;
      ctx.globalAlpha = 0.28;
      ctx.beginPath();
      for (let i = 0; i < lp.length - 1; i++) {{
        const a = PINS[lp[i]], b = PINS[lp[i + 1]];
        ctx.moveTo(a[0] * scale, a[1] * scale);
        ctx.lineTo(b[0] * scale, b[1] * scale);
      }}
      ctx.stroke();
    }}

    ctx.strokeStyle = LAYERS[curLayer].color || '#111111';
    ctx.lineWidth = 1.3 * scale;
    ctx.globalAlpha = 0.35;
    ctx.beginPath();
    for (let i = 0; i < curStep; i++) {{
      const a = PINS[path[i]], b = PINS[path[i + 1]];
      ctx.moveTo(a[0] * scale, a[1] * scale);
      ctx.lineTo(b[0] * scale, b[1] * scale);
    }}
    ctx.stroke();

    ctx.globalAlpha = 0.85;
    ctx.fillStyle = PIN_COLOR;
    for (let i = 0; i < PINS.length; i++) {{
      ctx.fillRect(PINS[i][0] * scale - 1.2, PINS[i][1] * scale - 1.2, 2.4, 2.4);
    }}

    const a = PINS[fromP], b = PINS[toP];
    if (a && b) {{
      ctx.globalAlpha = 1.0;
      ctx.strokeStyle = '#00b8d9';
      ctx.lineWidth = 3.6 * scale;
      ctx.shadowColor = '#00b8d9';
      ctx.shadowBlur = 10 * scale;
      ctx.beginPath();
      ctx.moveTo(a[0] * scale, a[1] * scale);
      ctx.lineTo(b[0] * scale, b[1] * scale);
      ctx.stroke();
      ctx.shadowBlur = 0;

      ctx.fillStyle = '#cf222e';
      ctx.beginPath(); ctx.arc(a[0] * scale, a[1] * scale, 5.5 * scale, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#0969da';
      ctx.beginPath(); ctx.arc(b[0] * scale, b[1] * scale, 7.0 * scale, 0, Math.PI * 2); ctx.fill();
    }}

    try {{ localStorage.setItem(PROJECT_KEY + curLayer, String(curStep)); }} catch (e) {{}}
  }}

  function next(d = 1) {{
    const prev = curStep;
    curStep = Math.min(getTotal(), curStep + d);
    if (curStep !== prev) {{
      speakPin(getPath()[curStep]);
      update();
    }}
  }}

  function prev(d = 1) {{
    curStep = Math.max(1, curStep - d);
    update();
  }}

  function restartAutoPlay() {{
    if (!isPlaying) return;
    clearInterval(playInterval);
    const delay = parseInt(speedSelect.value, 10) || 2500;
    playInterval = setInterval(() => {{
      if (curStep >= getTotal()) {{ btnPlay.click(); return; }}
      next(1);
    }}, delay);
  }}

  slider.oninput = (e) => {{ curStep = parseInt(e.target.value, 10) || 1; update(); }};
  document.getElementById('btn-next').onclick = () => next(1);
  document.getElementById('btn-prev').onclick = () => prev(1);
  document.getElementById('btn-next-10').onclick = () => next(10);
  document.getElementById('btn-prev-10').onclick = () => prev(10);
  document.getElementById('btn-reset').onclick = () => {{ curStep = 1; update(); }};
  document.getElementById('btn-jump').onclick = () => {{
    const v = parseInt(jumpInput.value, 10);
    if (!Number.isNaN(v)) {{
      curStep = Math.max(1, Math.min(getTotal(), v));
      update();
    }}
  }};
  jumpInput.addEventListener('keydown', (e) => {{
    if (e.key === 'Enter') document.getElementById('btn-jump').click();
  }});
  speedSelect.onchange = restartAutoPlay;

  btnVoice.onclick = () => {{
    voiceEnabled = !voiceEnabled;
    btnVoice.textContent = voiceEnabled ? TXT.voice_on : TXT.voice_off;
  }};

  btnTheme.onclick = () => {{
    const isDark = document.body.classList.toggle('dark-ui');
    btnTheme.textContent = isDark ? TXT.theme_light : TXT.theme_dark;
  }};

  btnPlay.onclick = () => {{
    isPlaying = !isPlaying;
    if (isPlaying) {{
      btnPlay.textContent = TXT.btn_pause;
      restartAutoPlay();
    }} else {{
      btnPlay.textContent = TXT.btn_play_js;
      clearInterval(playInterval);
    }}
  }};

  window.addEventListener('keydown', (e) => {{
    const tag = (e.target && e.target.tagName) ? e.target.tagName.toUpperCase() : '';
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'd' || e.key === 'D') {{
      e.preventDefault(); next(1);
    }} else if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {{
      e.preventDefault(); prev(1);
    }}
  }});

  try {{
    const saved = localStorage.getItem(PROJECT_KEY + curLayer);
    if (saved) curStep = parseInt(saved, 10) || 1;
  }} catch (e) {{}}

  update();
</script>
</body>
</html>"""
    return html


def export_html_assistant(
    filepath: str,
    title: str,
    canvas_manager: CanvasManager,
    layers: List[Dict[str, Any]],
    invert_board: bool = False,
    lang: str = "ru",
) -> None:
    html = build_interactive_html5_assistant(title, canvas_manager, layers, invert_board=invert_board, lang=lang)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)


def save_project_json(
    filepath: str,
    config: Dict[str, Any],
    layers: List[Dict[str, Any]],
    active_layer_idx: int = 0,
    step_scrubber: int = 1,
    image_path: Optional[str] = None,
) -> None:
    """Сохраняет проект String Art Studio (.sart.json): настройки, все слои и прогресс плеера."""
    clean_layers = []
    for layer in layers:
        item: Dict[str, Any] = {
            "name": str(layer.get("name", "Основной")),
            "color_hex": str(layer.get("color_hex", "#111111")),
            "path": [int(p) for p in layer.get("path", [])],
        }
        if "ref_lines" in layer and layer["ref_lines"] is not None:
            item["ref_lines"] = int(layer["ref_lines"])
        clean_layers.append(item)

    payload = {
        "format": "string_art_studio_project",
        "version": 1,
        "app": APP_TITLE,
        "image_path": image_path or "",
        "active_layer_idx": int(active_layer_idx),
        "step_scrubber": int(step_scrubber),
        "config": dict(config),
        "layers": clean_layers,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_project_json(filepath: str) -> Dict[str, Any]:
    """Загружает и валидирует файл проекта String Art Studio (.sart.json)."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or data.get("format") != "string_art_studio_project":
        raise ValueError("Неподдерживаемый формат файла проекта (.sart.json).")

    raw_layers = data.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ValueError("Файл проекта не содержит слоев нитей.")

    layers: List[Dict[str, Any]] = []
    for idx, layer in enumerate(raw_layers):
        if not isinstance(layer, dict) or not isinstance(layer.get("path"), list):
            raise ValueError(f"Поврежден слой #{idx + 1} в файле проекта.")
        item = {
            "name": str(layer.get("name", f"Слой {idx + 1}")),
            "color_hex": str(layer.get("color_hex", "#111111")),
            "path": [int(p) for p in layer["path"]],
        }
        if "ref_lines" in layer and layer["ref_lines"] is not None:
            item["ref_lines"] = int(layer["ref_lines"])
        layers.append(item)

    config = data.get("config")
    if not isinstance(config, dict):
        config = {}

    return {
        "format": data["format"],
        "version": int(data.get("version", 1)),
        "app": str(data.get("app", APP_TITLE)),
        "image_path": str(data.get("image_path", "")),
        "active_layer_idx": max(0, min(len(layers) - 1, int(data.get("active_layer_idx", 0)))),
        "step_scrubber": max(1, int(data.get("step_scrubber", 1))),
        "config": config,
        "layers": layers,
    }
