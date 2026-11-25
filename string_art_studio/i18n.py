# -*- coding: utf-8 -*-
"""
Локализация String Art Studio Pro (Русский / English) с автоопределением языка ОС
(если язык системы русский — интерфейс на русском, иначе — на английском)
и сохранением ручного выбора пользователя.
"""

import json
import locale
import os
import sys
from typing import Dict, List, Optional

SUPPORTED_LANGUAGES = ("ru", "en")
LANGUAGE_LABELS = {"ru": "Русский (RU)", "en": "English (EN)"}
_PREFS_FILE = os.path.join(os.path.expanduser("~"), ".stringart_studio_lang.json")

# ----------------------------------------------------------------------
# КАНОНИЧЕСКИЕ ЗНАЧЕНИЯ <-> ОТОБРАЖАЕМЫЕ НАЗВАНИЯ
# ----------------------------------------------------------------------

SHAPE_EN: Dict[str, str] = {
    "Круг": "Circle",
    "Квадрат": "Square",
    "Прямоугольник (4:3)": "Rectangle (4:3)",
    "Прямоугольник (16:9)": "Rectangle (16:9)",
    "Овал": "Oval",
    "Шестиугольник": "Hexagon",
}
_SHAPE_TO_CANON: Dict[str, str] = {
    **{k: k for k in SHAPE_EN},
    **{v: k for k, v in SHAPE_EN.items()},
}

COLOR_MODE_EN: Dict[str, str] = {
    "Монохром (Черная нить)": "Monochrome (Black Thread)",
    "Белая нить (Темная доска)": "White Thread (Dark Board)",
    "Цветной CMYK (4 нити)": "Color CMYK (4 Threads)",
    "Адаптивный подбор (N цветов)": "Adaptive Color Matching (N colors)",
}
_COLOR_MODE_TO_CANON: Dict[str, str] = {
    **{k: k for k in COLOR_MODE_EN},
    **{v: k for k, v in COLOR_MODE_EN.items()},
}

METRIC_EN: Dict[str, str] = {
    "Преобразование расстояний": "Distance Transform",
    "Контуры": "Contours",
}
_METRIC_TO_CANON: Dict[str, str] = {
    **{k: k for k in METRIC_EN},
    **{v: k for k, v in METRIC_EN.items()},
}

PRESET_EN: Dict[str, str] = {
    "Авто (по изображению)": "Auto (by Image)",
    "Портрет (Ultra Detail)": "Portrait (Ultra Detail)",
    "Белая нить на черном": "White Thread on Dark",
    "Пейзаж (High Contrast)": "Landscape (High Contrast)",
    "Минимализм (900 нитей)": "Minimalism (900 threads)",
    "Цветной CMYK (4 нити)": "Color CMYK (4 Threads)",
    "Ультра-детализация (3000)": "Ultra Detail (3000)",
    "Быстрый тест (600 нитей)": "Quick Test (600 threads)",
}
_PRESET_TO_CANON: Dict[str, str] = {
    **{k: k for k in PRESET_EN},
    **{v: k for k, v in PRESET_EN.items()},
}

LAYER_NAME_EN: Dict[str, str] = {
    "Черная нить": "Black Thread",
    "Белая нить": "White Thread",
    "Циан (Cyan)": "Cyan",
    "Маджента (Magenta)": "Magenta",
    "Желтый (Yellow)": "Yellow",
    "Черный (Black)": "Black",
    "Основной": "Main",
}
_LAYER_NAME_RU: Dict[str, str] = {v: k for k, v in LAYER_NAME_EN.items()}


def normalize_lang(lang: Optional[str]) -> str:
    if not lang:
        return "en"
    low = str(lang).strip().lower()
    if low.startswith("ru"):
        return "ru"
    return "en"


def load_saved_language() -> Optional[str]:
    try:
        if os.path.exists(_PREFS_FILE):
            with open(_PREFS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            val = data.get("lang")
            if val in SUPPORTED_LANGUAGES:
                return val
    except Exception:
        pass
    return None


def save_language_preference(lang: str) -> None:
    norm = normalize_lang(lang)
    try:
        with open(_PREFS_FILE, "w", encoding="utf-8") as f:
            json.dump({"lang": norm}, f)
    except Exception:
        pass


def detect_system_language(respect_saved: bool = True) -> str:
    """
    Определяет язык интерфейса:
    1) Переменная окружения STRINGART_LANG (если задана),
    2) Сохраненный выбор пользователя (если respect_saved=True),
    3) Язык интерфейса ОС (Windows GetUserDefaultUILanguage / POSIX locale):
       если русский -> 'ru', иначе -> 'en'.
    """
    env_override = os.environ.get("STRINGART_LANG")
    if env_override:
        return normalize_lang(env_override)

    if respect_saved:
        saved = load_saved_language()
        if saved in SUPPORTED_LANGUAGES:
            return saved

    # Windows UI Language API (точный даже без вызова locale.setlocale)
    if sys.platform == "win32":
        try:
            import ctypes

            lang_id = int(ctypes.windll.kernel32.GetUserDefaultUILanguage())
            # Primary language ID 0x19 == LANG_RUSSIAN (0x0419)
            return "ru" if (lang_id & 0x3FF) == 0x19 else "en"
        except Exception:
            pass

    # Fallback для Linux / macOS
    candidates = []
    try:
        loc = locale.getlocale()[0]
        if loc:
            candidates.append(loc)
    except Exception:
        pass
    for var in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        val = os.environ.get(var)
        if val:
            candidates.append(val)

    for cand in candidates:
        low = cand.strip().lower()
        if low.startswith("ru"):
            return "ru"
    return "en"


def to_canonical_shape(val: str) -> str:
    return _SHAPE_TO_CANON.get(val, val)


def to_display_shape(val: str, lang: str) -> str:
    canon = to_canonical_shape(val)
    return SHAPE_EN.get(canon, canon) if normalize_lang(lang) == "en" else canon


def get_display_shapes(lang: str) -> List[str]:
    return [to_display_shape(k, lang) for k in SHAPE_EN]


def to_canonical_color_mode(val: str) -> str:
    return _COLOR_MODE_TO_CANON.get(val, val)


def to_display_color_mode(val: str, lang: str) -> str:
    canon = to_canonical_color_mode(val)
    return COLOR_MODE_EN.get(canon, canon) if normalize_lang(lang) == "en" else canon


def get_display_color_modes(lang: str) -> List[str]:
    return [to_display_color_mode(k, lang) for k in COLOR_MODE_EN]


def to_canonical_metric(val: str) -> str:
    return _METRIC_TO_CANON.get(val, val)


def to_display_metric(val: str, lang: str) -> str:
    canon = to_canonical_metric(val)
    return METRIC_EN.get(canon, canon) if normalize_lang(lang) == "en" else canon


def get_display_metrics(lang: str, scipy_available: bool = True) -> List[str]:
    keys = ["Преобразование расстояний", "Контуры"] if scipy_available else ["Контуры"]
    return [to_display_metric(k, lang) for k in keys]


def to_canonical_preset(val: str) -> str:
    return _PRESET_TO_CANON.get(val, val)


def to_display_preset(val: str, lang: str) -> str:
    canon = to_canonical_preset(val)
    return PRESET_EN.get(canon, canon) if normalize_lang(lang) == "en" else canon


def get_display_presets(lang: str) -> List[str]:
    return [to_display_preset(k, lang) for k in PRESET_EN]


from .color_catalog import COLOR_NAME_EN, COLOR_NAME_RU


def translate_layer_name(name: str, lang: str) -> str:
    if normalize_lang(lang) == "en":
        return LAYER_NAME_EN.get(name, COLOR_NAME_EN.get(name, name))
    return _LAYER_NAME_RU.get(name, COLOR_NAME_RU.get(name, name))


def layer_short_code(name: str, lang: str) -> str:
    is_en = normalize_lang(lang) == "en"
    low = name.strip().lower()
    if low in ("циан (cyan)", "cyan"):
        return "C" if is_en else "Ц"
    if low in ("маджента (magenta)", "magenta"):
        return "M" if is_en else "М"
    if low in ("желтый (yellow)", "yellow"):
        return "Y" if is_en else "Ж"
    if low in ("черный (black)", "black", "черная нить", "black thread"):
        return "K" if is_en else "Ч"
    if low in ("белая нить", "white thread"):
        return "W" if is_en else "Б"
    disp = translate_layer_name(name, lang).strip()
    return disp[:3] or "L"


# ----------------------------------------------------------------------
# СЛОВАРИ ИНТЕРФЕЙСА И СООБЩЕНИЙ (RU / EN)
# ----------------------------------------------------------------------

STRINGS: Dict[str, Dict[str, str]] = {
    "ru": {
        "dashboard_frame": "Дашборд Эволюции и Аналитика Алгоритма",
        "target_frame": "Целевой шаблон (с живой предобработкой)",
        "sec_image": "1. Изображение, Проекты и Пресеты",
        "sec_geometry": "2. Геометрия, Цвет и Нить",
        "sec_algo": "3. Островной ГА (Полировка)",
        "sec_fitness": "4. Радон FBP и Дельта-Поиск",
        "sec_inspector": "5. Интерактивный Плеер и Инспектор",
        "sec_control": "6. Управление и Экспорт",
        "tab_ctrl_image": "🖼 Изображение",
        "tab_ctrl_algo": "⚙ Алгоритм",
        "tab_ctrl_player": "🎬 Плеер",
        "tab_ctrl_export": "📦 Экспорт",
        "lbl_language": "Язык / Lang:",
        "btn_load_image": "📁 Загрузить изображение (Ctrl+O)",
        "btn_save_project": "💾 Сохранить",
        "btn_load_project": "📂 Открыть проект",
        "lbl_preset": "Пресет:",
        "analysis_prompt": "Анализ: загрузите фото...",
        "analysis_preset": "Пресет: {name}",
        "btn_auto": "🔮 Авто-подбор (без каши)",
        "sl_gamma": "Светлота (Гамма)",
        "sl_clahe_clip": "Контраст (CLAHE)",
        "lbl_canvas_shape": "Форма холста:",
        "lbl_color_mode": "Режим цвета:",
        "sl_pins": "Гвозди",
        "sl_physical_diameter": "Размер (см)",
        "sl_lines": "Макс. нитей",
        "sl_line_weight": "Плотность нити (%)",
        "sl_thread_thickness": "Толщина нити",
        "sl_fearless_factor": "Защита фона (%)",
        "sl_num_islands": "Острова (потоки)",
        "sl_population_size": "Популяция",
        "sl_stages": "Этапы доводки",
        "sl_mutation_rate": "Мутация (%)",
        "sl_migration_interval": "Миграция (пок.)",
        "lbl_struct_metric": "Метрика краев:",
        "sl_contour_weight": "Вес контуров (%)",
        "chk_memetic": "Включить быстрый дельта-поиск",
        "sl_memetic_intensity": "Интенсивность (%)",
        "sl_memetic_depth": "Глубина поиска",
        "chk_radon": "Наведение Радона (Ramp FBP)",
        "sl_radon_influence": "Сила Радона (%)",
        "lbl_active_layer": "Активный слой:",
        "sl_step_scrubber": "Слой плетения",
        "player_idle": "Ход: --",
        "player_empty": "Ход: 0 нитей (цвет не требуется)",
        "player_step": "Шаг {step}/{total}: гвоздь {p1} → {p2}",
        "btn_prev": "◀ Назад",
        "btn_auto_play": "▶ Авто",
        "btn_auto_stop": "❚❚ Стоп",
        "btn_next": "Вперед ▶",
        "chk_neon": "✨ Подсветка нити",
        "chk_voice": "🔊 Голос (SAPI)",
        "sl_split_compare": "Шторка До/После %",
        "btn_start": "▶ Старт (CSR OMP + Острова)",
        "btn_polish": "✨ Доводка",
        "btn_pause": "❚❚ Пауза",
        "btn_resume": "► Продолжить",
        "btn_stop": "■ Стоп",
        "btn_prune": "✂ Оптимизировать путь (убрать лишние)",
        "btn_export_html": "🌐 Автономный веб-ассистент (.html)",
        "btn_export_txt": "📄 Схема (.txt)",
        "btn_export_csv": "📊 Таблица (.csv)",
        "btn_export_png": "🖼 Ultra HD (.png)",
        "btn_export_svg": "📐 Вектор (.svg)",
        "btn_export_template": "⭕ Шаблон 1:1 (.svg)",
        "btn_export_gcode": "⚙ ЧПУ (.gcode)",
        "status_waiting": "Ожидание запуска...",
        "fitness_idle": "Точность (Фитнес): --",
        "fitness_val": "Точность: {fitness:.5f} ({lines} нитей)",
        "engine_label": "Ядро: {mode}",
        "stat_time": "Время:",
        "stat_eta": "Осталось (ETA):",
        "stat_gens": "Поколений:",
        "stat_mig": "Миграций:",
        "stat_len": "Длина нити:",
        "eta_calc": "Расчет...",
        "len_fmt": "{m:.1f} м (на Ø{dia:.0f}см)",
        "unit_lines_short": "н.",
        "tab_overall": "★ Лучший результат",
        "tab_island": "Остров {i}",
        "tab_diff_map": "Карта Ошибок (Кр/Син)",
        "tab_density": "Плотность нитей (Heatmap)",
        "tab_saliency": "Карта Значимости (DoG)",
        "tab_radon": "Радон + FBP",
        "island_start": "Остров {i}: старт",
        "island_done": "Остров {i}: готово",
        "island_prog": "Ост.{i} Эт.{st}/{tot}: {pct}%",
        "chart_fit_title": "Рост точности (Фитнес)",
        "chart_fit_xlabel": "Поколения",
        "chart_omp_base": "OMP База",
        "chart_mut_title": "КПД операторов мутации (%)",
        "auto_rec_summary": "Детали: {comp}% | Тень: {dark}% → {pins} гвоздей, {lines} нитей",
        "dlg_warn": "Внимание",
        "dlg_info": "Информация",
        "dlg_error": "Ошибка",
        "dlg_success": "Успех",
        "dlg_exit_title": "Выход",
        "dlg_exit_msg": "Генерация еще выполняется. Остановить и выйти?",
        "dlg_crit_err": "Критическая ошибка",
        "msg_crit_start_err": "Не удалось запустить приложение:\n\n{tb}",
        "msg_load_first": "Пожалуйста, сначала загрузите изображение.",
        "msg_no_base_path": "Базовый путь еще не создан. Запускаем полный цикл OMP + Острова...",
        "msg_loaded_ok": "Изображение загружено и параметры подобраны.",
        "msg_load_err": "Не удалось загрузить изображение:\n{err}",
        "msg_synth_first": "Сначала выполните синтез картины.",
        "msg_prune_title": "Оптимизация пути",
        "msg_prune_done": "Удалено лишних нитей и петель: {rem}\nИтого четких нитей: {total}",
        "msg_prune_clean": "Лишних нитей и холостых петель не обнаружено, путь уже оптимален!",
        "msg_save_proj_title": "Сохранение проекта",
        "msg_save_proj_dlg": "Сохранить проект String Art Studio",
        "msg_save_proj_ok": "Проект успешно сохранен:\n{path}",
        "msg_save_proj_err": "Не удалось сохранить проект:\n{err}",
        "msg_load_proj_title": "Загрузка проекта",
        "msg_load_proj_stop": "Остановите текущую генерацию перед загрузкой проекта.",
        "msg_load_proj_dlg": "Открыть проект String Art Studio",
        "msg_load_proj_ok": "Проект успешно открыт:\n{path}",
        "msg_load_proj_err": "Не удалось открыть проект:\n{err}",
        "msg_proj_saved_status": "Проект сохранен: {name}",
        "msg_proj_loaded_status": "Проект загружен ({lines} нитей).",
        "msg_exp_txt_ok": "Схема гвоздей успешно сохранена:\n{path}",
        "msg_exp_csv_ok": "Таблица переходов CSV сохранена:\n{path}",
        "msg_exp_gcode_ok": "ЧПУ-программа G-Code сохранена:\n{path}",
        "msg_exp_png_bg": "Экспорт Ultra HD PNG (2400x2400px) в фоне...",
        "msg_exp_png_done_status": "Экспорт PNG 2400x2400 завершен.",
        "msg_exp_png_ok": "Изображение высокого разрешения сохранено в:\n{path}",
        "msg_exp_png_err_status": "Ошибка экспорта PNG.",
        "msg_exp_png_err": "Не удалось сохранить Ultra-HD PNG:\n{err}",
        "msg_exp_svg_ok": "Векторный файл SVG успешно сохранен:\n{path}",
        "msg_exp_tpl_ok": "Чертеж сверления гвоздей 1:1 (в мм) сохранен:\n{path}",
        "msg_exp_html_dlg": "Сохранить автономный веб-ассистент плетения",
        "msg_exp_html_ok": "Автономный веб-ассистент успешно сохранен в:\n{path}\n\nОткройте файл в любом браузере на ПК, планшете или смартфоне!",
        "msg_exp_html_err": "Не удалось экспортировать HTML:\n{err}",
        "st_prep_fc": "Подготовка фитнес-калькулятора для {n} островов...",
        "st_layer_no_threads": "Слой «{name}» не требует нитей (0 нитей).",
        "st_layer_opt_empty": "Слой «{name}» оптимален без нитей (0 нитей).",
        "st_start_islands": "Запуск {n} параллельных островов эволюции (макс. {max_l} нитей)...",
        "st_analyze": "Анализ изображения, геометрии и построение таблиц хорд...",
        "st_radon": "Вычисление томографической синограммы Радона (Ramp FBP)...",
        "st_omp_start": "Этап 1: Точный Beer-Lambert CSR OMP ({name}, макс. {max_l})...",
        "st_no_signal": "Сигнал для плетения отсутствует (0 нитей).",
        "st_stopped_omp": "Остановлено на этапе OMP.",
        "st_cmyk_budgets": "Расчет бюджетов CMYK: {info}",
        "st_cmyk_absent": "Слой {idx}/4 ({name}): цвет отсутствует — 0 нитей",
        "st_cmyk_omp": "CMYK слой {idx}/4 ({name}, бюджет ≤{bud}): CSR OMP-синтез...",
        "st_cmyk_layer_prog": "Слой {idx}/4 ({name}): {status}",
        "st_cmyk_polish": "Слой {idx}/4 ({name}): эволюционная доводка ({lines} нитей)...",
        "st_cmyk_layer_done": "Слой {name} готов ({lines} нитей)",
        "st_stopped_user": "Остановлено пользователем.",
        "st_cmyk_done": "Цветной CMYK синтез и полировка завершены!",
        "st_stopping": "Остановка потоков...",
        "st_paused": "Пауза...",
        "st_error": "Ошибка!",
        "st_done_cmyk_fmt": "{msg} ({per_layer} | Всего: {total})",
        "st_done_mono_fmt": "{msg} (Нитей: {total}, удалено петель: {rem})",
        "sl_adaptive_n_colors": "Количество цветов (N)",
        "st_adaptive_selecting": "Адаптивный подбор: выбор {n} цветов из каталога ({total} вариантов)...",
        "st_adaptive_layer": "Адаптивный слой {idx}/{n} ({name})...",
        "st_adaptive_done": "Адаптивный подбор завершен ({n} цветов)!",
        "all_colors_option": "🌈 Все цвета ({total} {unit})",
        "all_colors_step": "Все цвета: шаг {step}/{total}",
        "palette_title": "Палитра нитей:",
        "st_done_adaptive_fmt": "{msg} ({per_layer} | Всего: {total})",
        "kb_shortcuts_hint": "⌨ Пробел / D / → = Следующий шаг\n⌨ A / ← = Предыдущий шаг\n⌨ Панель управления плеером расположена под холстом ↓",
    },
    "en": {
        "dashboard_frame": "Evolution Dashboard & Algorithm Analytics",
        "target_frame": "Target Template (Live Preprocessing)",
        "sec_image": "1. Image, Projects & Presets",
        "sec_geometry": "2. Geometry, Color & Thread",
        "sec_algo": "3. Island GA (Polishing)",
        "sec_fitness": "4. Radon FBP & Delta Search",
        "sec_inspector": "5. Interactive Player & Inspector",
        "sec_control": "6. Controls & Export",
        "tab_ctrl_image": "🖼 Image",
        "tab_ctrl_algo": "⚙ Algorithm",
        "tab_ctrl_player": "🎬 Player",
        "tab_ctrl_export": "📦 Export",
        "lbl_language": "Language:",
        "btn_load_image": "📁 Load Image (Ctrl+O)",
        "btn_save_project": "💾 Save",
        "btn_load_project": "📂 Open Project",
        "lbl_preset": "Preset:",
        "analysis_prompt": "Analysis: load an image...",
        "analysis_preset": "Preset: {name}",
        "btn_auto": "🔮 Auto-Tune (No Mush)",
        "sl_gamma": "Lightness (Gamma)",
        "sl_clahe_clip": "Contrast (CLAHE)",
        "lbl_canvas_shape": "Canvas Shape:",
        "lbl_color_mode": "Color Mode:",
        "sl_pins": "Pins",
        "sl_physical_diameter": "Size (cm)",
        "sl_lines": "Max Threads",
        "sl_line_weight": "Thread Density (%)",
        "sl_thread_thickness": "Thread Thickness",
        "sl_fearless_factor": "Bg Protection (%)",
        "sl_num_islands": "Islands (Threads)",
        "sl_population_size": "Population",
        "sl_stages": "Polish Stages",
        "sl_mutation_rate": "Mutation (%)",
        "sl_migration_interval": "Migration (gens)",
        "lbl_struct_metric": "Edge Metric:",
        "sl_contour_weight": "Contour Weight (%)",
        "chk_memetic": "Enable fast delta-search",
        "sl_memetic_intensity": "Intensity (%)",
        "sl_memetic_depth": "Search Depth",
        "chk_radon": "Radon Guidance (Ramp FBP)",
        "sl_radon_influence": "Radon Strength (%)",
        "lbl_active_layer": "Active Layer:",
        "sl_step_scrubber": "Weaving Step",
        "player_idle": "Step: --",
        "player_empty": "Step: 0 threads (color not needed)",
        "player_step": "Step {step}/{total}: pin {p1} → {p2}",
        "btn_prev": "◀ Prev",
        "btn_auto_play": "▶ Auto",
        "btn_auto_stop": "❚❚ Stop",
        "btn_next": "Next ▶",
        "chk_neon": "✨ Highlight Thread",
        "chk_voice": "🔊 Voice (SAPI)",
        "sl_split_compare": "Split Before/After %",
        "btn_start": "▶ Start (CSR OMP + Islands)",
        "btn_polish": "✨ Polish",
        "btn_pause": "❚❚ Pause",
        "btn_resume": "► Resume",
        "btn_stop": "■ Stop",
        "btn_prune": "✂ Optimize Path (Prune Excess)",
        "btn_export_html": "🌐 Standalone Web Assistant (.html)",
        "btn_export_txt": "📄 Scheme (.txt)",
        "btn_export_csv": "📊 Table (.csv)",
        "btn_export_png": "🖼 Ultra HD (.png)",
        "btn_export_svg": "📐 Vector (.svg)",
        "btn_export_template": "⭕ 1:1 Template (.svg)",
        "btn_export_gcode": "⚙ CNC (.gcode)",
        "status_waiting": "Ready to start...",
        "fitness_idle": "Accuracy (Fitness): --",
        "fitness_val": "Accuracy: {fitness:.5f} ({lines} threads)",
        "engine_label": "Engine: {mode}",
        "stat_time": "Elapsed:",
        "stat_eta": "Remaining (ETA):",
        "stat_gens": "Generations:",
        "stat_mig": "Migrations:",
        "stat_len": "Thread Length:",
        "eta_calc": "Calculating...",
        "len_fmt": "{m:.1f} m (Ø{dia:.0f}cm)",
        "unit_lines_short": "thr.",
        "tab_overall": "★ Best Result",
        "tab_island": "Island {i}",
        "tab_diff_map": "Error Map (Red/Blue)",
        "tab_density": "Thread Density (Heatmap)",
        "tab_saliency": "Saliency Map (DoG)",
        "tab_radon": "Radon + FBP",
        "island_start": "Island {i}: start",
        "island_done": "Island {i}: done",
        "island_prog": "Isl.{i} St.{st}/{tot}: {pct}%",
        "chart_fit_title": "Accuracy Growth (Fitness)",
        "chart_fit_xlabel": "Generations",
        "chart_omp_base": "OMP Baseline",
        "chart_mut_title": "Mutation Operator KPI (%)",
        "auto_rec_summary": "Details: {comp}% | Shadow: {dark}% → {pins} pins, {lines} threads",
        "dlg_warn": "Warning",
        "dlg_info": "Information",
        "dlg_error": "Error",
        "dlg_success": "Success",
        "dlg_exit_title": "Exit",
        "dlg_exit_msg": "Generation is still running. Stop and exit?",
        "dlg_crit_err": "Critical Error",
        "msg_crit_start_err": "Failed to start application:\n\n{tb}",
        "msg_load_first": "Please load an image first.",
        "msg_no_base_path": "Base path does not exist yet. Starting full OMP + Islands pipeline...",
        "msg_loaded_ok": "Image loaded and parameters auto-tuned.",
        "msg_load_err": "Failed to load image:\n{err}",
        "msg_synth_first": "Please generate an artwork first.",
        "msg_prune_title": "Path Optimization",
        "msg_prune_done": "Removed excess threads & loops: {rem}\nTotal crisp threads: {total}",
        "msg_prune_clean": "No excess threads or idle loops found — path is already optimal!",
        "msg_save_proj_title": "Save Project",
        "msg_save_proj_dlg": "Save String Art Studio Project",
        "msg_save_proj_ok": "Project successfully saved:\n{path}",
        "msg_save_proj_err": "Failed to save project:\n{err}",
        "msg_load_proj_title": "Open Project",
        "msg_load_proj_stop": "Stop the current generation before opening a project.",
        "msg_load_proj_dlg": "Open String Art Studio Project",
        "msg_load_proj_ok": "Project successfully opened:\n{path}",
        "msg_load_proj_err": "Failed to open project:\n{err}",
        "msg_proj_saved_status": "Project saved: {name}",
        "msg_proj_loaded_status": "Project loaded ({lines} threads).",
        "msg_exp_txt_ok": "Pin weaving scheme saved to:\n{path}",
        "msg_exp_csv_ok": "CSV step table saved to:\n{path}",
        "msg_exp_gcode_ok": "CNC G-Code program saved to:\n{path}",
        "msg_exp_png_bg": "Exporting Ultra HD PNG (2400x2400px) in background...",
        "msg_exp_png_done_status": "PNG 2400x2400 export complete.",
        "msg_exp_png_ok": "Ultra-HD image saved to:\n{path}",
        "msg_exp_png_err_status": "PNG export error.",
        "msg_exp_png_err": "Failed to export Ultra-HD PNG:\n{err}",
        "msg_exp_svg_ok": "SVG vector drawing saved to:\n{path}",
        "msg_exp_tpl_ok": "1:1 millimeter drill template saved to:\n{path}",
        "msg_exp_html_dlg": "Save Standalone HTML5 Weaving Assistant",
        "msg_exp_html_ok": "Standalone web assistant saved to:\n{path}\n\nOpen it in any browser on PC, tablet, or smartphone!",
        "msg_exp_html_err": "Failed to export HTML:\n{err}",
        "st_prep_fc": "Preparing fitness calculator for {n} islands...",
        "st_layer_no_threads": "Layer '{name}' requires no threads (0 threads).",
        "st_layer_opt_empty": "Layer '{name}' is optimal with 0 threads.",
        "st_start_islands": "Starting {n} parallel evolution islands (max {max_l} threads)...",
        "st_analyze": "Analyzing image, geometry, and building chord tables...",
        "st_radon": "Computing Radon tomographic sinogram (Ramp FBP)...",
        "st_omp_start": "Stage 1: Exact Beer-Lambert CSR OMP ({name}, max {max_l})...",
        "st_no_signal": "No signal to weave (0 threads).",
        "st_stopped_omp": "Stopped during OMP stage.",
        "st_cmyk_budgets": "CMYK channel budgets: {info}",
        "st_cmyk_absent": "Layer {idx}/4 ({name}): color absent — 0 threads",
        "st_cmyk_omp": "CMYK layer {idx}/4 ({name}, budget ≤{bud}): CSR OMP synthesis...",
        "st_cmyk_layer_prog": "Layer {idx}/4 ({name}): {status}",
        "st_cmyk_polish": "Layer {idx}/4 ({name}): evolutionary polish ({lines} threads)...",
        "st_cmyk_layer_done": "Layer {name} complete ({lines} threads)",
        "st_stopped_user": "Stopped by user.",
        "st_cmyk_done": "Color CMYK synthesis and polishing complete!",
        "st_stopping": "Stopping worker threads...",
        "st_paused": "Paused...",
        "st_error": "Error!",
        "st_done_cmyk_fmt": "{msg} ({per_layer} | Total: {total})",
        "st_done_mono_fmt": "{msg} (Threads: {total}, loops pruned: {rem})",
        "sl_adaptive_n_colors": "Number of Colors (N)",
        "st_adaptive_selecting": "Adaptive matching: selecting {n} colors from catalog ({total} variants)...",
        "st_adaptive_layer": "Adaptive layer {idx}/{n} ({name})...",
        "st_adaptive_done": "Adaptive color matching done ({n} colors)!",
        "all_colors_option": "🌈 All Colors ({total} {unit})",
        "all_colors_step": "All colors: step {step}/{total}",
        "palette_title": "Thread Palette:",
        "st_done_adaptive_fmt": "{msg} ({per_layer} | Total: {total})",
        "kb_shortcuts_hint": "⌨ Space / D / → = Next step\n⌨ A / ← = Prev step\n⌨ Player controls are in the bar below ↓",
    },
}

TOOLTIPS_EN: Dict[str, str] = {
    "language": "Switch interface language (Russian / English). Auto-detected from OS language on startup.",
    "load_image": "Open an image file (JPG, PNG, WEBP, BMP).\nHigh-contrast portraits with clean backgrounds work best.",
    "save_project": "Save current project (.sart.json): settings, all thread layers, and player step (Ctrl+S).",
    "load_project": "Open a previously saved project (.sart.json) to continue weaving or polishing.",
    "preset": "Verified configuration presets for different weaving styles.",
    "clahe_clip": "Local contrast enhancement (CLAHE).\nBrings out eyes, lips, and fine details.",
    "gamma": "Gamma correction (< 1.0 lightens midtones and prevents muddy over-darkening).",
    "canvas_shape": "Frame geometry for pin placement (Circle, Square, Rectangle, Oval, Hexagon).",
    "color_mode": "Weaving mode: Monochrome (black/white thread), 4-layer CMYK, or Adaptive color matching (N colors from catalog).",
    "adaptive_n_colors": "Number of thread colors (2 to 20) automatically selected from the 1017-color catalog to match the image.",
    "physical_diameter": "Physical board size/diameter (cm) for accurate thread meterage and CNC G-Code.",
    "pins": "Number of perimeter pins.\nRecommended: 240–320.",
    "lines": "Maximum threads per layer.\nAutomatically reduces excess threads (down to 0 for absent colors) and adds up to max.",
    "line_weight": "Thread optical density (%).\nHigher values make each thread darker/stronger.",
    "thread_thickness": "Rendered & exported thread stroke thickness.",
    "fearless_factor": "Background protection against over-saturation (%).\nPrevents gray/color mush in empty areas.",
    "num_islands": "Number of independent parallel threads (islands) in the genetic algorithm.",
    "population_size": "Population size per island.",
    "stages": "Number of progressive evolutionary polishing stages.",
    "mutation_rate": "Base mutation probability (%).",
    "migration_interval": "Interval (in generations) for elite migration between islands.",
    "structural_metric": "Edge accuracy metric for guiding threads along image contours.",
    "contour_weight": "Weight of edges and facial features (%).",
    "memetic_enabled": "Enable ultra-fast O(L) delta-search polishing (replace, prune excess, insert deficit).",
    "memetic_intensity": "Fraction of top individuals polished per generation (%).",
    "memetic_depth": "Iterations per local delta-search call.",
    "radon_enabled": "Use Radon tomography (Ram-Lak Ramp filter) for global chord guidance.",
    "radon_influence": "Strength of Radon sinogram guidance (%).",
    "step_scrubber": "Interactive step-by-step weaving scrubber from 1st to last thread.",
    "active_layer": "Select layer (for color modes: CMYK / Adaptive) or 'All Colors' for full view.",
    "show_active_chord": "Highlight the active thread and target pins in neon on the preview canvas.",
    "voice_enabled": "Announce next target pin number via Windows SAPI when stepping forward.",
    "split_compare": "Before/After comparison curtain over the result (0% = threads only, 50% = half original).",
    "start": "Start full hybrid synthesis (Exact Beer-Lambert CSR OMP + Variable-Length Island GA).",
    "polish": "Run evolutionary polishing (Island GA + adaptive add/remove threads) on the current path.",
    "pause": "Pause or resume generation/evolution.",
    "stop": "Stop generation and keep the current best result.",
    "prune": "Optimize path: remove idle return loops A→B→A and excess over-darkening threads.",
    "export_txt": "Export step-by-step pin instructions (.txt) with cumulative meterage and spool count.",
    "export_csv": "Export step transition table in CSV format.",
    "export_gcode": "Export CNC machine path (.gcode) in millimeters (Cartesian Y-up).",
    "export_png": "Export Ultra-HD 2400×2400 raster image in background without freezing UI.",
    "export_svg": "Export layered vector drawing (.svg) with pin markers.",
    "export_template": "Export 1:1 millimeter drill template (.svg) with frame outline and 100 mm calibration bar.",
    "export_html": "Export standalone interactive HTML5 weaving assistant (.html) for PC/tablet/phone.",
}


def tr(key: str, lang: str = "ru", **kwargs) -> str:
    """Возвращает локализованную строку по ключу с подстановкой параметров."""
    code = normalize_lang(lang)
    table = STRINGS.get(code, STRINGS["en"])
    text = table.get(key, STRINGS["ru"].get(key, key))
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text


def get_tooltip(key: str, lang: str = "ru", ru_tooltips: Optional[Dict[str, str]] = None) -> str:
    code = normalize_lang(lang)
    if code == "en":
        return TOOLTIPS_EN.get(key, "")
    if ru_tooltips is not None and key in ru_tooltips:
        return ru_tooltips[key]
    if key == "language":
        return "Переключить язык интерфейса (Русский / English). При запуске определяется по языку системы."
    return ""
