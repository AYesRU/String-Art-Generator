# String Art Studio Pro v9.9.0 🎨🧵

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Algorithm](https://img.shields.io/badge/Engine-DDA%20CSR%20OMP%20%2B%20Island%20GA%20%2B%20Radon%20FBP-success)](#)
[![GPU](https://img.shields.io/badge/GPU-PyTorch%20CUDA%20(Fitness%20%2B%20Render)-76b900?logo=nvidia)](#)
[![UI Theme](https://img.shields.io/badge/UI-Clean%20Light%20Theme-informational)](#)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)](#)

**[English](#english) • [Русский](#русский)**

</div>

---

<a name="english"></a>
## 🇬🇧 English

**String Art Studio Pro v9.9.0** is an open-source algorithmic studio for generating high-definition **String Art** (thread art / filography / изонить) artworks from continuous thread paths — supporting **Monochrome Black Thread**, **White Thread on Dark Board**, **Adaptive 4-Layer Subtractive CMYK Color**, and **Adaptive Color Matching (`N = 2..20` threads from a 1,017-color catalog)**.

It combines tomography mathematics (**Radon Transform + Ram-Lak Filtered Back Projection**), an **Exact Beer-Lambert Pre-Sliced DDA SciPy CSR Orthogonal Matching Pursuit (OMP)** initializer with per-channel physical thread budgets (automatically assigning **0 threads** to absent colors), a variable-length multi-threaded **Island Genetic Algorithm (GA)** with `O(L)` scratch-buffer delta thread pruning & insertion (decreasing excess threads and adding missing ones up to `max_lines`), **PyTorch CUDA GPU** batch fitness & optical rendering (`~7 ms` at 550px, `~60 ms` at 2400px Ultra-HD on RTX 5060, with seamless CPU fallback), and full `.sart.json` project persistence.

### 🌟 Key Features (v9.9.0)
* 🖥️ **Ergonomic Tabbed UI & Dedicated Player Bar**:
  * **4 Sidebar Tabs & 5 Viewport Tabs**: Organizes controls into `🖼 Image`, `⚙ Algorithm`, `🎬 Player`, and `📦 Export` tabs alongside 5 clean result & analytical tabs (`★ Best Result`, `Error Map`, `Thread Density`, `Saliency`, `Radon + FBP`).
  * **Unlocked Numeric Input & Dynamic Slider Expansion**: Manually type exact parameter values beyond slider defaults (`10–100,000` threads, `20–1,200` pins, `1–1,000` cm size, `1–20` stroke thickness) — slider bounds expand automatically to match your input.
  * **Dedicated Bottom Player Bar**: Anchors the interactive timeline scrubber (working seamlessly across single layers AND cumulative `🌈 All Colors` mode), interactive layer/swatch switcher, step counter, navigation controls (`◀`, `▶ Auto`, `▶`), and quick toggles (`✨` Neon chord, `🔊` SAPI voice, `🪞` Split compare) directly below the artwork viewports.
  * **Compact 7-Icon Export Row**: Quick one-click export bar (`[🌐] [📄] [📊] [🖼] [📐] [⭕] [⚙]`) with detailed tooltips on hover.
* 🎨 **6 Frame Geometries & 4 Color Modes**:
  * **Shapes**: Circle, Square, Rectangle (4:3 & 16:9), Oval, Hexagon — with unified polygon masking and frame-fit image scaling.
  * **Color Modes**: **Monochrome (Black Thread)**, **White Thread (Dark Board `#121418`)**, **4-Layer Subtractive CMYK** (Cyan, Magenta, Yellow, Black), and **Adaptive Color Matching (`N = 2..20` catalog threads)** using de-tinted 3-channel Beer-Lambert pigment recovery, CIE Lab k-medoids clustering, Top-2 per-pixel optical allocation, clickable palette chips, and proportional total-budget distribution.
* ⚡ **Ultra-Fast Adaptive Engine (Exact Beer-Lambert DDA CSR OMP + CUDA + Variable-Length Island GA)**:
  * **Vectorized Integer DDA CSR (`build_line_tables`)**: Builds 29,250 chords (260 pins) in `~390 ms` with **zero duplicate pixels** per chord.
  * **Exact Beer-Lambert Pre-Sliced CSR OMP**: Evaluates exact per-pixel Beer-Lambert error reduction (`~5,300–6,700 lines/sec` on CPU), stops cleanly without forced minimums, trims trailing bridge hops to the global-best prefix, and excludes 0-thread layers.
  * **Bidirectional Adaptive Thread Count (`optimize_thread_count_delta`)**: Automatically removes excess over-darkening threads (`A→B→C` → `A→C`, down to `0` if empty is cleaner) and adds deficit threads up to the configured maximum (`max_lines`).
  * **Per-Channel Budgets (`estimate_channel_line_budget`)**: Computes physical thread budgets for CMYK and Adaptive channels from each channel's Beer-Lambert optical-depth integral before synthesis, keeping identical physical thread opacity (`ref_lines`) across spools.
  * **PyTorch CUDA Batch Fitness & Rendering**: Shares a single cached sparse matrix in VRAM across color layers and polish runs (`~12.6 ms` for 24 chromosomes on RTX 5060) and accelerates Beer-Lambert preview/Ultra-HD rendering by **11–12×** (`~7 ms` at 550px, `~60 ms` at 2400px), with automatic CPU fallback.
  * **Variable-Length Island GA & `O(L)` Delta Polish**: Independent island threads with elite migration, variable-length boundary-validated crossover (`min_pin_dist` invariant), 6 graph mutators (*Smart Hinge*, *Radon Bridge*, *2-Opt Invert*, *Jitter Pin*, *Prune Excess*, *Insert Thread*), and `O(L)` scratch-buffer delta evaluation (replace / prune / insert per step).
* 🌍 **Bilingual Interface (English / Русский) with OS Auto-Detection**:
  * Automatically detects the operating system UI language on first launch (Russian OS → Russian, all other locales → English), with a live **Language switcher** in the top sidebar (no restart required; persists preference, or override via `STRINGART_LANG=en|ru`).
  * Full localization across all controls, 1,017 catalog color names, tooltips, analytics charts, Windows SAPI voice dictation (`"to 142"` / `"на 142"`), and exported files (`.txt`, `.csv`, `1:1 .svg` drill template, and interactive `.html` Web Assistant).
* 🌀 **Radon Transform & FBP Reconstruction**:
  * Ram-Lak Ramp + Hamming frequency filtering and Filtered Back Projection (FBP) visualization tab, accurately aligned to chord `(rho, theta)` space (plus fast `compute_fbp=False` mode for secondary CMYK channels).
* 💾 **Project Persistence (`.sart.json`) & Non-Blocking UI**:
  * Save and resume full projects (`Ctrl+S` — settings, all layers, active layer, and player step).
  * Background Ultra-HD PNG rendering so the UI never freezes, plus `run_id`-isolated thread lifecycle management.
* 📊 **Live Matplotlib Analytics & Diagnostic Tabs**:
  * Real-time island convergence curves, OMP baseline, mutation operator KPI bar chart, Error Map (Red/Blue), Thread Density Heatmap, DoG Saliency Map, and Radon + FBP view.
* 🔊 **Interactive Physical Weaving Assistants**:
  * **Built-in Step Player**: Interactive scrubber, layer switcher, neon active-chord highlight, Auto-Play, Before/After split curtain, and **Windows SAPI Voice** (`"to 142"` / `"на 142"`).
  * **Autonomous HTML5 Web App**: Standalone `.html` export with Web Speech API, multi-layer CMYK & Dark Board support, isolated per-project `localStorage` keys, `O(1)` precomputed meterage, jump-to-step input, autoplay speed selector, and Light/Dark UI theme toggle.
* 📦 **7 Export Formats**:
  * `.txt` step-by-step map with cumulative meters & 500m spool estimate
  * `.csv` coordinate/step table
  * `.png` Ultra-HD 2400×2400 additive Beer-Lambert raster (consistent alpha calibration with optimizer)
  * `.svg` layered vector drawing with pin markers (light & dark board support)
  * `1:1 .svg` millimeter **Drill Template** with frame outline, crosshairs, pin numbering, and a **100 mm print-scale verification bar**
  * `.gcode` CNC machine path scaled in millimeters (**Cartesian Y-up**)
  * `.html` standalone interactive Web Assistant

### 🚀 Quick Start
```bash
# Clone the repository
git clone https://github.com/AYesRU/String-Art-Generator.git
cd String-Art-Generator

# Install dependencies
pip install -r requirements.txt

# Run String Art Studio
python string-art-generator.py
```
*On Windows, you can also double-click `run_studio.bat`. Note: the prebuilt standalone `.exe` release excludes PyTorch to remain compact (~122 MB, CPU SciPy CSR + NumPy), whereas running from source with `torch` installed enables CUDA GPU acceleration.*

### 🧪 Running Tests
```bash
python -m unittest discover -s tests -v
```

---

<a name="русский"></a>
## 🇷🇺 Русский

**String Art Studio Pro v9.9.0** — студия и высокопроизводительный алгоритмический комплекс с открытым исходным кодом для генерации картин в технике **String Art** (изонить / филография) из непрерывной нити: классические монохромные портреты, **белая нить на темной доске**, **4-слойный CMYK** и **Адаптивный подбор палитры (`N = 2..20` цветов из каталога 1017 оттенков)**.

### 🌟 Ключевые возможности v9.9.0

#### 1. 🖥️ Эргономичный интерфейс с вкладками и панелью плеера
* **4 вкладки управления и 5 вкладок анализа**: вместо длинной прокрутки все настройки разбиты по вкладкам `🖼 Изображение`, `⚙ Алгоритм`, `🎬 Плеер`, `📦 Экспорт`, а область просмотра содержит 5 вкладок (`★ Лучший результат`, `Карта Ошибок`, `Плотность нитей`, `Карта Значимости`, `Радон + FBP`).
* **Свободный ввод параметров и динамическое расширение шкал**: в поля ввода можно вводить значения шире диапазона бегунка (`10–100 000` нитей, `20–1 200` гвоздей, `1–1 000` см, `1–20` толщина) — шкала ползунка автоматически расширяется под введенное число.
* **Горизонтальная панель плеера под холстом**: ползунок шагов (поддерживающий как отдельные слои, так и сквозной просмотр в режиме `🌈 Все цвета`), переключатель слоёв, счетчик шагов, навигация (`◀`, `▶ Авто`, `▶`), неоновая подсветка (`✨`), диктор SAPI (`🔊`) и шторка сравнения (`🪞`).
* **Компактная линейка экспорта в 1 клик**: аккуратный ряд из 7 кнопок-иконок (`[🌐] [📄] [📊] [🖼] [📐] [⭕] [⚙]`) с подробными всплывающими подсказками.

#### 2. ⚡ Гибридный адаптивный движок: Beer-Lambert DDA CSR OMP + PyTorch CUDA + Островной ГА переменной длины
* **Векторизованный целочисленный DDA CSR (`~390 мс`)**: построение 29 250 хорд (260 гвоздей) без сортировок `np.unique` и с **нулем дубликатов пикселей** в каждой хорде.
* **Точный Beer-Lambert CSR OMP**: вычисляет точный выигрыш оптической модели по преднарезанным подматрицам `pin_sub_matrices` без принудительного минимума нитей, обрезает холостой хвост до лучшего префикса и ставит **0 нитей** для отсутствующих цветов.
* **Индивидуальный расчет бюджетов (`estimate_channel_line_budget`)**: на старте рассчитывает физический бюджет нитей для каждого слоя CMYK и пропорционально распределяет общий лимит проекта в адаптивном режиме, сохраняя единую оптическую плотность нити (`ref_lines`).
* **Двунаправленная адаптация числа нитей (`optimize_thread_count_delta`)**: автоматически **уменьшает** число нитей при их избытке (схлопывание `A→B→C` → `A→C`, удаление краевых хорд) и **добавляет** недостающие нити при нехватке, строго не превышая заданный максимум (`max_lines`).
* **Островная эволюция переменной длины (`IslandThread`)**: независимые потоки с миграцией, кроссовером разной длины, **6 графовыми мутаторами** (*Smart Hinge*, *Radon Bridge*, *2-Opt Invert*, *Jitter Pin*, *Prune Excess*, *Insert Thread*) и `O(L)` дельта-спуском на `int8` scratch-буфере (замена, удаление и вставка нитей на каждом шаге).
* **Ускорение на GPU NVIDIA CUDA (PyTorch) для фитнеса и рендера**: общий кэш матрицы в VRAM (`~12.6 мс` на 24 хромосомы на RTX 5060) и CUDA-рендер Бера-Ламберта в **11–12× быстрее CPU** (`~7.1 мс` при 550px и `~60 мс` при 2400px).

#### 3. 🌍 Двуязычный интерфейс (Русский / English) с автоопределением языка ОС
* **Автовыбор языка при запуске**: если язык системы русский — интерфейс открывается на русском, на всех остальных системах — на английском (можно принудительно задать через `STRINGART_LANG=ru|en`).
* **Мгновенное переключение в интерфейсе**: переключатель `Язык / Lang` в верхней панели переводит все вкладки, все **1017 названий цветов каталога**, подсказки, графики, голосового диктора Windows SAPI (`"на 142"` / `"to 142"`) и экспортируемые файлы (`.txt`, `.csv`, шаблон `1:1 .svg`, HTML5 веб-ассистент) без перезапуска.

#### 4. 📐 6 геометрий холста, Темная доска, CMYK и Адаптивный подбор цветов
* **Формы рамы**: Круг, Квадрат, Прямоугольник (4:3 и 16:9), Овал, Шестиугольник — единая полигональная маска и точное вписывание изображения.
* **4 режима цвета**:
  * **Монохром (Черная нить)** на белой основе;
  * **Белая нить (Темная доска `#121418`)** с готовым пресетом и корректным инверсным превью/экспортом;
  * **Цветной CMYK (4 нити)** с **индивидуальными контурами Собеля и синограммами Радона для каждого канала** (C, M, Y, K);
  * **Адаптивный подбор (`N = 2..20` цветов из каталога 1017 оттенков)**: 3-канальная декомпозиция Бера-Ламберта с восстановлением собственного цвета пигмента (`de-tinted ink color`), `k-medoids` кластеризацией в CIE Lab, интерактивными чипами палитры и полноцветным субтрактивным рендером.
* **Сохранение и загрузка проектов (`.sart.json`)**: сохраняйте настройки, все слои, палитру и текущий шаг сборки (`Ctrl+S`), чтобы продолжить позже.

#### 5. 🌀 Преобразование Радона (Ramp FBP) и аналитика
* **Точная синограмма Радона** с частотным фильтром Рама-Лака (окно Хэмминга) и визуализацией FBP.
* **Метрики краев**: «Преобразование расстояний» (`distance_transform_edt`) и «Контуры» (Sobel + DoG + FineGrained Saliency).
* **Согласованная яркость**: единая калибровка `get_line_alpha` между оптимизатором, превью и Ultra-HD PNG на любых объемах нитей (`200–10 000+`).

#### 6. 🔊 Интерактивный плеер и 7 форматов экспорта
* **Студийный плеер**: ползунок шагов, `◀ Назад` / `▶ Авто` / `Вперед ▶`, переключение слоев CMYK, неоновая подсветка активной нити, шторка «До / После» и **голосовая диктовка через Windows SAPI**.
* **Полный экспорт**:
  * **.txt** — пошаговая схема с кумулятивным метражом и расчетом катушек (500м);
  * **.csv** — таблица переходов с длинами сегментов;
  * **.png (Ultra HD 2400×2400)** — фоновый рендер без зависания интерфейса;
  * **.svg** — многослойный вектор (светлая и темная доска);
  * **Шаблон 1:1 (.svg)** — миллиметровый чертеж разметки с контуром рамы, нумерацией гвоздей и **контрольной шкалой 100 мм** для проверки масштаба печати;
  * **.gcode** — программа для ЧПУ-станка в миллиметрах в стандартной декартовой системе (**ось Y вверх**);
  * **.html** — автономный веб-ассистент с Web Speech API, изолированным `localStorage` на каждый проект, переходом к номеру шага, выбором скорости автоплея и темной темой интерфейса.

---

## ⌨ Горячие клавиши

| Клавиша | Действие |
|---|---|
| `Пробел` / `D` / `→` | Следующий шаг в плеере |
| `A` / `←` | Предыдущий шаг в плеере |
| `Ctrl + O` | Открыть изображение |
| `Ctrl + S` | Сохранить проект (`.sart.json`) |

---

## 📁 Структура проекта
```
StringArt_Studio/
├── string-art-generator.py         # Точка входа приложения
├── string_art_studio/
│   ├── i18n.py                     # Двуязычная локализация (RU/EN) и автоопределение языка ОС
│   ├── geometry.py                 # Геометрия 6 форм, DDA CSR, GPU-кэш и CUDA/CPU рендер
│   ├── preprocessing.py            # Предобработка, маска, DoG/Saliency, Dark Board, CMYK
│   ├── radon.py                    # Синограмма Радона, фильтр Рама-Лака и FBP
│   ├── engine.py                   # Pre-sliced CSR OMP, CUDA/CPU Fitness, Island GA и O(L) дельта-поиск
│   ├── audio.py                    # Голосовой диктор Windows SAPI (RU/EN)
│   ├── exporters.py                # TXT, CSV, G-Code (Y-up), SVG, Шаблон 1:1, PNG, HTML5, .sart.json
│   ├── coordinator.py              # AppState и потокобезопасный EvolutionCoordinator (run_id, async PNG)
│   └── ui.py                       # Интерфейс (Light Theme), плеер, живые графики и переключатель языка
├── tests/                          # Автоматические тесты (23 теста unittest)
├── requirements.txt
└── run_studio.bat
```

## 📄 Лицензия
Распространяется под лицензией [MIT](LICENSE).  
Автор: **AYes (AYesRU)**.
