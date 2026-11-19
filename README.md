# String Art Studio Pro v9.6 🎨🧵

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

**String Art Studio Pro v9.6** is an open-source algorithmic studio for generating high-definition **String Art** (thread art / filography / изонить) artworks from continuous thread paths — supporting **Monochrome Black Thread**, **White Thread on Dark Board**, and **4-Layer Subtractive CMYK Color**.

It combines tomography mathematics (**Radon Transform + Ram-Lak Filtered Back Projection**), a **6,700+ lines/sec Pre-Sliced DDA SciPy CSR Orthogonal Matching Pursuit (OMP)** initializer with background-zone protection ("no mush"), a multi-threaded **Island Genetic Algorithm (GA)** with `O(L)` scratch-buffer best-of-2 delta memetic polishing, **PyTorch CUDA GPU** batch fitness & Beer-Lambert optical rendering (`~7 ms` at 550px, `~60 ms` at 2400px Ultra-HD on RTX 5060, with seamless CPU fallback), and full `.sart.json` project persistence.

### 🌟 Key Features (v9.6)
* ⚡ **Ultra-Fast Hybrid Engine (DDA CSR OMP + CUDA + Island GA)**:
  * **Vectorized Integer DDA CSR (`build_line_tables`)**: Builds 29,250 chords (260 pins) in `~390 ms` with **zero duplicate pixels** per chord.
  * **Pre-Sliced CSR `pixel_gain` OMP**: Evaluates candidate chords without CSR slicing copies (`~6,700 lines/sec` end-to-end on CPU) while enforcing background protection (`white_guard`).
  * **PyTorch CUDA Batch Fitness & Rendering**: Shares a single cached sparse matrix in VRAM across CMYK layers and polish runs (`~12.6 ms` for 24 chromosomes on RTX 5060) and accelerates Beer-Lambert preview/Ultra-HD rendering by **11–12×** (`~7 ms` at 550px, `~60 ms` at 2400px), with automatic CPU fallback.
  * **Island GA & `O(L)` Scratch-Buffer Delta Search**: Independent island threads with elite migration, boundary-validated crossover (`min_pin_dist` invariant), 4 graph mutators (*Smart Hinge*, *Radon Bridge*, *2-Opt Invert*, *Jitter Pin*), and **2.4× faster** `O(L)` delta evaluation checking both Radon and local-jitter candidates per step.
* 🎨 **6 Frame Geometries & 3 Color Modes**:
  * **Shapes**: Circle, Square, Rectangle (4:3 & 16:9), Oval, Hexagon — with unified polygon masking and frame-fit image scaling.
  * **Color Modes**: **Monochrome (Black Thread)**, **White Thread (Dark Board `#121418`)**, and **4-Layer Subtractive CMYK** (Cyan, Magenta, Yellow, Black) with **per-channel Sobel contours & Radon sinograms** for crisp color separation.
* 🌀 **Radon Transform & FBP Reconstruction**:
  * Ram-Lak Ramp + Hamming frequency filtering and Filtered Back Projection (FBP) visualization tab, accurately aligned to chord `(rho, theta)` space (plus fast `compute_fbp=False` mode for secondary CMYK channels).
* 💾 **Project Persistence (`.sart.json`) & Non-Blocking UI**:
  * Save and resume full projects (`Ctrl+S` — settings, all layers, active layer, and player step).
  * Background Ultra-HD PNG rendering so the UI never freezes, plus `run_id`-isolated thread lifecycle management.
* 📊 **Live Matplotlib Analytics & Diagnostic Tabs**:
  * Real-time island convergence curves, OMP baseline, mutation operator KPI bar chart, Error Map (Red/Blue), Thread Density Heatmap, DoG Saliency Map, and Radon + FBP view.
* 🔊 **Interactive Physical Weaving Assistants**:
  * **Built-in Step Player**: Interactive scrubber, layer switcher, neon active-chord highlight, Auto-Play, Before/After split curtain, and **Windows SAPI Voice** (`"на 142"`).
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
*On Windows, you can also double-click `run_studio.bat`.*

### 🧪 Running Tests
```bash
python -m unittest discover -s tests -v
```

---

<a name="русский"></a>
## 🇷🇺 Русский

**String Art Studio Pro v9.6** — студия и высокопроизводительный алгоритмический комплекс с открытым исходным кодом для генерации картин в технике **String Art** (изонить / филография) из непрерывной нити: классические монохромные портреты, **белая нить на темной доске** и цветные **4-слойные CMYK** композиции.

### 🌟 Ключевые возможности v9.6

#### 1. ⚡ Гибридный движок: DDA CSR OMP + PyTorch CUDA + Островной ГА
* **Векторизованный целочисленный DDA CSR (`~390 мс`)**: построение 29 250 хорд (260 гвоздей) без сортировок `np.unique` и с **нулем дубликатов пикселей** в каждой хорде.
* **Преднарезанный CSR OMP (~6 700 нитей/сек на CPU)**: оценка всех допустимых хорд через преднарезанные подматрицы `pin_sub_matrices` без копирования CSR в цикле + защита фона (`white_guard`) от «каши».
* **Ускорение на GPU NVIDIA CUDA (PyTorch) для фитнеса и рендера**:
  * Общий кэш разреженной матрицы в VRAM (1 загрузка на геометрию вместо 4 в CMYK), пакетная оценка 24 хромосом за `~12.6 мс` на RTX 5060.
  * CUDA-рендер закона Бера-Ламберта: **12.5× быстрее CPU** (`~7.1 мс` на превью 550px и `~60 мс` на Ultra-HD 2400px), автоматический откат на CPU.
* **Островная эволюция (`IslandThread`) и `O(L)` дельта-спуск**: независимые потоки с миграцией, проверка `min_pin_dist` на обоих стыках кроссовера, 4 графовых мутатора и **2.4× более быстрый** дельта-поиск на `int8` scratch-буфере (оценка пары кандидатов Радон + локальный сдвиг на шаг).
* **Кнопка «✨ Доводка»**: эволюционная полировка поверх готовой нити без перезапуска OMP.

#### 2. 📐 6 геометрий холста, Темная доска и улучшенный CMYK
* **Формы рамы**: Круг, Квадрат, Прямоугольник (4:3 и 16:9), Овал, Шестиугольник — единая полигональная маска и точное вписывание изображения.
* **3 режима цвета**:
  * **Монохром (Черная нить)** на белой основе;
  * **Белая нить (Темная доска `#121418`)** с готовым пресетом и корректным инверсным превью/экспортом;
  * **Цветной CMYK (4 нити)** с **индивидуальными контурами Собеля и синограммами Радона для каждого канала** (C, M, Y, K).
* **Сохранение и загрузка проектов (`.sart.json`)**: сохраняйте настройки, все слои и текущий шаг сборки (`Ctrl+S`), чтобы продолжить позже.

#### 3. 🌀 Преобразование Радона (Ramp FBP) и аналитика
* **Точная синограмма Радона** с частотным фильтром Рама-Лака (окно Хэмминга) и визуализацией FBP.
* **Метрики краев**: «Преобразование расстояний» (`distance_transform_edt`) и «Контуры» (Sobel + DoG + FineGrained Saliency).
* **Согласованная яркость**: единая калибровка `get_line_alpha` между оптимизатором, превью и Ultra-HD PNG на любых объемах нитей (600–4500).

#### 4. 🔊 Интерактивный плеер и 7 форматов экспорта
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
│   ├── geometry.py                 # Геометрия 6 форм, DDA CSR, GPU-кэш и CUDA/CPU рендер
│   ├── preprocessing.py            # Предобработка, маска, DoG/Saliency, Dark Board, CMYK
│   ├── radon.py                    # Синограмма Радона, фильтр Рама-Лака и FBP
│   ├── engine.py                   # Pre-sliced CSR OMP, CUDA/CPU Fitness, Island GA и O(L) дельта-поиск
│   ├── audio.py                    # Голосовой диктор Windows SAPI
│   ├── exporters.py                # TXT, CSV, G-Code (Y-up), SVG, Шаблон 1:1, PNG, HTML5, .sart.json
│   ├── coordinator.py              # AppState и потокобезопасный EvolutionCoordinator (run_id, async PNG)
│   └── ui.py                       # Интерфейс (Light Theme), плеер и живые графики Matplotlib
├── tests/                          # Автоматические тесты (15 тестов unittest)
├── requirements.txt
└── run_studio.bat
```

## 📄 Лицензия
Распространяется под лицензией [MIT](LICENSE).  
Автор: **AYes (AYesRU)**.
