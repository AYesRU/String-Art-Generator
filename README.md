# String Art Studio Pro v9.5 🎨🧵

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Algorithm](https://img.shields.io/badge/Engine-CSR%20OMP%20%2B%20Island%20GA%20%2B%20Radon%20FBP-success)](#)
[![GPU](https://img.shields.io/badge/GPU-PyTorch%20CUDA%20%2B%20SciPy%20CSR-76b900?logo=nvidia)](#)
[![UI Theme](https://img.shields.io/badge/UI-Clean%20Light%20Theme-informational)](#)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)](#)

**[English](#english) • [Русский](#русский)**

</div>

---

<a name="english"></a>
## 🇬🇧 English

**String Art Studio Pro v9.5** is an open-source algorithmic studio for generating high-definition **String Art** (thread art / filography / изонить) artworks from continuous thread paths — supporting both **Monochrome Black Thread** and **4-Layer Subtractive CMYK Color**.

It combines tomography mathematics (**Radon Transform + Ram-Lak Filtered Back Projection**), a **3,000+ lines/sec SciPy CSR Orthogonal Matching Pursuit (OMP)** initializer with white-zone protection ("no mush"), a multi-threaded **Island Genetic Algorithm (GA)** with vectorized delta-coordinate memetic polishing, **PyTorch CUDA GPU** batch fitness evaluation (with seamless CPU fallback), and an additive **Beer-Lambert optical renderer** calibrated consistently from preview up to 2400×2400 Ultra-HD PNG.

### 🌟 Key Features
* ⚡ **Ultra-Fast Hybrid Engine (CSR OMP + CUDA + Island GA)**:
  * **CSR `pixel_gain` OMP**: Evaluates all candidate chords via sparse matrix-vector multiplication (`~3,200+ lines/sec` on CPU) while enforcing white-zone protection (`white_guard`).
  * **PyTorch CUDA Batch Fitness**: Evaluates full island populations in parallel on NVIDIA GPUs (`~15 ms` for 24 chromosomes × 1,400 chords on RTX 5060) with automatic CPU fallback.
  * **Island GA & Vectorized Delta-Search**: Independent island threads with elite migration and graph mutations (*Smart Hinge*, *Radon Bridge*, *2-Opt Invert*, *Jitter Pin*) plus vectorized coordinate-descent polishing.
* 🎨 **6 Frame Geometries & CMYK Color**:
  * **Shapes**: Circle, Square, Rectangle (4:3 & 16:9), Oval, Hexagon — with unified polygon masking and frame-fit image scaling.
  * **Color Modes**: Classic **Monochrome** or **4-Layer Subtractive CMYK** (Cyan, Magenta, Yellow, Black).
* 🌀 **Radon Transform & FBP Reconstruction**:
  * Ram-Lak Ramp + Hamming frequency filtering and Filtered Back Projection (FBP) visualization tab, accurately aligned to chord `(rho, theta)` space.
* 📊 **Live Matplotlib Analytics & Diagnostic Tabs**:
  * Real-time island convergence curves, OMP baseline, mutation operator KPI bar chart, Error Map (Red/Blue), Thread Density Heatmap, DoG Saliency Map, and Radon + FBP view.
* 🔊 **Interactive Physical Weaving Assistants**:
  * **Built-in Step Player**: Interactive scrubber, layer switcher, neon active-chord highlight, Auto-Play, Before/After split curtain, and **Windows SAPI Voice** (`"на 142"`).
  * **Autonomous HTML5 Web App**: Standalone `.html` export with Web Speech API, multi-layer CMYK tabs, real-time meterage, keyboard controls, and `localStorage` progress persistence.
* 📦 **7 Export Formats**:
  * `.txt` step-by-step map with cumulative meters & 500m spool estimate
  * `.csv` coordinate/step table
  * `.png` Ultra-HD 2400×2400 additive Beer-Lambert raster
  * `.svg` layered vector drawing with pin markers
  * `1:1 .svg` millimeter **Drill Template** with crosshairs and pin numbering
  * `.gcode` CNC machine path scaled in millimeters
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

**String Art Studio Pro v9.5** — студия и высокопроизводительный алгоритмический комплекс с открытым исходным кодом для генерации картин в технике **String Art** (изонить / филография) из непрерывной нити: как классических монохромных портретов, так и цветных **4-слойных CMYK** композиций.

### 🌟 Ключевые возможности v9.5

#### 1. ⚡ Гибридный движок: CSR OMP + PyTorch CUDA + Островной ГА
* **Сверхбыстрый CSR OMP (~3 200+ нитей/сек на CPU)**: оценка всех допустимых хорд выполняется через разреженное матричное умножение `SciPy CSR` по вектору `pixel_gain` с жесткой защитой светлых тонов (`white_guard`) от серой «каши».
* **Ускорение на GPU NVIDIA CUDA (PyTorch)**: пакетная оценка популяций хромосом на видеокарте (`~15 мс` на 24 хромосомы по 1 400 хорд на RTX 5060) с автоматическим откатом на CPU.
* **Островная эволюция (`IslandThread`)**: независимые потоки с элитной миграцией, графовыми мутаторами (*Smart Hinge*, *Radon Bridge*, *2-Opt Invert*, *Jitter Pin*) и быстрым векторизованным дельта-спуском без зависимости от `numba`.
* **Отдельная кнопка «✨ Доводка»**: запуск островной эволюции поверх уже сплетенного пути без перезапуска OMP.

#### 2. 📐 6 геометрий холста и 4-слойный CMYK
* **Формы рамы**: Круг, Квадрат, Прямоугольник (4:3 и 16:9), Овал, Шестиугольник — с единой полигональной маской и точным вписыванием изображения в габариты рамы.
* **Режимы цвета**: Монохром (черная нить) и Цветной CMYK (Циан, Маджента, Желтый, Черный) с физическим субтрактивным смешением по закону Бера-Ламберта.
* **Пресеты и Авто-подбор**: готовые профили качества + интеллектуальный подбор параметров под контраст и детализацию изображения.

#### 3. 🌀 Преобразование Радона (Ramp FBP) и аналитика
* **Точная томографическая синограмма Радона** с фильтром Рама-Лака (окно Хэмминга) и визуализацией реконструкции FBP.
* **Реальная метрика краев**: поддержка «Преобразование расстояний» (`distance_transform_edt`) и «Контуры» (Sobel + DoG + FineGrained Saliency).
* **Живые графики Matplotlib**: кривая роста точности (Фитнес) с базовым уровнем OMP и КПД каждого оператора мутации.
* **Разрешение-независимый рендер**: единая калибровка оптической плотности между превью (550px) и Ultra-HD экспортом (2400px).

#### 4. 🔊 Интерактивный плеер и 7 форматов экспорта
* **Студийный плеер**: ползунок шагов, кнопки `◀ Назад` / `▶ Авто` / `Вперед ▶`, переключение слоев CMYK, неоновая подсветка активной нити, шторка сравнения «До / После» и **голосовая диктовка через Windows SAPI**.
* **Полный экспорт**:
  * **.txt** — пошаговая схема с кумулятивным расходом нити в метрах и числом катушек (500м);
  * **.csv** — таблица переходов с длинами сегментов;
  * **.png (Ultra HD 2400×2400)** — аддитивный рендер по закону Бера-Ламберта;
  * **.svg** — масштабируемый векторный чертеж по слоям;
  * **Шаблон 1:1 (.svg)** — миллиметровый чертеж разметки и сверления гвоздей;
  * **.gcode** — программа для ЧПУ-станка в миллиметрах;
  * **.html** — автономный веб-ассистент для планшета/телефона с Web Speech API и сохранением прогресса в `localStorage`.

---

## ⌨ Горячие клавиши

| Клавиша | Действие |
|---|---|
| `Пробел` / `D` / `→` | Следующий шаг в плеере |
| `A` / `←` | Предыдущий шаг в плеере |
| `Ctrl + O` | Открыть изображение |

---

## 📁 Структура проекта
```
StringArt_Studio/
├── string-art-generator.py         # Точка входа приложения
├── string_art_studio/
│   ├── geometry.py                 # Геометрия 6 форм, CSR-матрица, аддитивный рендер
│   ├── preprocessing.py            # Предобработка, маска, DoG/Saliency, CMYK, авто-подбор
│   ├── radon.py                    # Синограмма Радона, фильтр Рама-Лака и FBP
│   ├── engine.py                   # CSR OMP, CUDA/CPU Fitness, Island GA и дельта-поиск
│   ├── audio.py                    # Голосовой диктор Windows SAPI
│   ├── exporters.py                # TXT, CSV, G-Code, SVG, Шаблон 1:1, PNG, HTML5
│   └── ui.py                       # Интерфейс (Light Theme), плеер, графики и координатор
├── tests/                          # Автоматические тесты (unittest)
├── requirements.txt
└── run_studio.bat
```

## 📄 Лицензия
Распространяется под лицензией [MIT](LICENSE).  
Автор: **AYes (AYesRU)**.
