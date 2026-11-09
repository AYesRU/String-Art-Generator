# String Art Studio Pro v9.0 🎨🧵

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Algorithm](https://img.shields.io/badge/Algorithm-OMP%20%2B%20Island%20GA%20%2B%20Radon%20FBP-success)](#)
[![UI Theme](https://img.shields.io/badge/UI-Clean%20Light%20Theme-informational)](#)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)](#)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/AYesRU/String-Art-Generator/pulls)

**[English](#english) • [Русский](#русский)**

</div>

---

<a name="english"></a>
## 🇬🇧 English

**String Art Studio Pro v9.0** is an algorithmic suite and creative studio for generating high-definition **String Art** (thread art / filography / изонить) portraits and complex geometric artworks from continuous thread lines.

By bridging tomography mathematics (**Radon Transform & Filtered Back Projection**), **Orthogonal Matching Pursuit (OMP)**, multi-threaded **Island Genetic Algorithm (GA)**, and physical optical modeling based on the **Beer-Lambert law**, it produces crisp, high-contrast string compositions without noisy artifacts or over-darkened regions ("no mush").

### 🌟 Key Features
* 🏝 **Multi-Threaded Island Genetic Algorithm**: Independent evolutionary threads with elite migration, graph mutations (*Smart Hinge*, *Radon Bridge*, *2-Opt Invert*, *Jitter Pin*), and coordinate-descent memetic polishing.
* 📊 **Live Matplotlib Convergence & Mutation Analytics**: Real-time fitness curves for all islands with OMP baselines and horizontal KPI bar charts measuring mutation operator efficiency.
* 👁 **Real-Time Visual Synthesis**: Progressive rendering on canvas during OMP synthesis (every 25 lines) and live evolutionary updates with frame-dropping protection.
* ✨ **Dedicated "Evolution Polish" Mode**: Continue refining existing thread paths via Island GA without recomputing OMP from scratch.
* 🌀 **Radon Transform & Ram-Lak FBP Filter**: Guides global thread placement to preserve delicate facial features, eyes, and sharp contours.
* 🔊 **Physical Weaving Assistants**:
  * **Windows SAPI Voice**: Hands-free voice instructions announcing nail target numbers in real time (*"to 142"*, *"to 89"*).
  * **Interactive Step Scrubber**: Step-by-step slider with neon highlights and "Before / After" split curtain.
  * **Autonomous HTML5 Web App**: Standalone `.html` file export with Web Speech API, keyboard controls, and local progress storage for tablets & mobile devices.
* 📦 **Comprehensive Export Options**: Printable Ultra-HD `.png` (up to 2400x2400 with 4x SSAA), vector `.svg`, numbered sequence `.txt` with exact thread length calculations in meters.
* ☀️ **Modern Light Theme**: Clean `#f6f8fa` interface with `#0969da` accents and high-contrast typography.

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
*On Windows, simply double-click `run_studio.bat`.*

---

<a name="русский"></a>
## 🇷🇺 Русский

**String Art Studio Pro v9.0** — профессиональная студия и высокоточный алгоритмический комплекс для создания картин в технике **String Art** (изонить / филография) из одной непрерывной нити на основе математических методов томографии и эволюционного моделирования.

Программа объединяет преобразование Радона с высокочастотным фильтром Рама-Лака (Filtered Back Projection), высокоточный жадный OMP-синтез, параллельную островную модель генетического алгоритма и оптическое моделирование поглощения света по закону Бера-Ламберта.

---

### 🌟 Подробный обзор возможностей v9.0

#### 1. 🏝 Островная модель генетического алгоритма (Island GA)
* **Параллельная многопоточная эволюция**: каждый вычислительный остров (`IslandThread`) работает в отдельном фоновом потоке со своей популяцией и обменивается элитными особями с другими островами через настраиваемый интервал миграций.
* **Графовые операторы мутации высокого порядка**:
  * **Smart Hinge** — локальное шарнирное перемещение гвоздя с аналитическим дельта-расчетом оптической невязки;
  * **Radon Bridge** — топологический мост с использованием матрицы смежности синограммы Радона;
  * **2-Opt Invert** — инверсия подпоследовательностей пути для распутывания неэффективных пересечений;
  * **Jitter Pin** — фазовый микросдвиг для микроподгонки полутонов.
* **Меметическая оптимизация**: координатный дельта-спуск для субмикронной полировки пути без застревания в локальных минимумах.

#### 2. 📊 Живая аналитика и динамические графики Matplotlib
* **Кривая сходимости (Convergence Curve)**: отслеживание динамики фитнеса каждого острова в реальном времени с пунктирным базовым уровнем OMP.
* **Гистограмма КПД операторов мутации (Mutation KPI)**: диаграмма результативности каждого мутатора (*Smart Hinge*, *Radon Bridge*, *2-Opt*, *Jitter Pin*).
* Графика адаптирована под чистую светлую тему: белый фон (`#ffffff`), мягкая сетка (`#e1e4e8`), контрастные шрифты.

#### 3. 👁 Визуализация в реальном времени (Real-Time Canvas)
* **Прогрессивная отрисовка OMP-синтеза**: картина формируется на глазах пользователя (каждые 25 нитей и на каждом шаге для первых 30 нитей) — видно, как штрих за штрихом рождается изображение.
* **Живое обновление эволюции**: каждое улучшение хромосомы немедленно визуализируется через выделенный поток рендеринга с интеллектуальным сбросом устаревших кадров (*Frame Dropping*), исключая лаги UI.
* **Интерактивные аналитические вкладки**:
  * `★ Лучший результат` — чистовой рендер картины с 4x субпиксельным сглаживанием;
  * `Остров 1..4` — независимые результаты каждого острова;
  * `Карта Ошибок (Кр/Син)` — красно-синяя матрица отклонения оптической плотности;
  * `Плотность нитей (Heatmap)` — карта локального перенасыщения волокон;
  * `Карта Значимости (DoG)` — разность гауссианов и взвешенные контуры лиц и глаз;
  * `Синограмма Радона (FBP)` — томографическая проекция лучевых интегралов.

#### 4. ✨ Выделенная кнопка «Эволюционная доводка»
* Запуск параллельной островной эволюции **на уже построенном пути без сброса и повторного выполнения OMP**.
* Позволяет дорабатывать картину в несколько подходов, изменяя на лету параметры мутаций и веса контуров.

#### 5. ☀️ Современная светлая тема (Light Theme)
* Полный редизайн интерфейса:
  * Фон приложения: `#f6f8fa`
  * Холсты и панели параметров: `#ffffff`
  * Акцентный цвет: `#0969da`
  * Текст и подписи: `#1f2328`
  * Шрифты: системный Segoe UI.

#### 6. 🌀 Преобразование Радона и фильтр Рама-Лака (FBP)
* Вычисление синограммы интегральных проекций по направлениям углов.
* Фильтрация проекций рамповым фильтром Рама-Лака ($|\omega|$) в частотной области Фурье.
* Наведение нитей по синограмме Радона исключает образование темной "каши" и сохраняет черты лица и мелкие детали.

#### 7. 🔊 Голосовой и интерактивный ассистент сборки
* **Windows SAPI Голос**: встроенное голосовое озвучивание номеров целевых гвоздей для работы обеими руками без экрана (*"на 142"*, *"на 89"*).
* **Интерактивный скраббер**: просмотр плетения от 1-й до последней нити с неоновой подсветкой активного хода.
* **Шторка сравнения "До / После" (Split Compare)**: динамическое сопоставление исходного фото и картины нитями прямо поверх холста.

#### 8. 🌐 Автономный интерактивный веб-ассистент (.html)
* Экспорт автономного HTML5-файла со встроенной базой гвоздей, голосовым синтезом Web Speech API, горячими клавишами и сохранением прогресса в `localStorage`.

#### 9. 📦 Полный комплекс экспорта
* **.txt** — пошаговая карта плетения с точным расчетом расхода нити в метрах.
* **.png Ultra HD** — печатный мастер-растр сверхвысокого разрешения (до 2400x2400px) с 4x SSAA сглаживанием.
* **.svg** — масштабируемый векторный чертеж.
* **.html** — автономный веб-помощник для планшета и смартфона.

---

## ⌨ Горячие клавиши

| Клавиша | Действие |
|---|---|
| `Пробел` / `D` / `→` | Следующий шаг в плеере |
| `A` / `←` | Предыдущий шаг в плеере |
| `Ctrl + O` | Открыть изображение |

---

## 📄 Лицензия
Распространяется под свободной лицензией [MIT](LICENSE).  
Автор: **AYes (AYesRU)**.
