# -*- coding: utf-8 -*-
"""
String Art Studio Pro v9.5 — точка входа приложения.
Гибридный комплекс: SciPy CSR OMP + Островной ГА (Island GA) + Преобразование Радона (Ramp FBP)
+ Пакетное ускорение GPU CUDA (PyTorch) + CMYK/Монохром + Интерактивный плеер и SAPI Голос.
"""

import logging
import os
import sys
import tkinter as tk
import traceback
from tkinter import messagebox

from string_art_studio import APP_TITLE, IMAGE_SIZE, MIN_PIN_DISTANCE, PRESETS, TOOLTIPS
from string_art_studio.audio import AudioAssistant
from string_art_studio.engine import (
    FitnessCalculator,
    GreedyInitializer,
    IslandManagerThread,
    IslandThread,
    prune_chromosome_loops,
)
from string_art_studio.geometry import CanvasManager
from string_art_studio.preprocessing import (
    auto_recommend_parameters,
    prepare_target_image,
    prepare_target_image_bundle,
)
from string_art_studio.radon import RadonGuide
from string_art_studio.ui import AppState, EvolutionCoordinator, StringArtApp, UIManager, main

__all__ = [
    "APP_TITLE",
    "IMAGE_SIZE",
    "MIN_PIN_DISTANCE",
    "PRESETS",
    "TOOLTIPS",
    "AudioAssistant",
    "CanvasManager",
    "RadonGuide",
    "FitnessCalculator",
    "GreedyInitializer",
    "IslandThread",
    "IslandManagerThread",
    "prune_chromosome_loops",
    "prepare_target_image",
    "prepare_target_image_bundle",
    "auto_recommend_parameters",
    "AppState",
    "UIManager",
    "EvolutionCoordinator",
    "StringArtApp",
    "main",
]

if __name__ == "__main__":
    try:
        if getattr(sys, "frozen", False):
            os.environ["OPENCV_SALIENCY_PATH"] = os.path.join(sys._MEIPASS, "opencv-saliency")
        main()
    except Exception:
        logging.critical("Критическая ошибка при запуске приложения", exc_info=True)
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Критическая ошибка", f"Не удалось запустить приложение:\n\n{traceback.format_exc()}")
        except Exception:
            pass
