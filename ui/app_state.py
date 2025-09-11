import tkinter as tk
import time
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

class AppState:
    """
    Хранит все состояние приложения в одном месте.
    Этот класс не содержит логики, только данные. Он передается
    в UIManager и EvolutionCoordinator, чтобы они могли обмениваться
    информацией и реагировать на изменения.
    """
    def __init__(self):
        # --- Переменные, связанные с виджетами в UI ---
        # Используются tk.Variable, чтобы автоматически связывать
        # значения слайдеров и полей ввода с состоянием.
        self.vars: Dict[str, tk.Variable] = {
            # Настройки изображения
            'pins': tk.IntVar(value=256),
            'lines': tk.IntVar(value=2000),
            'thread_thickness': tk.IntVar(value=1),
            'thread_scale': tk.DoubleVar(value=1.0),
            # Параметры алгоритма
            'population_size': tk.IntVar(value=50),
            'stages': tk.IntVar(value=4),
            'mutation_rate': tk.IntVar(value=15),
            'num_islands': tk.IntVar(value=4),
            'migration_interval': tk.IntVar(value=20)
        }
        
        # --- Состояние, связанное с исходным изображением ---
        self.original_image: Optional[Image.Image] = None
        self.original_image_path: Optional[str] = None
        self.target_image_tk: Optional[Any] = None # Ссылка на PhotoImage для холста

        # --- Состояние, хранящее лучший результат генерации ---
        self.overall_best_chromosome: Optional[List[int]] = None
        self.overall_best_fitness: float = -1.0
        
        # --- Флаги и счетчики текущего процесса ---
        self.is_running: bool = False
        self.is_paused: bool = False
        self.start_time: float = 0
        self.migration_count: int = 0
        
        # --- Данные для отслеживания прогресса по "островам" ---
        self.generation_counters: Dict[int, int] = {}
        self.chart_data_by_island: Dict[int, List[Tuple[int, float]]] = {}
        self.island_best_fitnesses: Dict[int, float] = {}
        self.island_progress_data: Dict[int, Any] = {}

    def get_config(self) -> Dict[str, Any]:
        """
        Возвращает словарь с текущими настройками (значениями из self.vars).
        Удобно для передачи конфигурации в фоновые потоки.
        """
        return {k: v.get() for k, v in self.vars.items()}

    def reset_for_new_run(self):
        """Сбрасывает состояние перед новым запуском генерации."""
        self.overall_best_chromosome = None
        self.overall_best_fitness = -1.0
        self.migration_count = 0
        self.is_paused = False
        self.start_time = time.time()
        
        # Сбрасываем статистику для каждого "острова".
        num_islands = self.vars['num_islands'].get()
        self.chart_data_by_island = {i: [] for i in range(num_islands)}
        self.island_best_fitnesses = {i: -1.0 for i in range(num_islands)}
        self.generation_counters = {i: 0 for i in range(num_islands)}
        self.island_progress_data = {i: (0, 1) for i in range(num_islands)}
