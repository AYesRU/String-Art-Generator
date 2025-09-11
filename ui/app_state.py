import tkinter as tk
import time
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

class AppState:
    """
    Централизованное хранилище состояния приложения.
    Содержит все переменные состояния, настройки UI и данные о ходе выполнения.
    """
    
    def __init__(self):
        # Переменные для связывания с элементами UI
        self.vars: Dict[str, tk.Variable] = {
            # Настройки изображения
            'pins': tk.IntVar(value=256),          # Количество гвоздей
            'lines': tk.IntVar(value=2000),        # Количество линий
            'thread_thickness': tk.IntVar(value=1), # Толщина нити
            
            # Параметры алгоритма
            'population_size': tk.IntVar(value=50),   # Размер популяции
            'stages': tk.IntVar(value=4),             # Количество стадий
            'mutation_rate': tk.IntVar(value=15),     # Процент мутаций
            'num_islands': tk.IntVar(value=4),        # Количество островов
            'migration_interval': tk.IntVar(value=20) # Интервал миграции
        }
        
        # Данные исходного изображения
        self.original_image: Optional[Image.Image] = None
        self.original_image_path: Optional[str] = None
        self.target_image_tk: Optional[Any] = None  # Ссылка на PhotoImage для отображения
        
        # Результаты генерации
        self.overall_best_chromosome: Optional[List[int]] = None
        self.overall_best_fitness: float = -1.0
        
        # Статус выполнения
        self.is_running: bool = False
        self.is_paused: bool = False
        self.start_time: float = 0
        self.migration_count: int = 0
        
        # Данные по островам
        self.generation_counters: Dict[int, int] = {}
        self.chart_data_by_island: Dict[int, List[Tuple[int, float]]] = {}
        self.island_best_fitnesses: Dict[int, float] = {}
        self.island_progress_data: Dict[int, Any] = {}

    def get_config(self) -> Dict[str, Any]:
        """
        Получение текущей конфигурации в виде словаря.
        
        Возвращает:
            Dict[str, Any]: Словарь с текущими значениями всех переменных
        """
        return {k: v.get() for k, v in self.vars.items()}

    def reset_for_new_run(self):
        """Сброс состояния перед новым запуском генерации."""
        self.overall_best_chromosome = None
        self.overall_best_fitness = -1.0
        self.migration_count = 0
        self.is_paused = False
        self.start_time = time.time()
        
        # Сброс данных по островам
        num_islands = self.vars['num_islands'].get()
        self.chart_data_by_island = {i: [] for i in range(num_islands)}
        self.island_best_fitnesses = {i: -1.0 for i in range(num_islands)}
        self.generation_counters = {i: 0 for i in range(num_islands)}
        self.island_progress_data = {i: (0, 1) for i in range(num_islands)}