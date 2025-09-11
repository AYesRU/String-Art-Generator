import numpy as np
import threading
import queue
import random
import traceback
import time
from typing import Dict, Any, List, Optional, Callable

from .canvas import CanvasManager
from .fitness import FitnessCalculator


class GreedyInitializer:
    """
    Инициализатор жадного алгоритма для создания начального приближения.
    Строит первоначальный путь, который затем оптимизируется эволюционным алгоритмом.
    """
    
    def __init__(self, canvas_manager: CanvasManager, target_flat: np.ndarray):
        self.canvas_manager = canvas_manager
        self.target_flat = target_flat  # Целевое изображение в виде 1D массива
        self.num_pins = canvas_manager.num_pins

    def run(self, num_lines: int, update_callback: Callable[[str, List[int]], None]) -> List[int]:
        """
        Запуск жадного алгоритма для построения начального пути.
        
        Параметры:
            num_lines: int - Количество линий для построения
            update_callback: Callable - Функция для обновления прогресса
            
        Возвращает:
            List[int]: Построенная хромосома (последовательность гвоздей)
        """
        # Начало с случайного гвоздя
        path = [np.random.randint(0, self.num_pins)]
        current_pin = path[0]
        residual_image = self.target_flat.copy().astype(np.float32)

        for i in range(num_lines - 1):
            scores = np.full(self.num_pins, -np.inf, dtype=np.float32)
            
            # Оценка каждого возможного следующего гвоздя
            for next_pin in range(self.num_pins):
                if next_pin == current_pin: 
                    continue  # Пропуск текущего гвоздя
                if len(path) > 1 and next_pin == path[-2]: 
                    continue  # Пропуск возврата назад
                    
                line_pixels = self.canvas_manager.get_line_pixel_indices(current_pin, next_pin)
                if len(line_pixels) > 0:
                    scores[next_pin] = np.sum(residual_image[line_pixels])

            if np.all(np.isneginf(scores)): 
                break  # Нет допустимых ходов
                
            best_next_pin = np.argmax(scores)
            best_line_pixels = self.canvas_manager.get_line_pixel_indices(current_pin, best_next_pin)
            
            if len(best_line_pixels) > 0:
                # Обновление остаточного изображения
                np.subtract.at(residual_image, best_line_pixels, 20)
                np.maximum(residual_image, 0, out=residual_image)
            
            path.append(best_next_pin)
            current_pin = best_next_pin

            # Периодическое обновление прогресса
            if i % 10 == 0:
                progress = int(((i+1) / num_lines) * 100)
                update_callback(f"Этап 1: Набросок... {progress}%", path)
        
        update_callback("Этап 1: Набросок... 100%", path)
        return path


class IslandThread(threading.Thread):
    """
    Поток для выполнения эволюционного алгоритма на отдельном острове.
    Каждый остров работает со своей популяцией и периодически обменивается лучшими решениями.
    """
    
    def __init__(self, island_id: int, config: Dict[str, Any], 
                 fitness_calculator: FitnessCalculator,
                 queues: Dict[str, queue.Queue], base_chromosome: List[int]):
        super().__init__()
        self.island_id = island_id
        self.config = config
        self.fitness_calculator = fitness_calculator
        self.queues = queues
        self.base_chromosome = base_chromosome
        
        self.status_queue = queues['status_queue']
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()  # Изначально не на паузе

        self.population: List[Dict[str, Any]] = []
        self.best_on_island: Optional[Dict[str, Any]] = None
        self.migrant_queue: queue.Queue = queue.Queue()

        # Стратегии мутации для разнообразия
        self.random_mutation_strategies = [
            self._reverse_segment_mutation, 
            self._swap_points_mutation,
            self._insert_point_mutation, 
            self._scramble_segment_mutation
        ]

    def stop(self): 
        """Остановка потока"""
        self.stop_event.set()
        
    def get_best_individual(self) -> Optional[Dict[str, Any]]: 
        """Получение лучшей особи на острове"""
        return self.best_on_island
        
    def accept_migrant(self, migrant: Dict[str, Any]): 
        """Прием мигранта с другого острова"""
        self.migrant_queue.put(migrant)
        
    def pause(self): 
        """Приостановка выполнения"""
        self.pause_event.clear()
        
    def resume(self): 
        """Возобновление выполнения"""
        self.pause_event.set()

    # Стратегии мутации
    def _reverse_segment_mutation(self, chromo: List[int], start: int, end: int) -> List[int]:
        """Реверсирование сегмента хромосомы"""
        segment = chromo[start:end]
        segment.reverse()
        chromo[start:end] = segment
        return chromo

    def _swap_points_mutation(self, chromo: List[int], start: int, end: int) -> List[int]:
        """Обмен двух точек в сегменте"""
        if end - start >= 2:
            idx1, idx2 = random.sample(range(start, end), 2)
            chromo[idx1], chromo[idx2] = chromo[idx2], chromo[idx1]
        return chromo

    def _insert_point_mutation(self, chromo: List[int], start: int, end: int) -> List[int]:
        """Перемещение точки в пределах сегмента"""
        if end - start >= 2:
            idx_from, idx_to = random.sample(range(start, end), 2)
            point = chromo.pop(idx_from)
            chromo.insert(idx_to, point)
        return chromo

    def _scramble_segment_mutation(self, chromo: List[int], start: int, end: int) -> List[int]:
        """Перемешивание сегмента хромосомы"""
        segment = chromo[start:end]
        random.shuffle(segment)
        chromo[start:end] = segment
        return chromo
    
    def run(self):
        """Основной метод выполнения эволюционного алгоритма"""
        try:
            # Инициализация популяции
            initial_fitness, diff_map = self.fitness_calculator.calculate_single_fitness_and_map(
                self.base_chromosome
            )
            
            self.best_on_island = {
                'chromosome': self.base_chromosome, 
                'fitness': initial_fitness
            }
            self.population = [{
                'chromosome': list(self.base_chromosome), 
                'fitness': initial_fitness
            } for _ in range(self.config['population_size'])]
            
            self._send_island_update()
            if self.island_id == 0:
                self._send_difference_map_update(diff_map)

            # Поэтапная оптимизация
            lines_per_stage = self.config['lines'] // self.config['stages']
            
            for stage in range(self.config['stages']):
                if self.stop_event.is_set(): 
                    break
                
                # Определение сегмента для оптимизации на текущем этапе
                start_line, end_line = stage * lines_per_stage, min(
                    (stage + 1) * lines_per_stage, 
                    self.config['lines']
                )
                
                # Настройка параметров терпения для ранней остановки
                patience, patience_limit = 0, 20 + (stage * 15)
                last_best_fitness = self.best_on_island['fitness']

                while patience < patience_limit:
                    self.pause_event.wait()  # Ожидание при паузе
                    if self.stop_event.is_set(): 
                        break
                    
                    self._send_generation_tick(stage, patience, patience_limit)
                    self._process_migrant()
                        
                    # Создание нового поколения
                    new_population = [self.population[0]]  # Элитизм
                    
                    # Динамическая настройка скорости мутации
                    base_mutation_rate = self.config['mutation_rate'] / 50.0
                    stagnation_factor = 1.0 + (patience / patience_limit) * 2.0
                    current_mutation_rate = min(base_mutation_rate * stagnation_factor, 0.8)

                    while len(new_population) < self.config['population_size']:
                        # Турнирный отбор родителей
                        p1 = min(random.sample(self.population, 5), 
                                key=lambda x: -x['fitness'])
                        p2 = min(random.sample(self.population, 5), 
                                key=lambda x: -x['fitness'])
                        
                        child_chromo = list(p1['chromosome'])
                        
                        # Кроссовер
                        if start_line < end_line -1:
                            crossover_point = random.randint(start_line, end_line-1)
                            child_chromo[crossover_point:end_line] = p2['chromosome'][crossover_point:end_line]
                        
                        # Мутация
                        if random.random() < current_mutation_rate:
                            mutation_func = random.choice(self.random_mutation_strategies)
                            child_chromo = mutation_func(child_chromo, start_line, end_line)
                        
                        new_population.append({'chromosome': child_chromo, 'fitness': 0})

                    # Оценка приспособленности нового поколения
                    self.population = self.fitness_calculator.calculate_batch_fitness(new_population)
                    
                    # Обновление лучшего решения
                    if self.population[0]['fitness'] > self.best_on_island['fitness']:
                        self.best_on_island = {
                            'chromosome': list(self.population[0]['chromosome']), 
                            'fitness': self.population[0]['fitness']
                        }
                        
                        _, diff_map = self.fitness_calculator.calculate_single_fitness_and_map(
                            self.best_on_island['chromosome']
                        )
                        if self.island_id == 0:
                            self._send_difference_map_update(diff_map)
                        self._send_island_update()
                    
                    # Обновление счетчика терпения
                    if self.best_on_island['fitness'] > last_best_fitness + 1e-6:
                        last_best_fitness = self.best_on_island['fitness']
                        patience = 0
                    else:
                        patience += 1
        except Exception:
            error_msg = f"Ошибка на острове {self.island_id}: {traceback.format_exc()}"
            self.status_queue.put({'type': 'error', 'data': error_msg})

    def _process_migrant(self):
        """Обработка мигрантов из других островов"""
        try:
            migrant = self.migrant_queue.get_nowait()
            if migrant['fitness'] > self.population[-1]['fitness']:
                self.population[-1] = migrant
        except queue.Empty:
            pass

    def _send_island_update(self):
        """Отправка обновления состояния острова"""
        self.status_queue.put({
            'type': 'island_update', 
            'data': {
                'id': self.island_id, 
                'chromosome': self.best_on_island['chromosome'], 
                'fitness': self.best_on_island['fitness']
            }
        })

    def _send_difference_map_update(self, diff_map: Optional[np.ndarray]):
        """Отправка обновления карты различий"""
        self.status_queue.put({'type': 'difference_map_update', 'data': diff_map})

    def _send_generation_tick(self, stage: int, patience: int, limit: int):
        """Отправка информации о текущем поколении"""
        self.status_queue.put({
            'type': 'generation_tick', 
            'data': (self.island_id, stage, patience, limit)
        })


class IslandManagerThread(threading.Thread):
    """
    Управляющий поток для координации работы всех островов
    и организации миграции между ними.
    """
    
    def __init__(self, config: Dict[str, Any], queues: Dict[str, queue.Queue],
                 fitness_calculator: FitnessCalculator, base_chromosome: List[int]):
        super().__init__()
        self.config = config
        self.queues = queues
        self.fitness_calculator = fitness_calculator
        self.base_chromosome = base_chromosome
        self.islands: List[IslandThread] = []
        self.stop_event = threading.Event()
        self.status_queue = queues['status_queue']

    def stop(self):
        """Остановка всех островов"""
        for island in self.islands: 
            island.stop()
        self.stop_event.set()
        
    def pause(self):
        """Приостановка всех островов"""
        for island in self.islands: 
            island.pause()
            
    def resume(self):
        """Возобновление всех островов"""
        for island in self.islands: 
            island.resume()

    def run(self):
        """Основной метод управления островами"""
        try:
            self.status_queue.put({'type': 'status', 'data': "Запуск островов..."})
            if self.stop_event.is_set(): 
                return

            # Создание и запуск островов
            self.islands = [
                IslandThread(
                    i, self.config, self.fitness_calculator, 
                    self.queues, self.base_chromosome
                ) for i in range(self.config['num_islands'])
            ]
            for island in self.islands: 
                island.start()

            generation_counter = 0
            # Главный цикл управления
            while any(island.is_alive() for island in self.islands):
                if self.stop_event.is_set(): 
                    break
                time.sleep(0.1) 
                generation_counter += 1

                # Организация миграции между островами
                if generation_counter % self.config.get('migration_interval', 20) == 0:
                    self.status_queue.put({'type': 'migration'})
                    migrants = [island.get_best_individual() for island in self.islands]
                    if all(m is not None for m in migrants):
                        for i, island in enumerate(self.islands):
                            island.accept_migrant(migrants[i-1])

            for island in self.islands: 
                island.join()
            
            # Формирование финального сообщения
            msg = 'Процесс остановлен.' if self.stop_event.is_set() else 'Генерация завершена!'
            self.status_queue.put({'type': 'done', 'data': msg})
        except Exception:
            error_msg = f"Ошибка в управляющем потоке: {traceback.format_exc()}"
            self.status_queue.put({'type': 'error', 'data': error_msg})