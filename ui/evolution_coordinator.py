import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import queue
import pickle
import threading
import traceback
import os
import time
import subprocess
import sys
from typing import Dict, Any, Optional, List, Tuple, Callable

from .app_state import AppState
from .ui_manager import UIManager
from core.evolution import IslandManagerThread, GreedyInitializer
from core.canvas import CanvasManager, IMAGE_SIZE
from core.fitness import FitnessCalculator
from core.image_processing import prepare_target_image

if False:
    from main import StringArtApp


class EvolutionCoordinator:
    """
    Класс-оркестратор. Управляет логикой эволюции, фоновыми потоками,
    обработкой данных и взаимодействием между UI (UIManager) и ядром алгоритма.
    """
    def __init__(self, root: 'StringArtApp', state: AppState, ui_manager: UIManager, queues: Dict[str, queue.Queue]):
        self.root = root
        self.state = state
        self.ui_manager = ui_manager
        self.queues = queues
        
        self.manager_thread: Optional[IslandManagerThread] = None
        self.renderer_thread: Optional[threading.Thread] = None
        
        self.render_queue: queue.Queue = queue.Queue()
        self.render_results_queue: queue.Queue = queue.Queue()
        
        # Словарь для хранения лучших хромосом (для 'overall' и для каждого острова).
        self.best_chromosomes_by_key: Dict[Any, List[int]] = {}
        self.stats_update_job: Optional[str] = None
        
        # Ссылка на оригинальное изображение для карты расхождений.
        self.target_image_original: Optional[np.ndarray] = None
        
        self._start_renderer_thread()

    def _start_renderer_thread(self):
        """Запускает фоновый поток, который занимается отрисовкой изображений."""
        self.renderer_thread = threading.Thread(target=self._renderer_worker, daemon=True)
        self.renderer_thread.start()

    def _renderer_worker(self):
        """
        Главная функция потока-отрисовщика. Бесконечно слушает очередь `render_queue`,
        генерирует изображения и кладет результат в `render_results_queue`.
        """
        while True:
            try:
                key, data, render_params = self.render_queue.get()
                
                # Рендеринг numpy-массива (например, карты расхождений).
                if isinstance(data, np.ndarray):
                    img_pil = Image.fromarray(data)
                # Рендеринг хромосомы.
                elif isinstance(data, tuple):
                    chromosome, pins, thread_thickness = data
                    size = render_params.get('size', 500)
                    cm = CanvasManager(size, pins)
                    visual_thickness = max(1, int(1 * thread_thickness))
                    img_pil = cm.render_chromosome_to_pil(chromosome, visual_thickness)
                else:
                    continue

                self.render_results_queue.put((key, img_pil))
            except Exception:
                # В случае ошибки просто пропускаем элемент, чтобы поток не падал.
                pass

    def _prepare_and_run_evolution(self):
        """
        Подготавливает все необходимое для запуска генетического алгоритма
        и запускает его в отдельном потоке.
        """
        try:
            config = self.state.get_config()
            self._send_status("Подготовка изображения...")
            
            target_image_np = prepare_target_image(self.state.original_image, IMAGE_SIZE)
            
            # Сохраняем оригинальное Ч/Б изображение для использования в карте расхождений.
            self.target_image_original = target_image_np.copy()
            
            cm = CanvasManager(IMAGE_SIZE, config['pins'])
            # ИСПРАВЛЕНО: Передаем толщину нити, чтобы FitnessCalculator мог
            # корректно определить, можно ли использовать GPU.
            fc = FitnessCalculator(target_image_np, cm, config['thread_thickness'], self.queues)
            
            # Если доступен GPU, предварительно кэшируем геометрию всех линий.
            if fc.device:
                self._send_status("Кэширование линий для GPU...")
                line_pixels_gpu = cm.precompute_line_data_for_gpu(fc.device)
                fc.enable_gpu_rendering(line_pixels_gpu)
            
            # Инвертируем изображение для жадного алгоритма (он ищет светлые области).
            target_flat = (255 - target_image_np).flatten()
            greedy = GreedyInitializer(cm, target_flat)
            
            # Callback-функция для отображения прогресса жадного алгоритма в реальном времени.
            def greedy_callback(status: str, path: List[int]):
                self.queues['status_queue'].put({'type': 'greedy_update', 'data': {'text': status, 'path': path}})
                render_data = (path, config['pins'], self.state.vars['thread_thickness'].get())
                self.request_canvas_update('overall', render_data, {'size': 500})

            base_chromosome = greedy.run(config['lines'], greedy_callback)
            
            # Проверяем, что жадный алгоритм создал валидный путь.
            if not base_chromosome or len(base_chromosome) < 2:
                self.queues['status_queue'].put({'type': 'error', 'data': "Жадный алгоритм не смог создать начальный путь. Попробуйте другое изображение."})
                return
            
            # Запускаем главный управляющий поток для "островов".
            self.manager_thread = IslandManagerThread(config, self.queues, fc, base_chromosome)
            self.manager_thread.start()
            
            self.stats_update_job = self.root.after(1000, self._update_progress_stats)

        except Exception:
            error_msg = f"Ошибка в потоке подготовки: {traceback.format_exc()}"
            self.queues['status_queue'].put({'type': 'error', 'data': error_msg})
    
    def start_generation(self):
        """Начинает процесс генерации. Вызывается по нажатию кнопки 'Старт'."""
        if self.state.original_image is None:
            messagebox.showwarning("Внимание", "Пожалуйста, сначала загрузите изображение.")
            return
            
        self.state.reset_for_new_run()
        self.ui_manager.toggle_ui_state(True)
        self.best_chromosomes_by_key.clear()
        self.ui_manager.setup_for_new_run(self.state.get_config()['num_islands'])

        threading.Thread(target=self._prepare_and_run_evolution, daemon=True).start()

    def stop_generation(self):
        """Останавливает процесс генерации."""
        if self.stats_update_job:
            self.root.after_cancel(self.stats_update_job)
            self.stats_update_job = None
        if self.manager_thread and self.manager_thread.is_alive():
            self.manager_thread.stop()
            self._send_status("Остановка...")

    def toggle_pause(self):
        """Приостанавливает или возобновляет генерацию."""
        if not self.manager_thread or not self.manager_thread.is_alive(): return
        
        self.state.is_paused = not self.state.is_paused
        if self.state.is_paused:
            if self.stats_update_job:
                self.root.after_cancel(self.stats_update_job)
                self.stats_update_job = None
            self.manager_thread.pause()
            self.ui_manager.widgets['pause_button'].config(text="► Возобновить")
            self._send_status("Пауза...")
        else:
            self.manager_thread.resume()
            self.ui_manager.widgets['pause_button'].config(text="❚❚ Пауза")
            self.stats_update_job = self.root.after(1000, self._update_progress_stats)

    def process_queues(self):
        """
        Периодически опрашивает очереди сообщений от фоновых потоков и
        очередь с готовыми изображениями для обновления UI.
        """
        try:
            while not self.queues['status_queue'].empty():
                msg = self.queues['status_queue'].get_nowait()
                handler = self._get_message_handler(msg.get('type'))
                handler(msg.get('data'))
        except queue.Empty:
            pass
        
        try:
            while not self.render_results_queue.empty():
                key, img_pil = self.render_results_queue.get_nowait()
                self.ui_manager.update_canvas_image(key, img_pil)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queues)

    def _get_message_handler(self, msg_type: str) -> Callable:
        """Возвращает метод-обработчик для конкретного типа сообщения."""
        handlers = {
            'greedy_update': self._handle_greedy_update,
            'status': self._handle_status,
            'error': self._handle_error,
            'gpu_status': self._handle_gpu_status,
            'generation_tick': self._handle_generation_tick,
            'island_update': self._handle_island_update,
            'done': self._handle_done,
            'migration': self._handle_migration,
            'difference_map_update': self._handle_difference_map_update,
        }
        return handlers.get(msg_type, lambda data: None)

    # --- Методы-обработчики сообщений из очереди ---
    def _handle_greedy_update(self, data: Dict[str, Any]):
        self.ui_manager.widgets['status_label'].config(text=data['text'])
    
    def _handle_status(self, data: str):
        self.ui_manager.widgets['status_label'].config(text=data)
        
    def _handle_error(self, data: str):
        messagebox.showerror("Ошибка в потоке", data)
        self.ui_manager.widgets['status_label'].config(text="Ошибка!")
        self.ui_manager.toggle_ui_state(False)

    def _handle_gpu_status(self, data: bool):
        mode_text = f"Режим: {'GPU (CUDA)' if data else 'CPU'}"
        self.ui_manager.widgets['compute_mode_label'].config(text=mode_text)

    def _handle_generation_tick(self, data: Tuple[int, int, int, int]):
        island_id, stage, patience, limit = data
        total_gens = self.state.generation_counters.get(island_id, 0) + 1
        self.state.generation_counters[island_id] = total_gens
        self.state.island_progress_data[island_id] = (stage, limit)
        self.ui_manager.update_island_progress(island_id, stage, patience, limit)
        
        if total_gens % 10 == 0: 
            current_fitness = self.state.island_best_fitnesses.get(island_id, 0)
            self.state.chart_data_by_island.setdefault(island_id, []).append((sum(self.state.generation_counters.values()), current_fitness))
            self.ui_manager.chart_needs_update = True
        
    def _handle_island_update(self, data: Dict[str, Any]):
        island_id, chromosome, fitness = data['id'], data['chromosome'], data['fitness']
        
        self.state.island_best_fitnesses[island_id] = fitness
        self.best_chromosomes_by_key[island_id] = chromosome
        
        render_data = (chromosome, self.state.get_config()['pins'], self.state.vars['thread_thickness'].get())
        self.request_canvas_update(island_id, render_data, {'size': 500})
        
        if fitness > self.state.overall_best_fitness:
            self.state.overall_best_fitness = fitness
            self.state.overall_best_chromosome = chromosome
            self.best_chromosomes_by_key['overall'] = chromosome
            self.ui_manager.update_best_fitness_display(fitness)
            self.request_canvas_update('overall', render_data, {'size': 500})
        
    def _handle_migration(self, _: Any):
        self.state.migration_count += 1
        
    def _handle_done(self, data: str):
        """Обработчик завершения генерации"""
        self.ui_manager.widgets['status_label'].config(text=data)
        self.ui_manager.toggle_ui_state(False)
        if self.manager_thread: 
            self.manager_thread.join()
        
    def _handle_difference_map_update(self, data: np.ndarray):
        if data is not None and self.target_image_original is not None:
            heatmap = self.create_difference_map_overlay(data, self.target_image_original)
            self.request_canvas_update('ssim_map', heatmap)
    
    def _send_status(self, message: str):
        """Отправляет текстовый статус в очередь для отображения в UI."""
        self.queues['status_queue'].put({'type': 'status', 'data': message})

    def request_canvas_update(self, key: Any, data: Any, render_params: Dict[str, Any] = {}):
        """Отправляет задачу на отрисовку в фоновый поток."""
        canvas = self.ui_manager.result_canvases.get(key)
        if not canvas or (canvas.winfo_width() < 10): 
            return
        self.render_queue.put((key, data, render_params))
    
    def _update_progress_stats(self):
        """Периодически вызывается для обновления статистики времени и ETA."""
        if not self.state.is_running or not self.manager_thread or not self.manager_thread.is_alive(): 
            return
        
        self.ui_manager.update_stats_display(self.state)
        self.stats_update_job = self.root.after(1000, self._update_progress_stats)

    def create_difference_map_overlay(self, diff_map: np.ndarray, base_image: np.ndarray) -> np.ndarray:
        """
        Создание карты расхождений без легенды
        """
        # Нормализация карты расхождений
        max_abs_val = np.max(np.abs(diff_map))
        if max_abs_val == 0:
            max_abs_val = 1.0
        norm_diff = diff_map / max_abs_val

        # Создание цветовой маски
        h, w = diff_map.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Цветовая кодировка:
        # Красный - избыток нитей (цель светлее рендера)
        red_mask = norm_diff > 0.1
        color_mask[red_mask] = [255, 0, 0]

        # Синий - недостаток нитей (цель темнее рендера)
        blue_mask = norm_diff < -0.1
        color_mask[blue_mask] = [0, 0, 255]

        # Конвертация в PIL изображение
        base_pil = Image.fromarray(base_image).convert("RGBA")
        color_pil = Image.fromarray(color_mask).convert("RGBA")

        # Создание альфа-канала с учетом величины расхождения
        alpha = Image.fromarray((np.abs(norm_diff) * 200).astype(np.uint8))
        color_pil.putalpha(alpha)

        # Наложение цветовой маски
        combined = Image.alpha_composite(base_pil, color_pil)
        
        return np.array(combined.convert("RGB"))

    def prune_chromosome(self):
        """Удаляет из пути лишние линии вида A->B->A."""
        if not self.state.overall_best_chromosome:
            messagebox.showinfo("Информация", "Сначала нужно сгенерировать изображение.")
            return
        
        chromo = self.state.overall_best_chromosome
        pruned_chromo = [chromo[0]]
        i = 1
        while i < len(chromo) - 1:
            if chromo[i-1] == chromo[i+1]:
                # Пропускаем точки A->B->A, оставляя только A.
                i += 2
            else:
                pruned_chromo.append(chromo[i])
                i += 1
        pruned_chromo.append(chromo[-1])
        
        original_len = len(chromo)
        pruned_len = len(pruned_chromo)
        
        if original_len != pruned_len:
            self.state.overall_best_chromosome = pruned_chromo
            self.best_chromosomes_by_key['overall'] = pruned_chromo
            render_data = (pruned_chromo, self.state.get_config()['pins'], self.state.vars['thread_thickness'].get())
            self.request_canvas_update('overall', render_data, {'size': 500})
            messagebox.showinfo("Оптимизация", f"Путь оптимизирован!\nУдалено линий: {original_len - pruned_len}")
        else:
            messagebox.showinfo("Оптимизация", "Оптимизация не требуется. Лишних линий не найдено.")

    def export_txt(self):
        """Экспортирует путь в текстовый файл."""
        if not self.state.overall_best_chromosome: return
        filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if not filepath: return
        
        with open(filepath, 'w') as f:
            f.write("String Art Pin Sequence\n")
            f.write(f"Total Pins: {self.state.get_config()['pins']}\n")
            f.write(f"Total Lines: {len(self.state.overall_best_chromosome) - 1}\n\n")
            f.write("Sequence (Pin numbers start from 0):\n")
            # Нумеруем шаги для удобства.
            for i, pin in enumerate(self.state.overall_best_chromosome):
                f.write(f"{i}: {pin}\n")
        messagebox.showinfo("Экспорт", f"Путь успешно сохранен в {os.path.basename(filepath)}")

    def export_png(self):
        """Экспортирует изображение в PNG высокого разрешения."""
        if not self.state.overall_best_chromosome: return
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG images", "*.png")])
        if not filepath: return
        
        self._send_status("Экспорт PNG (2000px)...")
        # Рендерим в высоком разрешении для качественного экспорта.
        EXPORT_SIZE = 2000
        config = self.state.get_config()
        cm = CanvasManager(EXPORT_SIZE, config['pins'])
        img = cm.render_chromosome_to_pil(self.state.overall_best_chromosome, config['thread_thickness'])
        img.save(filepath)
        self._send_status("Экспорт PNG завершен.")
        messagebox.showinfo("Экспорт", f"Изображение успешно сохранено в {os.path.basename(filepath)}")

    def export_svg(self):
        """Экспортирует в SVG."""
        if not self.state.overall_best_chromosome: return
        filepath = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG images", "*.svg")])
        if not filepath: return

        self._send_status("Экспорт SVG...")
        config = self.state.get_config()
        SVG_SIZE = 1000
        cm = CanvasManager(SVG_SIZE, config['pins'])
        coords = cm.pin_coords

        with open(filepath, 'w') as f:
            f.write(f'<svg width="{SVG_SIZE}" height="{SVG_SIZE}" xmlns="http://www.w3.org/2000/svg">\n')
            f.write(f'  <rect width="100%" height="100%" fill="white"/>\n')
            
            path_data = "M " + " L ".join([f"{coords[p][0]:.2f},{coords[p][1]:.2f}" for p in self.state.overall_best_chromosome])
            f.write(f'  <path d="{path_data}" stroke="black" stroke-width="{config["thread_thickness"]}" fill="none" />\n')
            
            f.write('</svg>\n')
        self._send_status("Экспорт SVG завершен.")
        messagebox.showinfo("Экспорт", f"SVG успешно сохранен в {os.path.basename(filepath)}")
