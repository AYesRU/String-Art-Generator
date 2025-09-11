import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import queue
import sys
import os
import traceback

# Добавляем корневую папку проекта в системный путь, чтобы импорты
# из папок core/ и ui/ работали корректно.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.app_state import AppState
from ui.ui_manager import UIManager
from ui.evolution_coordinator import EvolutionCoordinator

# Проверяем доступность CUDA для GPU-ускорений.
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


class StringArtApp(tk.Tk):
    """
    Основной класс приложения. Наследуется от tk.Tk и является главным окном.
    Он инициализирует все основные компоненты (состояние, UI, координатор),
    связывает их и запускает главный цикл обработки событий.
    """
    def __init__(self):
        super().__init__()
        self.geometry("1400x900")
        self.title("String Art Generator")

        # AppState хранит все переменные и настройки UI.
        self.state = AppState()
        
        # Очереди для обмена сообщениями между основным потоком UI и фоновыми потоками.
        self.queues = {
            'status_queue': queue.Queue(),
        }
        
        # UIManager отвечает за создание и обновление всех виджетов.
        self.ui_manager = UIManager(self, self.state)
        # EvolutionCoordinator управляет всей логикой генерации и фоновыми потоками.
        self.coordinator = EvolutionCoordinator(self, self.state, self.ui_manager, self.queues)
        # Передаем ссылку на координатор в UI, чтобы кнопки могли вызывать его методы.
        self.ui_manager.coordinator = self.coordinator
        
        self.ui_manager.create_widgets()
        self.ui_manager.update_title()
        
        # Привязываем команду к кнопке загрузки изображения напрямую.
        self.ui_manager.widgets['load_image_button'].config(command=self.load_image)

        # Запускаем "слушателя" очередей, который будет обрабатывать сообщения от фоновых потоков.
        self.after(100, self.coordinator.process_queues)
        # Запускаем периодическое обновление "тяжелых" элементов UI, таких как график.
        self.after(500, self._periodic_ui_update) 
        self.after(200, self.show_startup_messages)
        
        # Устанавливаем обработчик для события закрытия окна.
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        """Обработчик закрытия окна. Спрашивает подтверждение, если генерация активна."""
        if self.state.is_running:
            if messagebox.askokcancel("Выход", "Процесс генерации активен. Вы уверены, что хотите выйти?"):
                self.coordinator.stop_generation()
                self.destroy()
        else:
            self.destroy()

    def _periodic_ui_update(self):
        """
        Периодически обновляет "тяжелые" элементы UI.
        Это сделано для оптимизации, чтобы не перерисовывать, например,
        график на каждое мелкое обновление данных.
        """
        if self.ui_manager.chart_needs_update:
            self.ui_manager.update_chart()
        self.after(500, self._periodic_ui_update)

    def show_startup_messages(self):
        """Показывает информацию о доступных технологиях ускорения (CPU/GPU) при запуске."""
        self.ui_manager.widgets['compute_mode_label'].config(text=f"Режим: {'GPU (CUDA)' if CUDA_AVAILABLE else 'CPU'}")

    def load_image(self):
        """Открывает диалог выбора файла и загружает изображение в состояние приложения."""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path: return
        try:
            self.state.original_image = Image.open(file_path)
            self.state.original_image_path = file_path
            self.ui_manager.update_title()
            self.redraw_target_canvas()
            self.ui_manager.widgets['status_label'].config(text="Изображение загружено.")
        except Exception as e: 
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {e}")

    def redraw_target_canvas(self, event=None):
        """
        Перерисовывает холст с целевым изображением. Вызывается при загрузке
        нового изображения или при изменении размеров окна.
        """
        if not self.state.original_image: return
        canvas = self.ui_manager.widgets['target_canvas']
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 10 or h < 10: return # Не пытаемся рисовать на свернутом холсте.
        
        # Создаем копию, масштабируем ее под размер холста и размещаем по центру на белом фоне.
        img_copy = self.state.original_image.copy()
        img_copy.thumbnail((w, h), Image.Resampling.LANCZOS)
        bg = Image.new('RGB', (w, h), 'white')
        bg.paste(img_copy, ((w - img_copy.width) // 2, (h - img_copy.height) // 2))
        
        # Преобразуем в PhotoImage, который может отображать Tkinter.
        # Важно сохранить ссылку на этот объект, иначе он будет удален сборщиком мусора.
        self.state.target_image_tk = ImageTk.PhotoImage(bg)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=self.state.target_image_tk)


if __name__ == "__main__":
    # Точка входа в приложение.
    try:
        app = StringArtApp()
        app.mainloop()
    except Exception:
        # Отлавливаем критические ошибки при запуске и показываем их в окне.
        error_message = f"Критическая ошибка при запуске приложения:\n\n{traceback.format_exc()}"
        root = tk.Tk()
        root.withdraw() # Прячем пустое корневое окно.
        messagebox.showerror("Критическая ошибка", error_message)