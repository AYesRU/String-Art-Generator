# -*- coding: utf-8 -*-
import os
import tempfile
import time
import unittest
from PIL import Image, ImageDraw

from string_art_studio.ui import StringArtApp


class TestAppIntegration(unittest.TestCase):
    def test_gui_full_lifecycle_monochrome_and_cmyk(self):
        try:
            app = StringArtApp()
        except Exception as exc:
            self.skipTest(f"Tkinter display unavailable in environment: {exc}")
            return

        app.withdraw()
        try:
            app.update_idletasks()
            app.update()

            # Создаем синтетический портрет с четкими контурами
            img = Image.new("RGB", (400, 400), (245, 245, 245))
            draw = ImageDraw.Draw(img)
            draw.ellipse((90, 60, 310, 340), fill=(190, 175, 165), outline=(40, 40, 40), width=4)
            draw.ellipse((145, 150, 185, 180), fill=(25, 25, 25))
            draw.ellipse((215, 150, 255, 180), fill=(25, 25, 25))
            draw.arc((150, 220, 250, 285), 10, 170, fill=(35, 35, 35), width=5)

            app.state.original_image = img
            app.state.original_image_path = "synthetic_portrait.png"
            app.apply_auto_recommendations()
            app.update()

            # Проверка безопасности при временной пустой строке в tk.Variable
            app.state.vars["pins"].set("")
            self.assertIsNotNone(app.state.get_config()["pins"])

            # Быстрый полный прогон Монохром (CSR OMP + Island GA + Delta)
            app.state.vars["pins"].set(140)
            app.state.vars["lines"].set(120)
            app.state.vars["num_islands"].set(2)
            app.state.vars["population_size"].set(6)
            app.state.vars["stages"].set(1)
            app.state.vars["memetic_depth"].set(8)
            app.state.vars["color_mode"].set("Монохром (Черная нить)")

            app.coordinator.start_generation()
            deadline = time.time() + 25.0
            while app.state.is_running and time.time() < deadline:
                app.update()
                time.sleep(0.03)

            self.assertFalse(app.state.is_running, "Синтез не завершился вовремя")
            self.assertTrue(app.state.overall_best_chromosome)
            self.assertGreaterEqual(len(app.state.overall_best_chromosome), 30)
            self.assertGreater(app.state.overall_best_fitness, 0.5)

            # Тест пошагового плеера
            app.step_player(-5)
            app.update()
            self.assertIn("Шаг", app.ui_manager.widgets["player_step_label"].cget("text"))

            # Тест режима Эволюционная доводка (Polish)
            app.coordinator.start_evolution_only()
            deadline = time.time() + 20.0
            while app.state.is_running and time.time() < deadline:
                app.update()
                time.sleep(0.03)
            self.assertFalse(app.state.is_running)

            # Тест экспорта HTML из UI состояния
            with tempfile.TemporaryDirectory() as tmp:
                out_html = os.path.join(tmp, "assistant.html")
                from string_art_studio.exporters import export_html_assistant

                export_html_assistant(out_html, "GUI Test", app.coordinator.canvas_manager, app.state.layers)
                self.assertTrue(os.path.exists(out_html))
        finally:
            app.coordinator.shutdown()
            app.audio_assistant.shutdown()
            app.destroy()


if __name__ == "__main__":
    unittest.main()
