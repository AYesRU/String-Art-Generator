# -*- coding: utf-8 -*-
import os
import tempfile
import time
import unittest
import numpy as np
from PIL import Image, ImageDraw

from string_art_studio.ui import StringArtApp


class TestAppIntegration(unittest.TestCase):
    def test_gui_full_lifecycle_monochrome_and_cmyk(self):
        try:
            app = StringArtApp(lang="ru")
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

            # Проверка, что вкладки островов удалены и осталось ровно 5 вкладок результатов
            nb = app.ui_manager.widgets["results_notebook"]
            self.assertEqual(len(nb.tabs()), 5)
            self.assertEqual(list(app.ui_manager.result_canvases.keys()), ["overall", "diff_map", "density_heatmap", "fit_saliency", "radon_sinogram"])

            # Проверка переключения режима Адаптивный подбор (N цветов) и видимости спинбокса
            app.state.vars["color_mode"].set("Адаптивный подбор (N цветов)")
            app.ui_manager._update_adaptive_n_visibility()
            app.update()
            self.assertTrue(app.ui_manager._adaptive_n_frame.winfo_ismapped())

            # Проверка адаптивного подбора цветов: исключение белого фона и паразитных оттенков
            test_white_bg = np.full((100, 100, 3), 255, dtype=np.uint8)
            test_white_bg[20:50, 20:80] = [180, 20, 30]   # Темно-красный
            test_white_bg[50:80, 20:80] = [20, 30, 140]   # Темно-синий
            chosen_white_bg = app.coordinator._select_adaptive_colors(test_white_bg, n=3, invert_board=False)
            chosen = chosen_white_bg
            names_wb = [c["name"] for c in chosen_white_bg]
            self.assertNotIn("Белый", names_wb)
            self.assertNotIn("Абрикосовый", names_wb)
            for c in chosen_white_bg:
                # Нити должны иметь реальный физический контраст на белом холсте
                self.assertLessEqual(c["rgb"][0] * 0.299 + c["rgb"][1] * 0.587 + c["rgb"][2] * 0.114, 215)

            app.state.vars["color_mode"].set("Монохром (Черная нить)")
            app.ui_manager._update_adaptive_n_visibility()
            app.update()
            self.assertFalse(app.ui_manager._adaptive_n_frame.winfo_ismapped())

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

            # Тест изоляции устаревших сообщений по run_id
            old_run = app.coordinator._run_id
            app.coordinator._run_id = old_run + 1
            app.queues["status_queue"].put({"type": "status", "run_id": old_run, "data": "STALE_MESSAGE"})
            app.update()
            time.sleep(0.08)
            app.update()
            self.assertNotEqual(app.ui_manager.widgets["status_label"].cget("text"), "STALE_MESSAGE")

            # Тест режима Темная доска (Белая нить)
            app.state.vars["color_mode"].set("Белая нить (Темная доска)")
            app.redraw_target_canvas()
            self.assertEqual(app.ui_manager.widgets["target_canvas"].cget("bg"), "#121418")
            app.state.vars["lines"].set(90)
            app.coordinator.start_generation()
            deadline = time.time() + 25.0
            while app.state.is_running and time.time() < deadline:
                app.update()
                time.sleep(0.03)
            self.assertFalse(app.state.is_running)
            self.assertEqual(app.state.layers[0]["name"], "Белая нить")

            # Тест полного 4-слойного CMYK цикла на КРАСНОМ изображении (Циан должен быть 0 нитей!)
            red_img = Image.new("RGB", (400, 400), (255, 255, 255))
            rdraw = ImageDraw.Draw(red_img)
            rdraw.ellipse((65, 65, 335, 335), fill=(225, 28, 22))
            rdraw.ellipse((135, 135, 265, 265), fill=(115, 12, 12))
            app.state.original_image = red_img
            app.state.vars["color_mode"].set("Цветной CMYK (4 нити)")
            app.state.vars["pins"].set(120)
            app.state.vars["lines"].set(120)
            app.state.vars["stages"].set(1)
            app.state.vars["num_islands"].set(1)
            app.state.vars["population_size"].set(4)
            app.state.vars["memetic_depth"].set(4)
            app.coordinator.start_generation()
            deadline = time.time() + 30.0
            while app.state.is_running and time.time() < deadline:
                app.update()
                time.sleep(0.03)
            self.assertFalse(app.state.is_running, "CMYK синтез не завершился вовремя")
            self.assertEqual(len(app.state.layers), 4)
            # На чисто красном изображении Циан обязан быть пустым (0 нитей), а Маджента и Желтый > 0
            self.assertEqual(len(app.state.layers[0]["path"]), 0, "Циан не должен содержать нитей на красной картинке")
            self.assertGreater(len(app.state.layers[1]["path"]), 10)
            self.assertGreater(len(app.state.layers[2]["path"]), 10)
            for lyr in app.state.layers:
                self.assertLessEqual(max(0, len(lyr["path"]) - 1), 120)

            # Проверка возврата к «🌈 Все цвета» в цветном режиме
            cb = app.ui_manager.widgets["active_layer_combo"]
            self.assertIn("🌈", cb["values"][0])
            cb.current(0)
            app.ui_manager._on_active_layer_changed()
            app.update()
            self.assertEqual(app.state.active_layer_idx, -1)
            self.assertIn("Все цвета", app.ui_manager.widgets["player_step_label"].cget("text"))

            # Проверка цветного рендеринга с произвольными цветами (не только CMYK)
            test_layers = [
                {"name": "Коралл", "color_hex": "#FF7F50", "path": [0, 40, 80, 0]},
                {"name": "Лазурь", "color_hex": "#007FFF", "path": [20, 60, 100, 20]},
            ]
            rendered_pil = app.coordinator.canvas_manager.render_layers_to_pil(test_layers, output_size=100)
            rendered_arr = np.array(rendered_pil)
            # R, G, B каналы не должны быть идентичны (т.е. изображение цветное, а не серое)
            self.assertFalse(np.array_equal(rendered_arr[:, :, 0], rendered_arr[:, :, 1]))

            # Проверка отображения чипов палитры в UI
            app.ui_manager.update_palette(chosen)
            app.update()
            self.assertGreater(len(app.ui_manager._palette_frame.winfo_children()), 0)

            # Тест экспорта HTML, TXT и сохранения/загрузки проекта (.sart.json) с пустым слоем Cyan
            with tempfile.TemporaryDirectory() as tmp:
                out_html = os.path.join(tmp, "assistant.html")
                out_txt = os.path.join(tmp, "scheme.txt")
                proj_path = os.path.join(tmp, "project.sart.json")
                from string_art_studio.exporters import (
                    export_html_assistant,
                    export_instructions_txt,
                    load_project_json,
                    save_project_json,
                )

                export_html_assistant(out_html, "GUI Test", app.coordinator.canvas_manager, app.state.layers)
                self.assertTrue(os.path.exists(out_html))
                export_instructions_txt(out_txt, app.state.layers, app.coordinator.canvas_manager)
                with open(out_txt, encoding="utf-8") as f:
                    self.assertIn("Нитей в слое: 0", f.read())

                save_project_json(
                    proj_path,
                    app.state.get_config(),
                    app.state.layers,
                    active_layer_idx=2,
                    step_scrubber=10,
                    image_path=app.state.original_image_path,
                )
                proj = load_project_json(proj_path)
                app.coordinator.apply_loaded_project(proj, source_path=proj_path)
                app.update()
                self.assertEqual(len(app.state.layers), 4)
                self.assertEqual(len(app.state.layers[0]["path"]), 0)
                self.assertEqual(app.state.active_layer_idx, 2)
                self.assertEqual(int(app.state.safe_get("step_scrubber")), 10)

            # Тест живого переключения языка RU -> EN -> RU без перезапуска
            app.set_language("en", persist=False)
            app.update()
            self.assertEqual(app.state.lang, "en")
            self.assertEqual(app.ui_manager.widgets["start_button"].cget("text"), "▶ Start (CSR OMP + Islands)")
            self.assertEqual(app.state.vars["color_mode"].get(), "Color CMYK (4 Threads)")
            self.assertEqual(app.state.vars["canvas_shape"].get(), "Circle")
            cfg_en = app.state.get_config()
            self.assertEqual(cfg_en["language"], "en")
            self.assertEqual(cfg_en["color_mode"], "Цветной CMYK (4 нити)")
            self.assertEqual(cfg_en["canvas_shape"], "Круг")

            app.set_language("ru", persist=False)
            app.update()
            self.assertEqual(app.state.lang, "ru")
            self.assertEqual(app.ui_manager.widgets["start_button"].cget("text"), "▶ Старт (CSR OMP + Острова)")
            self.assertEqual(app.state.vars["color_mode"].get(), "Цветной CMYK (4 нити)")
        finally:
            app.coordinator.shutdown()
            app.audio_assistant.shutdown()
            app.destroy()


if __name__ == "__main__":
    unittest.main()
