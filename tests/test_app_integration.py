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

            # Проверка 6 вкладок результатов (включая отдельную вкладку Палитры нитей) и 5 графических холстов
            nb = app.ui_manager.widgets["results_notebook"]
            self.assertEqual(len(nb.tabs()), 6)
            self.assertIn("palette", app.ui_manager._results_tab_keys)
            self.assertIn("palette_tab_frame", app.ui_manager.widgets)
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
                {"name": "Темный желтовато-коричневый", "color_hex": "#3F2512", "path": [20, 60, 100, 20]},
            ]
            rendered_pil = app.coordinator.canvas_manager.render_layers_to_pil(test_layers, output_size=100)
            rendered_arr = np.array(rendered_pil)
            # R, G, B каналы не должны быть идентичны (т.е. изображение цветное, а не серое)
            self.assertFalse(np.array_equal(rendered_arr[:, :, 0], rendered_arr[:, :, 1]))

            # Проверка 8-цветного адаптивного цветоделения Бера-Ламберта: ни один присутствующий цвет не обнуляется
            multi_img = np.full((120, 120, 3), 255, dtype=np.uint8)
            multi_img[10:40, 10:55] = [250, 165, 25]   # Оранжево-желтый
            multi_img[10:40, 65:110] = [45, 115, 210]  # Синий
            multi_img[45:75, 10:55] = [35, 140, 50]    # Зеленый
            multi_img[45:75, 65:110] = [205, 30, 35]   # Красный
            multi_img[80:110, 10:55] = [220, 170, 135] # Телесный/бежевый
            multi_img[80:110, 65:110] = [65, 35, 20]   # Темно-коричневый
            sel_multi = app.coordinator._select_adaptive_colors(multi_img, n=6, invert_board=False)
            self.assertEqual(len(sel_multi), 6)
            mask_multi = np.full((120, 120), 255, dtype=np.uint8)
            dens_multi = app.coordinator._decompose_adaptive_channels(
                multi_img.astype(np.float32) / 255.0, sel_multi, mask_multi, invert_board=False
            )
            self.assertEqual(len(dens_multi), 6)
            active_channels = sum(1 for d_m in dens_multi if int(np.sum(d_m > 0.12)) >= 40)
            self.assertEqual(active_channels, 6, "Все 6 подобранных цветов должны получить информативную плотность")

            # Запрос N=12 на 6-цветной картинке не должен плодить фантомные цвета и близкие дубли (CIEDE2000 >= 10)
            sel_over = app.coordinator._select_adaptive_colors(multi_img, n=12, invert_board=False)
            self.assertLessEqual(len(sel_over), 8)
            self.assertGreaterEqual(len(sel_over), 5)
            over_rgb = np.array([[c["rgb"][0], c["rgb"][1], c["rgb"][2]] for c in sel_over], dtype=np.float32) / 255.0
            over_lab = app.coordinator._rgb_to_lab(over_rgb)
            for idx_a in range(len(over_lab)):
                for idx_b in range(idx_a + 1, len(over_lab)):
                    de00 = float(app.coordinator._ciede2000_vec(over_lab[idx_a], over_lab[idx_b : idx_b + 1])[0])
                    self.assertGreaterEqual(de00, 10.0)

            # Проверка подробной вкладки «Палитра нитей» (включая 0-слой Cyan, метры и катушки)
            app.ui_manager.refresh_palette_tab()
            app.update()
            self.assertEqual(len(app.ui_manager._palette_cards_frame.winfo_children()), 4)
            cmyk_spec = app.ui_manager._build_palette_text_spec()
            self.assertIn("HEX:", cmyk_spec)
            self.assertIn("RGB:", cmyk_spec)
            self.assertIn("CMYK:", cmyk_spec)
            self.assertIn("0 нитей", cmyk_spec)
            self.assertIn("Катушки 500 м", cmyk_spec)

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
                    active_layer_idx=-1,
                    step_scrubber=10,
                    image_path=app.state.original_image_path,
                )
                proj = load_project_json(proj_path)
                self.assertEqual(proj["active_layer_idx"], -1)
                app.coordinator.apply_loaded_project(proj, source_path=proj_path)
                app.update()
                self.assertEqual(len(app.state.layers), 4)
                self.assertEqual(len(app.state.layers[0]["path"]), 0)
                self.assertEqual(app.state.active_layer_idx, -1)
                self.assertEqual(int(app.state.safe_get("step_scrubber")), 10)

            # Тест сквозного скроллинга шагов в режиме «Все цвета» (-1) и полного разрешения preview
            app.redraw_target_canvas()
            self.assertEqual(app.state.preprocessed_rgb_np.shape, (500, 500, 3))
            cm = app.coordinator.canvas_manager
            rgb_step1 = cm.render_layers_to_numpy(app.state.layers, output_size=120, max_step=1, active_layer_idx=-1)
            rgb_full = cm.render_layers_to_numpy(app.state.layers, output_size=120, max_step=None, active_layer_idx=-1)
            self.assertGreater(int(np.mean(rgb_step1)), int(np.mean(rgb_full)))

            # Тест полного жизненного цикла адаптивного режима: синтез + ручной Prune + Polish + переключение языка на палитре
            app.state.vars["color_mode"].set("Адаптивный подбор (N цветов)")
            app.ui_manager._update_adaptive_n_visibility()
            app.state.vars["adaptive_n_colors"].set(3)
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
            self.assertFalse(app.state.is_running, "Адаптивный синтез не завершился вовремя")
            self.assertIsNone(app.coordinator.active_fitness_calc)
            self.assertGreaterEqual(len(app.state.layers), 2)
            for lyr in app.state.layers:
                self.assertGreaterEqual(len(lyr.get("path", [])) - 1, 5, "Адаптивный слой не должен состоять из 1 нити")
            adapt_names_before = [l["name"] for l in app.state.layers]
            for nm in adapt_names_before:
                self.assertNotEqual(nm, "Черная нить")

            from unittest import mock as _mock
            with _mock.patch("tkinter.messagebox.showinfo"):
                app.coordinator.prune_chromosome(silent=False)
            self.assertGreaterEqual(len(app.state.layers), 1)
            self.assertNotIn("Черная нить", [l["name"] for l in app.state.layers])

            app.state.active_layer_idx = -1
            app.coordinator.start_evolution_only()
            deadline = time.time() + 25.0
            while app.state.is_running and time.time() < deadline:
                app.update()
                time.sleep(0.03)
            self.assertFalse(app.state.is_running, "Адаптивная доводка не завершилась вовремя")
            for nm in [l["name"] for l in app.state.layers]:
                self.assertNotEqual(nm, "Черная нить")

            # Тест живого переключения языка RU -> EN -> RU без перезапуска (включая вкладку Палитры нитей)
            app.set_language("en", persist=False)
            app.update()
            self.assertEqual(app.state.lang, "en")
            self.assertEqual(app.ui_manager.widgets["start_button"].cget("text"), "▶ Start (CSR OMP + Islands)")
            self.assertEqual(app.state.vars["color_mode"].get(), "Adaptive Color Matching (N colors)")
            self.assertEqual(app.state.vars["canvas_shape"].get(), "Circle")
            nb = app.ui_manager.widgets["results_notebook"]
            self.assertEqual(nb.tab(nb.tabs()[1], "text"), "🎨 Thread Palette")
            cards = app.ui_manager._palette_cards_frame.winfo_children()
            self.assertEqual(len(cards), len(app.state.layers))
            en_spec = app.ui_manager._build_palette_text_spec()
            self.assertTrue(en_spec)
            self.assertFalse(any("\u0400" <= ch <= "\u04ff" for ch in en_spec))
            nb.select(1)
            app.update()
            app.ui_manager._select_layer_by_index(0, switch_to_canvas=True)
            app.update()
            self.assertEqual(app.state.active_layer_idx, 0)
            self.assertEqual(nb.index(nb.select()), 0)
            cfg_en = app.state.get_config()
            self.assertEqual(cfg_en["language"], "en")
            self.assertEqual(cfg_en["color_mode"], "Адаптивный подбор (N цветов)")
            self.assertEqual(cfg_en["canvas_shape"], "Круг")

            app.set_language("ru", persist=False)
            app.update()
            self.assertEqual(app.state.lang, "ru")
            self.assertEqual(app.ui_manager.widgets["start_button"].cget("text"), "▶ Старт (CSR OMP + Острова)")
            self.assertEqual(app.state.vars["color_mode"].get(), "Адаптивный подбор (N цветов)")
            self.assertEqual(nb.tab(nb.tabs()[1], "text"), "🎨 Палитра нитей")
            self.assertIn("Спецификация нитей", app.ui_manager._build_palette_text_spec())
        finally:
            app.coordinator.shutdown()
            app.audio_assistant.shutdown()
            app.destroy()


if __name__ == "__main__":
    unittest.main()
