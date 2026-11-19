# -*- coding: utf-8 -*-
import os
import re
import tempfile
import unittest

import numpy as np
from PIL import Image

from string_art_studio.exporters import (
    build_interactive_html5_assistant,
    export_csv,
    export_drill_template_svg,
    export_gcode_cnc,
    export_html_assistant,
    export_instructions_txt,
    export_png_ultra_hd,
    export_svg_vector,
    load_project_json,
    save_project_json,
)
from string_art_studio.geometry import CanvasManager


class TestExporters(unittest.TestCase):
    def test_all_exporters_mono_and_cmyk(self):
        cm = CanvasManager(size=500, num_pins=120, shape="Квадрат", physical_cm=45.0)
        layers = [
            {"name": "Циан (Cyan)", "color_hex": "#00a8e8", "path": [0, 30, 65, 95, 20, 75]},
            {"name": "Черный (Black)", "color_hex": "#111111", "path": [5, 45, 85, 15, 60]},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            txt_p = os.path.join(tmp, "out.txt")
            csv_p = os.path.join(tmp, "out.csv")
            gc_p = os.path.join(tmp, "out.gcode")
            svg_p = os.path.join(tmp, "out.svg")
            tpl_p = os.path.join(tmp, "tpl.svg")
            png_p = os.path.join(tmp, "out.png")
            html_p = os.path.join(tmp, "out.html")

            total_m = export_instructions_txt(txt_p, layers, cm)
            self.assertGreater(total_m, 0.5)
            with open(txt_p, encoding="utf-8") as f:
                txt_data = f.read()
            self.assertIn("СЛОЙ 1", txt_data)
            self.assertIn("СЛОЙ 2", txt_data)

            export_csv(csv_p, layers, cm)
            with open(csv_p, encoding="utf-8-sig") as f:
                self.assertIn("От_гвоздя,К_гвоздю", f.readline())

            export_gcode_cnc(gc_p, layers, cm)
            with open(gc_p, encoding="utf-8") as f:
                gc_data = f.read()
            self.assertIn("G21", gc_data)
            self.assertIn("Cartesian Y-up", gc_data)
            self.assertIn("M2", gc_data)
            # Pin 0 на квадрате находится вверху (малый y_img) -> в ЧПУ Y-up должен иметь большой Y (~434 мм)
            m0 = re.search(r"G0 X([\d.]+) Y([\d.]+)", gc_data)
            self.assertIsNotNone(m0)
            self.assertGreater(float(m0.group(2)), 400.0)

            export_svg_vector(svg_p, layers, cm)
            with open(svg_p, encoding="utf-8") as f:
                svg_data = f.read()
            self.assertIn('id="layer_0"', svg_data)
            self.assertIn('id="layer_1"', svg_data)

            export_drill_template_svg(tpl_p, cm)
            with open(tpl_p, encoding="utf-8") as f:
                tpl_data = f.read()
            self.assertIn("450.00mm", tpl_data)
            self.assertIn("<polygon", tpl_data)
            self.assertIn('id="calibration_bar"', tpl_data)
            self.assertIn("100 mm", tpl_data)

            export_png_ultra_hd(png_p, layers, cm, output_size=800)
            with Image.open(png_p) as im:
                self.assertEqual(im.size, (800, 800))

            export_html_assistant(html_p, "Test Art <v9.6>", cm, layers)
            with open(html_p, encoding="utf-8") as f:
                html = f.read()
            self.assertIn("border-radius: 12px;", html)
            self.assertIn("PROJECT_KEY", html)
            self.assertIn("CUM_LEN", html)
            self.assertIn('id="jump-input"', html)
            self.assertIn('id="speed-select"', html)
            self.assertIn('id="btn-theme"', html)
            self.assertNotIn("textContent = voiceEnabled ? '&#", html)
            self.assertIn("&lt;v9.6&gt;", html)

            # Темная доска (Белая нить): SVG, PNG, HTML5
            dark_layer = [{"name": "Белая нить", "color_hex": "#f5f5f5", "path": [0, 35, 70, 105, 15]}]
            svg_dark_p = os.path.join(tmp, "dark.svg")
            png_dark_p = os.path.join(tmp, "dark.png")
            export_svg_vector(svg_dark_p, dark_layer, cm, invert_board=True)
            with open(svg_dark_p, encoding="utf-8") as f:
                self.assertIn('fill="#121418"', f.read())
            export_png_ultra_hd(png_dark_p, dark_layer, cm, output_size=400, invert_board=True)
            with Image.open(png_dark_p) as im_d:
                arr_d = np.array(im_d)
                self.assertLessEqual(int(arr_d[2, 2, 0]), 24)

            html_dark = build_interactive_html5_assistant("Dark Project", cm, dark_layer, invert_board=True)
            self.assertIn('class="dark-ui"', html_dark)
            self.assertIn('"#121418"', html_dark)

            # Уникальность ключа localStorage между разными проектами
            html_other = build_interactive_html5_assistant("Other Project", cm, layers)
            k1 = re.search(r'PROJECT_KEY = "([^"]+)"', html).group(1)
            k2 = re.search(r'PROJECT_KEY = "([^"]+)"', html_other).group(1)
            self.assertNotEqual(k1, k2)

            # Сохранение и загрузка проекта (.sart.json)
            proj_p = os.path.join(tmp, "demo.sart.json")
            save_project_json(
                proj_p,
                config={"pins": 120, "canvas_shape": "Квадрат", "color_mode": "Цветной CMYK (4 нити)"},
                layers=layers,
                active_layer_idx=1,
                step_scrubber=3,
                image_path="photo.jpg",
            )
            loaded = load_project_json(proj_p)
            self.assertEqual(loaded["active_layer_idx"], 1)
            self.assertEqual(loaded["step_scrubber"], 3)
            self.assertEqual(loaded["config"]["pins"], 120)
            self.assertEqual(loaded["layers"][0]["path"], layers[0]["path"])

            bad_p = os.path.join(tmp, "bad.json")
            with open(bad_p, "w", encoding="utf-8") as f:
                f.write('{"format": "unknown"}')
            with self.assertRaises(ValueError):
                load_project_json(bad_p)

        # Для круга border-radius должен быть 50%
        cm_circle = CanvasManager(size=500, num_pins=120, shape="Круг")
        html_c = build_interactive_html5_assistant("Circle", cm_circle, layers[:1])
        self.assertIn("border-radius: 50%;", html_c)


if __name__ == "__main__":
    unittest.main()
