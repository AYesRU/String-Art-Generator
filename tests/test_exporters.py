# -*- coding: utf-8 -*-
import os
import tempfile
import unittest

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
            self.assertIn("M2", gc_data)

            export_svg_vector(svg_p, layers, cm)
            with open(svg_p, encoding="utf-8") as f:
                svg_data = f.read()
            self.assertIn('id="layer_0"', svg_data)
            self.assertIn('id="layer_1"', svg_data)

            export_drill_template_svg(tpl_p, cm)
            with open(tpl_p, encoding="utf-8") as f:
                self.assertIn("450.00mm", f.read())

            export_png_ultra_hd(png_p, layers, cm, output_size=800)
            with Image.open(png_p) as im:
                self.assertEqual(im.size, (800, 800))

            export_html_assistant(html_p, "Test Art <v9.5>", cm, layers)
            with open(html_p, encoding="utf-8") as f:
                html = f.read()
            self.assertIn("border-radius: 12px;", html)
            self.assertIn("localStorage.getItem", html)
            self.assertNotIn("textContent = voiceEnabled ? '&#", html)
            self.assertIn("&lt;v9.5&gt;", html)

        # Для круга border-radius должен быть 50%
        cm_circle = CanvasManager(size=500, num_pins=120, shape="Круг")
        html_c = build_interactive_html5_assistant("Circle", cm_circle, layers[:1])
        self.assertIn("border-radius: 50%;", html_c)


if __name__ == "__main__":
    unittest.main()
