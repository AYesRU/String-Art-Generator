# -*- coding: utf-8 -*-
import unittest
import numpy as np
from PIL import Image, ImageDraw

from string_art_studio import CANVAS_SHAPES
from string_art_studio.geometry import CanvasManager
from string_art_studio.preprocessing import auto_recommend_parameters, prepare_target_image_bundle


class TestGeometryAndPreprocessing(unittest.TestCase):
    def test_all_shapes_pins_and_masks(self):
        for shape in CANVAS_SHAPES:
            cm = CanvasManager(size=200, num_pins=120, shape=shape, physical_cm=50.0)
            self.assertEqual(cm.pin_coords.shape, (120, 2), msg=shape)
            self.assertTrue(np.all(cm.pin_coords >= 0.0), msg=shape)
            self.assertTrue(np.all(cm.pin_coords <= 200.0), msg=shape)

            mask = cm.get_polygon_mask()
            self.assertEqual(mask.shape, (200, 200), msg=shape)
            self.assertGreater(int(np.sum(mask == 255)), 5000, msg=shape)
            self.assertGreater(int(np.sum(mask == 0)), 100, msg=shape)

    def test_csr_and_flat_table_consistency(self):
        cm = CanvasManager(size=150, num_pins=60, shape="Круг")
        cm.build_line_tables()
        self.assertIsNotNone(cm.line_matrix)
        self.assertTrue(cm.flat_ready)
        for p1, p2 in [(0, 20), (5, 35), (10, 42)]:
            pix_a = cm.get_line_pixel_indices(p1, p2)
            pix_b = cm.get_line_pixel_indices(p2, p1)
            np.testing.assert_array_equal(pix_a, pix_b)
            lid = cm.line_dict[(p1, p2)]
            np.testing.assert_array_equal(pix_a, cm.line_matrix[lid].indices)

    def test_render_resolution_invariance_and_cmyk(self):
        cm = CanvasManager(size=500, num_pins=180, shape="Круг")
        rng = np.random.default_rng(7)
        path = rng.integers(0, 180, size=600).tolist()
        layers = [{"name": "Черная нить", "color_hex": "#111111", "path": path}]

        img_500 = cm.render_layers_to_numpy(layers, output_size=500)
        img_1600 = cm.render_layers_to_numpy(layers, output_size=1600)
        dark_500 = 1.0 - float(np.mean(img_500)) / 255.0
        dark_1600 = 1.0 - float(np.mean(img_1600)) / 255.0
        self.assertAlmostEqual(dark_500, dark_1600, delta=0.05)

        # CMYK и подсветка активного шага
        cmyk_layers = [
            {"name": "Циан (Cyan)", "color_hex": "#00a8e8", "path": path[:200]},
            {"name": "Маджента (Magenta)", "color_hex": "#e6007a", "path": path[200:400]},
            {"name": "Желтый (Yellow)", "color_hex": "#f5c400", "path": path[400:550]},
            {"name": "Черный (Black)", "color_hex": "#111111", "path": path[100:300]},
        ]
        rgb = cm.render_layers_to_numpy(cmyk_layers, output_size=400, max_step=50, active_layer_idx=1, active_chord=(path[249], path[250]))
        self.assertEqual(rgb.shape, (400, 400, 3))
        self.assertEqual(rgb.dtype, np.uint8)

    def test_rgba_transparent_composited_white_and_masked_for_all_shapes(self):
        rgba = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(rgba)
        draw.ellipse((60, 60, 140, 140), fill=(20, 20, 20, 255))

        for shape in ["Круг", "Прямоугольник (16:9)", "Шестиугольник"]:
            bundle = prepare_target_image_bundle(rgba, 200, shape=shape, color_mode="Цветной CMYK (4 нити)")
            # Углы за пределами маски обязаны оставаться чистыми белыми (255)
            self.assertEqual(int(bundle["gray"][0, 0]), 255, msg=shape)
            self.assertEqual(int(bundle["saliency"][0, 0]), 0, msg=shape)
            self.assertEqual(len(bundle["channels"]), 4)
            # Центр черного круга должен иметь ненулевую темноту в канале K
            self.assertGreater(float(bundle["channels"][3]["darkness"][100, 100]), 0.5)

        rec = auto_recommend_parameters(rgba, "Круг")
        self.assertGreaterEqual(rec["pins"], 220)
        self.assertGreaterEqual(rec["lines"], 950)

    def test_dda_zero_duplicate_pixels_and_pin_sub_matrices(self):
        for shape in CANVAS_SHAPES:
            cm = CanvasManager(size=160, num_pins=72, shape=shape)
            cm.build_line_tables()
            A = cm.line_matrix
            self.assertIsNotNone(A)
            # Проверяем отсутствие дубликатов пикселей в КАЖДОЙ хорде
            for lid in range(A.shape[0]):
                row = A.indices[A.indptr[lid] : A.indptr[lid + 1]]
                self.assertEqual(len(row), len(np.unique(row)), msg=f"Дубликаты в {shape}, chord {lid}")

            sub_mats = cm.ensure_pin_sub_matrices()
            self.assertEqual(len(sub_mats), cm.num_pins)
            probe = np.linspace(0.1, 1.0, cm.size * cm.size, dtype=np.float32)
            for pin_i in (0, 17, 50):
                lids = cm.pin_neighbor_lids[pin_i]
                np.testing.assert_allclose(sub_mats[pin_i].dot(probe), A[lids].dot(probe), atol=1e-4)

    def test_dark_board_mode_and_cmyk_per_channel_contours(self):
        # Темная доска: светлый круг на черном фоне
        dark_img = Image.new("RGB", (200, 200), (5, 5, 5))
        draw = ImageDraw.Draw(dark_img)
        draw.ellipse((65, 65, 135, 135), fill=(245, 245, 245))

        bundle_dark = prepare_target_image_bundle(
            dark_img, 200, shape="Круг", color_mode="Белая нить (Темная доска)"
        )
        self.assertTrue(bundle_dark["invert_board"])
        self.assertEqual(len(bundle_dark["channels"]), 1)
        self.assertEqual(bundle_dark["channels"][0]["name"], "Белая нить")
        self.assertEqual(float(bundle_dark["channels"][0]["darkness"][0, 0]), 0.0)
        self.assertGreater(float(bundle_dark["channels"][0]["darkness"][100, 100]), 0.7)

        cm = CanvasManager(size=400, num_pins=120, shape="Круг")
        path = [(i * 37) % 120 for i in range(180)]
        layers = [{"name": "Белая нить", "color_hex": "#f5f5f5", "path": path}]
        rgb_400 = cm.render_layers_to_numpy(layers, output_size=400, invert_board=True)
        rgb_1000 = cm.render_layers_to_numpy(layers, output_size=1000, invert_board=True)
        # Угол доски темный, а внутри есть светлые нити
        self.assertLessEqual(int(rgb_400[2, 2, 0]), 24)
        self.assertGreater(float(np.mean(rgb_400)), 25.0)
        self.assertAlmostEqual(float(np.mean(rgb_400)) / 255.0, float(np.mean(rgb_1000)) / 255.0, delta=0.05)

        # CMYK индивидуальные контуры по каналам (красный квадрат слева, синий справа)
        col_img = Image.new("RGB", (200, 200), (255, 255, 255))
        d2 = ImageDraw.Draw(col_img)
        d2.rectangle((45, 65, 95, 135), fill=(255, 10, 10))
        d2.rectangle((105, 65, 155, 135), fill=(10, 10, 255))
        bundle_cmyk = prepare_target_image_bundle(col_img, 200, shape="Квадрат", color_mode="Цветной CMYK (4 нити)")
        c_ch, _, y_ch, _ = bundle_cmyk["channels"]
        self.assertIn("edges", c_ch)
        self.assertFalse(np.array_equal(c_ch["edges"], y_ch["edges"]))
        self.assertFalse(np.allclose(c_ch["importance"], y_ch["importance"]))

    def test_gpu_cpu_renderer_parity(self):
        cm = CanvasManager(size=400, num_pins=120, shape="Круг")
        if not cm._gpu_render_enabled:
            self.skipTest("CUDA недоступен")
        path = [(i * 41) % 120 for i in range(250)]
        d_gpu = cm._accumulate_path_depth(path, len(path) - 1, 400, 1, 0.16)
        cm._gpu_render_enabled = False
        d_cpu = cm._accumulate_path_depth(path, len(path) - 1, 400, 1, 0.16)
        cm.release_gpu_cache()
        self.assertLess(float(np.max(np.abs(d_gpu - d_cpu))), 1e-3)


if __name__ == "__main__":
    unittest.main()
