# -*- coding: utf-8 -*-
import unittest
import cv2
import numpy as np

from string_art_studio.geometry import CanvasManager
from string_art_studio.radon import RadonGuide


class TestRadonGuide(unittest.TestCase):
    def test_radon_chord_alignment_across_angles(self):
        size = 300
        cm = CanvasManager(size=size, num_pins=120, shape="Круг")
        cm.build_line_tables()

        # Рисуем четкую диагональную линию, соединяющую гвозди 15 и 75 (под углом)
        p_a, p_b = 15, 75
        p_perp_a, p_perp_b = 45, 105  # Перпендикулярная хорда (в белой области)

        img = np.full((size, size), 255, dtype=np.uint8)
        pt1 = tuple(np.round(cm.pin_coords[p_a]).astype(int))
        pt2 = tuple(np.round(cm.pin_coords[p_b]).astype(int))
        cv2.line(img, pt1, pt2, 0, thickness=4)

        edges = np.zeros((size, size), dtype=np.uint8)
        cv2.line(edges, pt1, pt2, 255, thickness=3)

        rg = RadonGuide(img, edges, cm, {"radon_enabled": True})
        self.assertIsNotNone(rg.sinogram)
        self.assertIsNotNone(rg.fbp_recon)

        w_aligned = float(rg.pin_adj_weights[p_a, p_b])
        w_perp = float(rg.pin_adj_weights[p_perp_a, p_perp_b])
        self.assertGreater(w_aligned, 0.45, f"Ожидался высокий вес Радона по линии, получено {w_aligned}")
        self.assertGreater(w_aligned, w_perp * 3.0)

    def test_radon_non_circular_shapes(self):
        for shape in ["Квадрат", "Прямоугольник (16:9)", "Овал", "Шестиугольник"]:
            cm = CanvasManager(size=200, num_pins=80, shape=shape)
            cm.build_line_tables()
            img = np.full((200, 200), 255, dtype=np.uint8)
            cv2.circle(img, (100, 100), 40, 20, -1)
            edges = cv2.Canny(img, 50, 150)
            rg = RadonGuide(img, edges, cm, {"radon_enabled": True})
            self.assertIsNotNone(rg.sinogram, msg=shape)
            self.assertEqual(rg.fbp_recon.shape, (250, 250), msg=shape)

    def test_radon_density_override_and_skip_fbp(self):
        size = 240
        cm = CanvasManager(size=size, num_pins=96, shape="Круг")
        cm.build_line_tables()

        p_a, p_b = 12, 60
        p_perp_a, p_perp_b = 36, 84
        density = np.zeros((size, size), dtype=np.float32)
        pt1 = tuple(np.round(cm.pin_coords[p_a]).astype(int))
        pt2 = tuple(np.round(cm.pin_coords[p_b]).astype(int))
        cv2.line(density, pt1, pt2, 1.0, thickness=4)
        edges = (density * 255.0).astype(np.uint8)
        dummy_gray = np.zeros((size, size), dtype=np.uint8)

        rg = RadonGuide(dummy_gray, edges, cm, {"radon_enabled": True}, compute_fbp=False, density_override=density)
        self.assertIsNotNone(rg.sinogram)
        self.assertIsNone(rg.fbp_recon)
        self.assertEqual(len(rg.top_neighbors_cache), cm.num_pins)
        self.assertGreater(float(rg.pin_adj_weights[p_a, p_b]), float(rg.pin_adj_weights[p_perp_a, p_perp_b]) * 3.0)


if __name__ == "__main__":
    unittest.main()
