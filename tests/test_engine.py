# -*- coding: utf-8 -*-
import queue
import unittest
import cv2
import numpy as np

from string_art_studio.engine import (
    GPU_ENABLED_FLAG,
    FitnessCalculator,
    GreedyInitializer,
    IslandThread,
    compute_delta_hinge_np,
    compute_pixel_gain_omp,
    prune_chromosome_loops,
)
from string_art_studio.geometry import CanvasManager
from string_art_studio.radon import RadonGuide


def _ref_evaluate_line_omp(residual, target_darkness, importance, indices, line_alpha, fearless):
    n = len(indices)
    if n == 0:
        return -1e12
    net = 0.0
    bright = 0
    for idx in indices:
        res = float(residual[idx])
        tar = float(target_darkness[idx])
        imp = float(importance[idx])
        if res > 0.02:
            after = res - line_alpha
            if after >= 0.0:
                net += (res * res - after * after) * imp
            else:
                net += (res * res) * imp - (after * after) * imp * (1.1 + fearless * 1.6)
        else:
            over = line_alpha - res
            wg = 2.2 if tar < 0.08 else 1.2
            if tar < 0.08:
                bright += 1
            net -= (over * over) * imp * fearless * wg
    br = bright / n
    if br > 0.78:
        net -= (br - 0.78) * n * (line_alpha * line_alpha) * fearless * 1.2
    return net / (n ** 0.90)


class TestEngineAndOptimizers(unittest.TestCase):
    def test_pixel_gain_omp_equivalence(self):
        rng = np.random.default_rng(123)
        cm = CanvasManager(size=120, num_pins=60, shape="Круг")
        cm.build_line_tables()
        flat_size = 120 * 120
        residual = rng.uniform(-0.2, 0.9, flat_size).astype(np.float32)
        target = rng.uniform(0.0, 1.0, flat_size).astype(np.float32)
        imp = rng.uniform(0.5, 3.0, flat_size).astype(np.float32)
        alpha = 0.16
        fearless = 0.94

        gain = compute_pixel_gain_omp(residual, target, imp, alpha, fearless)
        A = cm.line_matrix
        lens = np.maximum(1.0, (A.indptr[1:] - A.indptr[:-1]).astype(np.float32))
        bright_hits = A.dot((target < 0.08).astype(np.float32))
        bright_pen = np.maximum(0.0, (bright_hits / lens) - 0.78) * lens * (alpha ** 2) * fearless * 1.2
        scores = (A.dot(gain) - bright_pen) / (lens ** 0.90)

        for lid in range(0, min(50, A.shape[0]), 5):
            pix = A.indices[A.indptr[lid] : A.indptr[lid + 1]]
            ref = _ref_evaluate_line_omp(residual, target, imp, pix, alpha, fearless)
            self.assertAlmostEqual(float(scores[lid]), ref, places=4)

    def test_delta_hinge_and_gpu_cpu_consistency(self):
        size = 160
        cm = CanvasManager(size=size, num_pins=72, shape="Круг")
        cm.build_line_tables()
        img = np.full((size, size), 240, dtype=np.uint8)
        cv2.circle(img, (80, 80), 40, 40, -1)
        edges = cv2.Canny(img, 50, 150)
        sal = edges.copy()
        imp = np.ones((size, size), dtype=np.float32)

        cfg = {
            "lines": 120,
            "line_weight": 72,
            "fearless_factor": 94,
            "contour_weight": 55,
            "structural_metric": "Преобразование расстояний",
            "radon_influence": 35,
        }
        fc_dt = FitnessCalculator(img, edges, sal, imp, cm, cfg, {"status_queue": queue.Queue()})
        self.assertIsNotNone(fc_dt.distance_transform_map)

        cfg_edge = dict(cfg, structural_metric="Контуры")
        fc_edge = FitnessCalculator(img, edges, sal, imp, cm, cfg_edge, {"status_queue": queue.Queue()})
        self.assertFalse(np.allclose(fc_dt.importance_flat, fc_edge.importance_flat))

        rg = RadonGuide(img, edges, cm, {"radon_enabled": True})
        greedy = GreedyInitializer(cm, fc_dt.target_darkness_flat, fc_dt.importance_flat, cfg, rg)
        path = greedy.run(80, lambda _s, _p: None)
        self.assertGreaterEqual(len(path), 30)

        # Проверяем точность дельта-расчета (учитывая перекрывающиеся пиксели)
        depth = cm.compute_optical_depth_1d(path, fc_dt.line_alpha)
        err_before = fc_dt.calculate_total_error(depth)
        idx = 10
        prev_p, cur_p, next_p = path[idx - 1], path[idx], path[idx + 1]
        cand_p = (cur_p + 3) % cm.num_pins
        rem1 = cm.get_line_pixel_indices(prev_p, cur_p)
        rem2 = cm.get_line_pixel_indices(cur_p, next_p)
        add1 = cm.get_line_pixel_indices(prev_p, cand_p)
        add2 = cm.get_line_pixel_indices(cand_p, next_p)

        delta_err, af, nd = compute_delta_hinge_np(
            depth, fc_dt.target_darkness_flat, fc_dt.importance_flat, rem1, rem2, add1, add2, fc_dt.line_alpha, fc_dt.fearless_penalty
        )
        new_path = list(path)
        new_path[idx] = cand_p
        depth_after = cm.compute_optical_depth_1d(new_path, fc_dt.line_alpha)
        err_after = fc_dt.calculate_total_error(depth_after)
        self.assertAlmostEqual(delta_err, err_after - err_before, places=2)

        # Дельта-спуск не должен ухудшать фитнес
        isl = IslandThread(0, {"population_size": 8}, fc_dt, {"status_queue": queue.Queue()}, path, rg)
        f_before = fc_dt.calculate_single_fitness(path)
        polished, f_after = isl._fast_delta_coordinate_descent(path, 0, len(path) - 1, 20)
        self.assertGreaterEqual(f_after + 1e-6, f_before)
        self.assertAlmostEqual(f_after, fc_dt.calculate_single_fitness(polished), places=4)

        # Проверка согласованности GPU и CPU, если CUDA доступна
        if GPU_ENABLED_FLAG and fc_dt.gpu_backend.available:
            batch = [path, polished, path[::-1], polished[::-1]]
            gpu_scores = fc_dt.gpu_backend.evaluate_batch(batch)
            cpu_scores = [fc_dt.calculate_single_fitness(c) for c in batch]
            self.assertIsNotNone(gpu_scores)
            for gs, cs in zip(gpu_scores, cpu_scores):
                self.assertAlmostEqual(gs, cs, places=4)

    def test_prune_chromosome_loops(self):
        pruned, rem = prune_chromosome_loops([1, 2, 3, 2, 1, 5, 5, 6, 7, 6, 8])
        self.assertEqual(pruned, [1, 5, 6, 8])
        self.assertEqual(rem, 7)


if __name__ == "__main__":
    unittest.main()
