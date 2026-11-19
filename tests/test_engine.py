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
    compute_delta_insert_pin_np,
    compute_delta_remove_pin_np,
    compute_pixel_gain_omp,
    estimate_channel_line_budget,
    optimize_thread_count_delta,
    prune_chromosome_loops,
)
from string_art_studio.geometry import CanvasManager
from string_art_studio.preprocessing import prepare_target_image_bundle
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

        # Проверяем точность дельта-расчета (scratch-буфер vs эталон vs полный пересчет)
        depth = cm.compute_optical_depth_1d(path, fc_dt.line_alpha)
        err_before = fc_dt.calculate_total_error(depth)
        idx = 10
        prev_p, cur_p, next_p = path[idx - 1], path[idx], path[idx + 1]
        cand_p = (cur_p + 3) % cm.num_pins
        rem1 = cm.get_line_pixel_indices(prev_p, cur_p)
        rem2 = cm.get_line_pixel_indices(cur_p, next_p)
        add1 = cm.get_line_pixel_indices(prev_p, cand_p)
        add2 = cm.get_line_pixel_indices(cand_p, next_p)

        delta_err, _, _ = compute_delta_hinge_np(
            depth, fc_dt.target_darkness_flat, fc_dt.importance_flat, rem1, rem2, add1, add2, fc_dt.line_alpha, fc_dt.fearless_penalty
        )
        scratch = np.zeros(len(depth), dtype=np.int8)
        delta_scratch, _, _ = compute_delta_hinge_np(
            depth,
            fc_dt.target_darkness_flat,
            fc_dt.importance_flat,
            rem1,
            rem2,
            add1,
            add2,
            fc_dt.line_alpha,
            fc_dt.fearless_penalty,
            scratch=scratch,
        )
        self.assertEqual(int(np.count_nonzero(scratch)), 0)
        self.assertAlmostEqual(delta_scratch, delta_err, places=5)

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

        # Проверка инварианта min_pin_dist при кроссовере на 200 случайных попытках
        p2 = list(reversed(polished))
        for _ in range(200):
            child = isl._graph_splice_crossover(polished, p2, 0, len(polished) - 1)
            self.assertEqual(len(child), len(polished))
            for a, b in zip(child[:-1], child[1:]):
                self.assertGreaterEqual(isl._pin_dist(a, b, cm.num_pins), isl.min_pin_dist)

        # Проверка согласованности GPU и CPU + шаринга кэша CSR в VRAM
        if GPU_ENABLED_FLAG and fc_dt.gpu_backend.available:
            self.assertIs(fc_dt.gpu_backend.a_t_csr, fc_edge.gpu_backend.a_t_csr)
            batch = [path, polished, path[::-1], polished[::-1]]
            gpu_scores = fc_dt.gpu_backend.evaluate_batch(batch)
            cpu_scores = [fc_dt.calculate_single_fitness(c) for c in batch]
            self.assertIsNotNone(gpu_scores)
            for gs, cs in zip(gpu_scores, cpu_scores):
                self.assertAlmostEqual(gs, cs, places=4)
            cm.release_gpu_cache()
            self.assertIsNone(cm._gpu_a_t_csr)

    def test_prune_chromosome_loops(self):
        pruned, rem = prune_chromosome_loops([1, 2, 3, 2, 1, 5, 5, 6, 7, 6, 8])
        self.assertEqual(pruned, [1, 5, 6, 8])
        self.assertEqual(rem, 7)

    def test_cmyk_red_image_zero_cyan_and_variable_budgets(self):
        from PIL import Image, ImageDraw

        # Чисто красная фигура с темным центром — синего/циана нет совсем
        img = Image.new("RGB", (300, 300), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.ellipse((45, 45, 255, 255), fill=(225, 25, 20))
        draw.ellipse((100, 100, 200, 200), fill=(110, 10, 10))

        bundle = prepare_target_image_bundle(
            img, 240, clahe_clip=1.4, gamma=0.92, shape="Круг", color_mode="Цветной CMYK (4 нити)"
        )
        cm = CanvasManager(size=240, num_pins=120, shape="Круг")
        cm.build_line_tables()
        cfg = {
            "lines": 400,
            "line_weight": 66,
            "fearless_factor": 92,
            "contour_weight": 55,
            "structural_metric": "Контуры",
            "radon_enabled": False,
            "radon_influence": 0,
        }

        budgets = [
            estimate_channel_line_budget(ch["darkness"], bundle["mask"], cm, cfg["lines"], cfg["line_weight"])
            for ch in bundle["channels"]
        ]
        # Циан обязан иметь нулевой бюджет; Маджента и Желтый > Черного > 0
        self.assertEqual(budgets[0], 0, f"Циан должен получить бюджет 0 на красном фото, получено {budgets}")
        self.assertGreater(budgets[1], 100)
        self.assertGreater(budgets[2], 100)
        self.assertLessEqual(max(budgets), cfg["lines"])

        # Даже если принудительно запустить OMP на пустом Циане с лимитом 400 — он вернет [] (0 нитей)
        ch_c = bundle["channels"][0]
        fc_c = FitnessCalculator(
            bundle["gray"], ch_c["edges"], bundle["saliency"], ch_c["importance"], cm, cfg, {"status_queue": queue.Queue()}, target_darkness_override=ch_c["darkness"]
        )
        self.assertAlmostEqual(fc_c.calculate_single_fitness([]), 1.0, places=5)
        g_c = GreedyInitializer(cm, fc_c.target_darkness_flat, fc_c.importance_flat, cfg, None)
        p_c = g_c.run(400, lambda _s, _p: None, layer_label=ch_c["name"], line_alpha=fc_c.line_alpha)
        self.assertEqual(p_c, [])

        # Маджента при этом должна сплестись полноценно и строго в рамках лимита
        ch_m = bundle["channels"][1]
        fc_m = FitnessCalculator(
            bundle["gray"], ch_m["edges"], bundle["saliency"], ch_m["importance"], cm, cfg, {"status_queue": queue.Queue()}, target_darkness_override=ch_m["darkness"]
        )
        g_m = GreedyInitializer(cm, fc_m.target_darkness_flat, fc_m.importance_flat, cfg, None)
        p_m = g_m.run(budgets[1], lambda _s, _p: None, layer_label=ch_m["name"], line_alpha=fc_m.line_alpha)
        self.assertGreater(len(p_m) - 1, 40)
        self.assertLessEqual(len(p_m) - 1, cfg["lines"])

    def test_delta_remove_insert_and_adaptive_thread_count(self):
        size = 160
        cm = CanvasManager(size=size, num_pins=72, shape="Круг")
        cm.build_line_tables()
        img = np.full((size, size), 255, dtype=np.uint8)
        cv2.circle(img, (80, 80), 42, 30, -1)
        edges = cv2.Canny(img, 50, 150)
        imp = np.ones((size, size), dtype=np.float32)
        cfg = {
            "lines": 140,
            "line_weight": 72,
            "fearless_factor": 94,
            "contour_weight": 50,
            "structural_metric": "Контуры",
            "radon_influence": 0,
            "population_size": 8,
        }
        fc = FitnessCalculator(img, edges, edges, imp, cm, cfg, {"status_queue": queue.Queue()})
        scratch = np.zeros(size * size, dtype=np.int8)

        # 1) Точность compute_delta_remove_pin_np и compute_delta_insert_pin_np vs полный пересчет
        sample_path = [0, 25, 50, 12, 38, 62, 18, 44]
        depth = cm.compute_optical_depth_1d(sample_path, fc.line_alpha)
        err0 = fc.calculate_total_error(depth)

        # Удаление pin idx=2 (50): 25->50->12 -> 25->12
        d_rem, _, _ = compute_delta_remove_pin_np(
            depth,
            fc.target_darkness_flat,
            fc.importance_flat,
            cm.get_line_pixel_indices(25, 50),
            cm.get_line_pixel_indices(50, 12),
            cm.get_line_pixel_indices(25, 12),
            fc.line_alpha,
            fc.fearless_penalty,
            scratch=scratch,
        )
        self.assertEqual(int(np.count_nonzero(scratch)), 0)
        path_rem = sample_path[:2] + sample_path[3:]
        err_rem = fc.calculate_total_error(cm.compute_optical_depth_1d(path_rem, fc.line_alpha))
        self.assertAlmostEqual(d_rem, err_rem - err0, places=3)

        # Вставка pin=60 между 0 и 25: 0->25 -> 0->60->25
        d_ins, _, _ = compute_delta_insert_pin_np(
            depth,
            fc.target_darkness_flat,
            fc.importance_flat,
            cm.get_line_pixel_indices(0, 25),
            cm.get_line_pixel_indices(0, 60),
            cm.get_line_pixel_indices(60, 25),
            fc.line_alpha,
            fc.fearless_penalty,
            scratch=scratch,
        )
        self.assertEqual(int(np.count_nonzero(scratch)), 0)
        path_ins = [0, 60] + sample_path[1:]
        err_ins = fc.calculate_total_error(cm.compute_optical_depth_1d(path_ins, fc.line_alpha))
        self.assertAlmostEqual(d_ins, err_ins - err0, places=3)

        # 2) Дефицит нитей -> алгоритм ДОБАВЛЯЕТ нити, не превышая max_lines, и повышает фитнес
        short_path = [0, 36, 18, 54]
        f_short = fc.calculate_single_fitness(short_path)
        grown, f_grown, _, n_add = optimize_thread_count_delta(short_path, fc, max_lines=60, allow_add=True)
        self.assertGreater(n_add, 5)
        self.assertLessEqual(len(grown) - 1, 60)
        self.assertGreater(f_grown, f_short)
        self.assertAlmostEqual(f_grown, fc.calculate_single_fitness(grown), places=4)

        # 3) Избыток нитей на пустом слое -> алгоритм УМЕНЬШАЕТ до 0 нитей
        empty_img = np.full((size, size), 255, dtype=np.uint8)
        fc_empty = FitnessCalculator(empty_img, np.zeros_like(edges), np.zeros_like(edges), imp, cm, cfg, {"status_queue": queue.Queue()})
        excess_path = [(i * 23) % 72 for i in range(45)]
        f_bad = fc_empty.calculate_single_fitness(excess_path)
        trimmed, f_trimmed, n_rem, n_add2 = optimize_thread_count_delta(excess_path, fc_empty, max_lines=100, allow_add=True)
        self.assertEqual(trimmed, [])
        self.assertEqual(n_add2, 0)
        self.assertGreater(n_rem, 0)
        self.assertGreater(f_trimmed, f_bad)
        self.assertAlmostEqual(f_trimmed, 1.0, places=5)

        # 4) Переменная длина в IslandThread (кроссовер + дельта-спуск в рамках max_lines)
        isl = IslandThread(0, cfg, fc, {"status_queue": queue.Queue()}, grown, None, max_lines=75)
        p_shorter = grown[:-5]
        for _ in range(50):
            ch = isl._graph_splice_crossover(grown, p_shorter, 0, len(grown) - 1)
            self.assertLessEqual(len(ch) - 1, 75)
            for a, b in zip(ch[:-1], ch[1:]):
                self.assertGreaterEqual(isl._pin_dist(a, b, cm.num_pins), isl.min_pin_dist)

        pol, f_pol = isl._fast_delta_coordinate_descent(grown, 0, len(grown) - 1, 25)
        self.assertLessEqual(len(pol) - 1, 75)
        self.assertGreaterEqual(f_pol + 1e-6, f_grown)
        self.assertAlmostEqual(f_pol, fc.calculate_single_fitness(pol), places=4)


if __name__ == "__main__":
    unittest.main()
