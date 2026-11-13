# Зона 2, просунутий рівень: оптимальні контрольні точки (D1 #13)
import itertools
import random
import unittest
from src import checkpoints as cp


class TestCheckpoints(unittest.TestCase):
    def test_equals_brute_force(self):
        rng = random.Random(2)
        for _ in range(20):
            n = 8
            w = [rng.randint(50, 150) for _ in range(n)]
            snap = rng.randint(100, 2000)
            rate = rng.uniform(0.001, 0.1)
            best = min(cp.cost_of(list(c) + [n], w, snap, rate)
                       for k in range(n) for c in itertools.combinations(range(1, n), k))
            cost, points = cp.optimal_checkpoints(w, snap, rate)
            self.assertAlmostEqual(cost, best, places=6)
            self.assertAlmostEqual(cp.cost_of(points, w, snap, rate), cost, places=6)
            self.assertEqual(points[-1], n)

    def test_not_worse_than_fixed_step(self):
        rng = random.Random(3)
        w = [rng.randint(60, 140) for _ in range(300)]
        for rate in [0.005, 0.05, 0.2]:
            cost, _ = cp.optimal_checkpoints(w, 4096, rate)
            for step in [1, 10, 48, 100, 300]:
                self.assertLessEqual(cost, cp.fixed_step(w, 4096, rate, step)[0] + 1e-6)

    def test_extremes(self):
        w = [100] * 20
        # дуже дорогий знімок -> одна точка в кінці; безкоштовний -> після кожного запису
        self.assertEqual(cp.optimal_checkpoints(w, 1e9, 0.01)[1], [20])
        self.assertEqual(len(cp.optimal_checkpoints(w, 0.0, 0.01)[1]), 20)


if __name__ == "__main__":
    unittest.main()
