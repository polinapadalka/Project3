# Зона 3: ознаки з вікном історії
import os
import unittest
import numpy as np
from src.features import make_features, load_csv
from tests.helpers import rec

ROOT = os.path.join(os.path.dirname(__file__), "..")


class TestFeatures(unittest.TestCase):
    def test_features_use_only_past(self):
        # ознаки перших 10 подій не мають залежати від того, що буде далі
        recs = [rec(ts=1000 + 100 * i, src=1 + i % 3) for i in range(20)]
        a = make_features(recs)
        b = make_features(recs[:10])
        self.assertTrue(np.allclose(a[:10], b))

    def test_same_source_delta(self):
        # зміна metric рахується від попередньої події ТОГО САМОГО джерела
        recs = [rec(ts=1000, src=1, metric=100), rec(ts=1100, src=2, metric=9000),
                rec(ts=1200, src=1, metric=110)]
        f = make_features(recs)
        self.assertAlmostEqual(f[2][5], np.log1p(10))

    def test_adjacent_codes(self):
        recs = [rec(ts=1000 + 50 * i, code=130 + i) for i in range(5)]
        self.assertEqual(make_features(recs)[4][4], 1.0)

    def test_csv_loads(self):
        recs, labels = load_csv(os.path.join(ROOT, "data", "train.csv"))
        self.assertEqual(len(recs), len(labels))
        self.assertTrue(set(labels) <= {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
