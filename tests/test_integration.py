# Інтеграція (зона 1): повний тракт на сеансі з data/session.csv (C1: 7, 8)
import os
import unittest
from src.features import load_csv
from src.pipeline import run_session
from src.record import pack_record
from tests.helpers import CFG

ROOT = os.path.join(os.path.dirname(__file__), "..")
LIMITS = CFG["пороги_втрати_критичних_подій"]


class TestFullRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs, _ = load_csv(os.path.join(ROOT, "data", "session.csv"))

    def test_session_bin_matches_csv(self):
        # session.bin - ті самі 1200 записів, серіалізовані по 16 байтів
        with open(os.path.join(ROOT, "data", "session.bin"), "rb") as f:
            raw = f.read()
        self.assertEqual(raw, b"".join(pack_record(r) for r in self.recs))

    def test_thresholds_all_scenarios(self):
        # втрата critical-подій не перевищує поріг сценарію з конфігу
        for sc in LIMITS:
            for seed in [CFG["реплікація"]["seed_моделі_реплікації"], 1, 2]:
                res = run_session(self.recs, CFG, sc, seed)
                self.assertLessEqual(1 - res["R"], LIMITS[sc], sc)

    def test_normal_no_rejected_replicas(self):
        res = run_session(self.recs, CFG, "штатний", seed=1)
        self.assertEqual(res["R"], 1.0)
        self.assertEqual(res["rejected"], {})

    def test_global_budget_respected(self):
        for sc in LIMITS:
            res = run_session(self.recs, CFG, sc, seed=1)
            st = res["store_stats"]
            self.assertLessEqual(st["global_used_bytes"], st["global_budget_bytes"])

    def test_empty_session(self):
        res = run_session([], CFG, "штатний", seed=1)
        self.assertEqual(res["restored_records"], 0)

    def test_time_limit(self):
        res = run_session(self.recs, CFG, "аварійний", seed=1)
        self.assertLess(res["t_s"], CFG["ліміт_часу_повного_прогону_секунд"])


if __name__ == "__main__":
    unittest.main()
