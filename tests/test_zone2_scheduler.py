# Зона 2: планувальник вузлів
import unittest
from src import scheduler
from tests.helpers import CFG

NODES = CFG["реплікація"]["сценарії"]["штатний"]["вузли"]


def full_free():
    return {name: NODES[name]["capacity_bytes"] for name in NODES}


class TestScheduler(unittest.TestCase):
    def test_copies_on_different_nodes(self):
        r = [3, 2, 1, 0]
        batches, placed, _ = scheduler.place(r, [80] * 4, [9, 5, 1, 1], NODES, full_free(), 1024, 60, False, 0.7)
        self.assertEqual(placed, [3, 2, 1, 0])
        for name in batches:
            self.assertEqual(len(batches[name]), len(set(batches[name])))   # не більше 1 разу на вузлі
        self.assertEqual(sum(1 for name in batches if 0 in batches[name]), 3)  # r=3 -> A, B і C

    def test_batch_limit(self):
        r = [1] * 48
        batches, _, sizes = scheduler.place(r, [128] * 48, [1] * 48, NODES, full_free(), 1024, 0, True, 0.7)
        for name in batches:
            self.assertLessEqual(sizes[name], 1024)

    def test_capacity_limit(self):
        # на вузол без місця нічого не кладемо
        free = full_free()
        free["A"] = 0
        batches, _, _ = scheduler.place([3, 3], [80, 80], [5, 5], NODES, free, 1024, 60, False, 0.7)
        self.assertEqual(batches["A"], [])

    def test_free_not_changed(self):
        # place лише планує - місце списує той, хто реально записує
        free = full_free()
        scheduler.place([1, 1], [80, 80], [5, 5], NODES, free, 1024, 60, False, 0.7)
        self.assertEqual(free, full_free())

    def test_small_batch_raw_is_cheaper(self):
        # 1 запис і таблиця на 100 символів: сирі байти дешевші за стиснення
        size, raw = scheduler.batch_encoding(1, 70, 100, False)
        self.assertTrue(raw)
        self.assertEqual(size, 75 + 14 + 16)


if __name__ == "__main__":
    unittest.main()
