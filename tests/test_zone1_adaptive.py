# Зона 1, просунутий рівень: адаптивний Гаффман
import random
import unittest
from src import adaptive_huffman as ah
from src.record import pack_record
from tests.helpers import rec


class TestAdaptiveHuffman(unittest.TestCase):
    def check(self, data):
        stream, _ = ah.encode(data)
        self.assertEqual(ah.decode(stream, len(data)), data)
        return stream

    def test_edge_cases(self):
        self.check(b"")
        self.check(b"\x01")
        self.check(b"\x01" * 16)          # вироджений алфавіт
        self.check(bytes(range(256)))      # кожен символ новий

    def test_random_and_records(self):
        rng = random.Random(4)
        self.check(bytes(rng.randrange(256) for _ in range(3000)))
        self.check(b"".join(pack_record(rec(ts=1000 + 37 * i, metric=i * 11)) for i in range(48)))

    def test_skewed_data_compresses(self):
        data = b"a" * 900 + b"b" * 90 + b"c" * 10
        self.assertLess(len(self.check(data)), len(data) // 4)

    def test_no_table_needed(self):
        # для маленького блоку адаптивний коротший за канонічний разом з таблицею
        from src import huffman
        data = b"".join(pack_record(rec(ts=1000 + 37 * i)) for i in range(4))
        lengths = huffman.code_lengths(data)
        canon = 2 * len(lengths) + len(huffman.encode(data, lengths))
        self.assertLess(len(ah.encode(data)[0]), canon)


if __name__ == "__main__":
    unittest.main()
