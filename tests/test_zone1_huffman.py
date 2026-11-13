# Зона 1: канонічний Гаффман (C1: 4, 5)
import unittest
from src.errors import ReplicaError
from src import huffman


class TestHuffman(unittest.TestCase):
    def test_roundtrip(self):
        data = b"abracadabra" * 20 + bytes(range(256))
        lengths = huffman.code_lengths(data)
        enc = huffman.encode(data, lengths)
        self.assertEqual(huffman.decode(enc, lengths, len(data)), data)

    def test_prefix_free(self):
        lengths = huffman.code_lengths(b"aaaabbbccd")
        codes = huffman.canonical_codes(lengths)
        words = [format(codes[s], "0%db" % lengths[s]) for s in codes]
        for a in words:
            for b in words:
                if a != b:
                    self.assertFalse(b.startswith(a))

    def test_canonical_from_lengths_only(self):
        # кейс 4: декодер бачить лише довжини
        data = b"hello world, hello log"
        lengths = huffman.code_lengths(data)
        enc = huffman.encode(data, lengths)
        only_lengths = dict(sorted(lengths.items()))
        self.assertEqual(huffman.decode(enc, only_lengths, len(data)), data)

    def test_degenerate_alphabet(self):
        # кейс 5: один символ -> length=1, code=0
        lengths = huffman.code_lengths(b"\x01" * 16)
        self.assertEqual(lengths, {1: 1})
        self.assertEqual(huffman.canonical_codes(lengths), {1: 0})
        enc = huffman.encode(b"\x01" * 16, lengths)
        self.assertEqual(enc, b"\x00\x00")

    def test_kraft_violation(self):
        with self.assertRaises(ReplicaError):
            huffman.canonical_codes({0: 1, 1: 2})          # Kraft < 1
        with self.assertRaises(ReplicaError):
            huffman.canonical_codes({0: 1, 1: 1, 2: 1})    # Kraft > 1

    def test_max_length_15(self):
        # частоти Фібоначчі дають дуже глибоке дерево
        data = b""
        a, b = 1, 1
        for sym in range(25):
            data += bytes([sym]) * a
            a, b = b, a + b
        lengths = huffman.code_lengths(data)
        self.assertLessEqual(max(lengths.values()), 15)
        huffman.check_lengths(lengths)
        self.assertEqual(huffman.decode(huffman.encode(data, lengths), lengths, len(data)), data)


if __name__ == "__main__":
    unittest.main()
