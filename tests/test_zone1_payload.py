# Зона 1: payload (C1: 1, 2, 3, 11)
import unittest
from src.record import pack_record
from src.errors import ReplicaError
from src import huffman
from src.payload import encode_payload, decode_payload
from tests.helpers import rec, raw_payload


class TestPayload(unittest.TestCase):
    def test_one_record(self):
        b = pack_record(rec())
        lengths = huffman.code_lengths(b)
        out, _ = decode_payload(encode_payload([b], lengths))
        self.assertEqual(out, [b])

    def test_empty_log(self):
        # кейс 2: порожній журнал без аварії
        out, _ = decode_payload(encode_payload([], {}))
        self.assertEqual(out, [])

    def test_no_compression(self):
        # кейс 3
        recs = [pack_record(rec(ts=1000 + i, metric=i * 977)) for i in range(10)]
        p = encode_payload(recs, {}, no_compression=True)
        out, _ = decode_payload(p)
        self.assertEqual(out, recs)

    def test_table_includes_unused_symbols(self):
        # таблиця вікна може мати символи, яких немає в batch
        recs = [pack_record(rec(ts=1000 + i)) for i in range(5)]
        lengths = huffman.code_lengths(b"".join(recs) + bytes(range(256)))
        out, _ = decode_payload(encode_payload(recs[1:3], lengths))
        self.assertEqual(out, recs[1:3])

    def test_symbol_count_invariant(self):
        b = pack_record(rec())
        p = raw_payload(1, 8, 0, [], 15, b)
        with self.assertRaises(ReplicaError) as e:
            decode_payload(p)
        self.assertEqual(e.exception.code, "E_PAYLOAD")

    def test_reserved_nonzero(self):
        p = raw_payload(1, 8, 1, [], 16, pack_record(rec()))
        with self.assertRaises(ReplicaError):
            decode_payload(p)

    def test_unknown_extension(self):
        p = raw_payload(1, 8 | 1, 0, [], 16, pack_record(rec()))
        with self.assertRaises(ReplicaError):
            decode_payload(p)

    def test_extra_bitstream_byte(self):
        b = pack_record(rec())
        lengths = huffman.code_lengths(b)
        p = encode_payload([b], lengths)
        table = sorted(lengths.items())
        stream = huffman.encode(b, lengths) + b"\x00"
        bad = raw_payload(1, 0, 0, table, 16, stream)
        with self.assertRaises(ReplicaError):
            decode_payload(bad)

    def test_garbage_never_crashes(self):
        # погані дані -> тільки ReplicaError, без IndexError/struct.error
        import random
        rng = random.Random(5)
        b = pack_record(rec())
        good = encode_payload([b], huffman.code_lengths(b))
        for _ in range(300):
            bad = bytearray(good)
            for _ in range(3):
                bad[rng.randrange(len(bad))] = rng.randrange(256)
            try:
                decode_payload(bytes(bad[:rng.randint(0, len(bad))]))
            except ReplicaError:
                pass


if __name__ == "__main__":
    unittest.main()
