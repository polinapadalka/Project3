# Зона 1: запис журналу (C1: 1)
import unittest
from src.record import pack_record, parse_record
from src.errors import ReplicaError
from tests.helpers import rec


class TestRecord(unittest.TestCase):
    def test_record_roundtrip(self):
        r = rec(ts=4000000000, src=4095, metric=-2**31, cat=15, sev=7, crit=1, flags=3)
        b = pack_record(r)
        self.assertEqual(len(b), 16)
        self.assertEqual(parse_record(b), r)

    def test_bad_source_rejected(self):
        with self.assertRaises(ReplicaError):
            pack_record(rec(src=0))
        with self.assertRaises(ReplicaError):
            pack_record(rec(src=4096))

    def test_bad_ranges_on_parse(self):
        b = bytearray(pack_record(rec()))
        b[12] = 16     # category > 15
        with self.assertRaises(ReplicaError):
            parse_record(bytes(b))


if __name__ == "__main__":
    unittest.main()
