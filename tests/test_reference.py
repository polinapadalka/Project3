# Зона 1: еталонні артефакти викладача (C1: 12)
import json
import os
import unittest
from src import crypto, huffman
from src.errors import ReplicaError
from src.payload import decode_payload, encode_payload
from src.record import pack_record, parse_record

REF = os.path.join(os.path.dirname(__file__), "..", "reference")


def read(name):
    with open(os.path.join(REF, name), "rb") as f:
        return f.read()


def read_json(name):
    with open(os.path.join(REF, name), encoding="utf-8") as f:
        return json.load(f)


class TestReference(unittest.TestCase):
    def test_good_replicas_decode(self):
        for name in sorted(os.listdir(REF)):
            if not (name.startswith("ref_") and name.endswith(".json")):
                continue
            d = read_json(name)
            data = read(name[:-5] + ".bin")
            self.assertEqual(len(data), d["довжина_репліки"], name)
            k_enc, k_mac = crypto.derive_keys(d["пароль"], bytes.fromhex(d["salt_hex"]))
            _, payload = crypto.open_replica(data, k_enc, k_mac)
            records, classes = decode_payload(payload)
            self.assertEqual([parse_record(r) for r in records], d["записи"], name)
            self.assertEqual(classes, d["class_table"], name)

    def test_canonical_codes_match(self):
        # наші канонічні коди збігаються з кодами в еталоні біт у біт
        for name in sorted(os.listdir(REF)):
            if not (name.startswith("ref_") and name.endswith(".json")):
                continue
            d = read_json(name)
            table = {t["символ"]: t["довжина"] for t in d["таблиця_кодів"]}
            codes = huffman.canonical_codes(table)
            for t in d["таблиця_кодів"]:
                self.assertEqual(format(codes[t["символ"]], "0%db" % t["довжина"]), t["код"], name)

    def test_our_replica_byte_identical(self):
        # з тими самими salt, nonce і записами наш кодер дає ту саму репліку
        for name in sorted(os.listdir(REF)):
            if not (name.startswith("ref_") and name.endswith(".json")):
                continue
            d = read_json(name)
            data = read(name[:-5] + ".bin")
            salt = bytes.fromhex(d["salt_hex"])
            nonce = bytes.fromhex(d["nonce_hex"])
            k_enc, k_mac = crypto.derive_keys(d["пароль"], salt)
            h = crypto.parse_header(data)
            recs = [pack_record(r) for r in d["записи"]]
            table = {t["символ"]: t["довжина"] for t in d["таблиця_кодів"]}
            payload = encode_payload(recs, table, no_compression=bool(d["flags"] & 8),
                                     class_table=d["class_table"])
            ours = crypto.build_replica(h["group_id"], h["batch_seq"], salt, nonce, payload, k_enc, k_mac)
            self.assertEqual(ours, data, name)

    def test_bad_replicas_codes(self):
        expected = read_json("expected_failures.json")
        for name in expected:
            data = read(name)
            with self.assertRaises(ReplicaError, msg=name) as e:
                h = crypto.parse_header(data)
                k_enc, k_mac = crypto.derive_keys("algo2026", h["salt"])
                _, payload = crypto.open_replica(data, k_enc, k_mac)
                decode_payload(payload)
            self.assertEqual(e.exception.code, expected[name]["очікуваний_код"], name)


if __name__ == "__main__":
    unittest.main()
