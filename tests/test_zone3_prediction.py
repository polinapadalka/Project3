# Зона 3, просунутий рівень: прогноз і залишки (D1 #14)
import os
import random
import unittest
from src import prediction, huffman, crypto
from src.errors import ReplicaError
from src.features import load_csv
from src.payload import encode_payload, decode_payload
from src.record import pack_record
from tests.helpers import rec, seal

ROOT = os.path.join(os.path.dirname(__file__), "..")


class TestPrediction(unittest.TestCase):
    def test_inverse_restores_bytes(self):
        rng = random.Random(1)
        recs = [pack_record(rec(ts=1000 + 50 * i, src=rng.randint(1, 4), code=rng.randint(0, 65535),
                                metric=rng.randint(-2**31, 2**31 - 1), cat=rng.randint(0, 15)))
                for i in range(200)]
        self.assertEqual(prediction.inverse(prediction.transform(recs)), recs)

    def test_session_payload_roundtrip(self):
        recs, _ = load_csv(os.path.join(ROOT, "data", "session.csv"))
        all_b = [pack_record(r) for r in recs]
        for start in range(0, len(all_b), 48):
            wb = all_b[start:start + 48]
            p = encode_payload(wb, huffman.code_lengths(b"".join(wb)), prediction=True)
            self.assertEqual(decode_payload(p, extensions=True)[0], wb)

    def test_prediction_smaller_on_session(self):
        recs, _ = load_csv(os.path.join(ROOT, "data", "session.csv"))
        wb = [pack_record(r) for r in recs[:48]]
        lengths = huffman.code_lengths(b"".join(wb))
        self.assertLess(len(encode_payload(wb, lengths, prediction=True)), len(encode_payload(wb, lengths)))

    def test_base_decoder_rejects_extension(self):
        # для базового формату PREDICTION - невідоме розширення
        wb = [pack_record(rec(ts=1000 + i)) for i in range(5)]
        p = encode_payload(wb, huffman.code_lengths(b"".join(wb)), prediction=True)
        with self.assertRaises(ReplicaError) as e:
            decode_payload(p)
        self.assertEqual(e.exception.code, "E_PAYLOAD")


if __name__ == "__main__":
    unittest.main()
