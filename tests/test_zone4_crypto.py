# Зона 4: криптографія (C1: 10, 11)
import unittest
from src import crypto
from src.errors import ReplicaError
from src.record import pack_record
from src.payload import encode_payload, decode_payload
from tests.helpers import rec, seal, raw_payload, flip_byte


class TestCrypto(unittest.TestCase):
    def test_aes_known_vector(self):
        # NIST SP 800-38A F.5.5 (CTR-AES256), перший блок
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        key = bytes.fromhex("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4")
        ctr = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff")
        pt = bytes.fromhex("6bc1bee22e409f96e93d7e117393172a")
        ks = Cipher(algorithms.AES(key), modes.ECB()).encryptor().update(ctr)
        ct = bytes(a ^ b for a, b in zip(pt, ks))
        self.assertEqual(ct.hex(), "601ec313775789a5b7a7f504bbf3d228")

    def test_ctr_counter_starts_at_1(self):
        key = b"k" * 32
        nonce = b"n" * 12
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        ks = Cipher(algorithms.AES(key), modes.ECB()).encryptor().update(nonce + b"\x00\x00\x00\x01")
        self.assertEqual(crypto.aes_ctr(key, nonce, bytes(16)), ks)

    def test_replica_roundtrip(self):
        p = encode_payload([pack_record(rec())], {}, no_compression=True)
        replica, keys = seal(p)
        _, out = crypto.open_replica(replica, *keys)
        self.assertEqual(out, p)
        self.assertEqual(len(replica), 43 + len(p) + 32)

    def check_code(self, data, keys, code):
        with self.assertRaises(ReplicaError) as e:
            crypto.open_replica(data, *keys)
        self.assertEqual(e.exception.code, code)

    def test_negative_cases(self):
        p = encode_payload([pack_record(rec())], {}, no_compression=True)
        replica, keys = seal(p)
        self.check_code(replica[:50], keys, "E_SHORT")                    # обрізана
        self.check_code(b"LOG4" + replica[4:], keys, "E_MAGIC")           # поганий magic
        self.check_code(replica[:4] + b"\x02" + replica[5:], keys, "E_MAGIC")  # версія
        self.check_code(replica + b"\x00", keys, "E_LENGTH")
        self.check_code(flip_byte(replica, 50), keys, "E_HMAC")      # змінений шифротекст
        self.check_code(flip_byte(replica, len(replica) - 1), keys, "E_HMAC")  # змінений tag
        wrong = crypto.derive_keys("wrong", b"s" * 16)
        self.check_code(replica, wrong, "E_HMAC")                         # невірний пароль

    def test_reserved_inside_valid_hmac(self):
        bad = raw_payload(1, 8, 7, [], 16, pack_record(rec()))
        replica, keys = seal(bad)
        _, p = crypto.open_replica(replica, *keys)
        with self.assertRaises(ReplicaError):
            decode_payload(p)

    def test_nonce_unique(self):
        src = crypto.NonceSource()
        ns = [src.next() for _ in range(1000)]
        self.assertEqual(len(set(ns)), 1000)
        self.assertTrue(all(len(n) == 12 for n in ns))


if __name__ == "__main__":
    unittest.main()
