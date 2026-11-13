# Зона 4: ГА і модуль атаки
import unittest
from src import crypto, ga, attack
from src.errors import ReplicaError
from src.record import pack_record
from src.payload import encode_payload
from tests.helpers import rec, seal


class TestGA(unittest.TestCase):
    def test_elitism_not_worse(self):
        genes = [(0, 100, "int"), (0.0, 1.0, "float")]
        fit = lambda x: -(x[0] - 37) ** 2 - (x[1] - 0.3) ** 2
        _, best, hist = ga.run_ga(fit, genes, pop_size=10, generations=10, seed=1)
        for k in range(1, len(hist)):
            self.assertGreaterEqual(hist[k]["best"], hist[k - 1]["best"])
        self.assertGreaterEqual(best, hist[0]["best"])

    def test_adaptive_and_local_not_worse(self):
        # просунутий ГА: адаптивна мутація + локальний пошук теж не погіршує найкращий
        genes = [(0, 100, "int"), (0.0, 1.0, "float")]
        fit = lambda x: -(x[0] - 37) ** 2 - (x[1] - 0.3) ** 2
        _, best, hist = ga.run_ga(fit, genes, pop_size=10, seed=1, adaptive=True, local=3, max_evals=200)
        for k in range(1, len(hist)):
            self.assertGreaterEqual(hist[k]["best"], hist[k - 1]["best"])
        self.assertLessEqual(hist[-1]["evals"], 200 + 10)
        self.assertTrue(all(0.05 <= h["p_mut"] <= 0.6 for h in hist))

    def test_local_search_improves(self):
        import random
        fit = lambda x: -abs(x[0] - 50)
        ind, f = ga.local_search([10], fit([10]), fit, [(0, 100, "int")], random.Random(0), 50)
        self.assertGreater(f, fit([10]))

    def test_same_eval_budget(self):
        genes = [(0, 10, "int")]
        _, _, h = ga.run_ga(lambda x: x[0], genes, pop_size=5, generations=3, seed=0)
        self.assertEqual(h[-1]["evals"], 5 + 3 * 3)

    def test_attack_recovers_key(self):
        stats = attack.char_stats(attack.make_training_text(500, 99))   # навчальний набір
        text = attack.make_training_text(200, 1)
        key = bytes([17, 200, 3, 99, 250, 64])
        ct = attack.weak_encrypt(text, key)
        genes = [(0, 255, "int")] * attack.KEY_LEN
        fit = lambda k: attack.key_fitness(k, ct, stats)
        best, f, hist = ga.run_ga(fit, genes, pop_size=40, generations=150, seed=1, p_mut=0.3)
        self.assertEqual(bytes(best), key)
        self.assertEqual(attack.weak_decrypt(ct, bytes(best)), text)
        _, f_rs, _ = ga.random_search(fit, genes, hist[-1]["evals"], seed=1)
        self.assertGreater(f, f_rs)

    def test_tampered_replica_rejected(self):
        # атака на цілісність: змінений байт має відхилятися за HMAC
        p = encode_payload([pack_record(rec())], {}, no_compression=True)
        replica, keys = seal(p)
        for pos in [45, len(replica) // 2, len(replica) - 1]:
            with self.assertRaises(ReplicaError) as e:
                crypto.open_replica(attack.tamper(replica, pos), *keys)
            self.assertEqual(e.exception.code, "E_HMAC")


if __name__ == "__main__":
    unittest.main()
