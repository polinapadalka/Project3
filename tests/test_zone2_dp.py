# Зона 2: ДП і greedy (C1: 6)
import random
import unittest
from src import knapsack

F2, F3 = 1.55, 1.85


class TestKnapsack(unittest.TestCase):
    def test_dp_equals_brute_force(self):
        rng = random.Random(1)
        for _ in range(30):
            n = 6
            w = [rng.randint(5, 40) for _ in range(n)]
            v = [rng.uniform(1, 10) for _ in range(n)]
            W = rng.randint(0, 200)
            r_dp, u_dp = knapsack.dp_copies(w, v, W, F2, F3)
            _, u_bf = knapsack.brute_force(w, v, W, F2, F3)
            self.assertAlmostEqual(u_dp, u_bf, places=6)
            # відновлений розв'язок допустимий і дає ту саму utility
            self.assertLessEqual(sum(r_dp[i] * w[i] for i in range(n)), W)
            self.assertAlmostEqual(sum(knapsack.utility(v[i], r_dp[i], F2, F3) for i in range(n)), u_dp)

    def test_dp_not_worse_than_greedy(self):
        rng = random.Random(2)
        for _ in range(50):
            n = 20
            w = [rng.randint(30, 140) for _ in range(n)]
            v = [rng.choice([1, 3, 6]) + rng.choice([0, 0, 4]) for _ in range(n)]
            W = rng.randint(100, 3000)
            _, u_dp = knapsack.dp_copies(w, v, W, F2, F3)
            _, u_gr = knapsack.greedy_copies(w, v, W, F2, F3)
            self.assertGreaterEqual(u_dp + 1e-9, u_gr)

    def test_greedy_counterexample(self):
        # greedy бере A (краще du/cost), а B+C разом вигідніші
        w, v, W = [6, 5, 5], [13, 10, 10], 10
        _, u_gr = knapsack.greedy_copies(w, v, W, F2, F3)
        _, u_dp = knapsack.dp_copies(w, v, W, F2, F3)
        self.assertEqual(u_gr, 13)
        self.assertEqual(u_dp, 20)

    def test_boundaries(self):
        w, v = [10, 20], [5, 5]
        self.assertEqual(knapsack.dp_copies(w, v, 0, F2, F3)[0], [0, 0])     # W = 0
        self.assertEqual(knapsack.dp_copies(w, v, 9, F2, F3)[0], [0, 0])     # min_cost - 1
        self.assertEqual(knapsack.dp_copies(w, v, 10, F2, F3)[0], [1, 0])    # min_cost
        self.assertEqual(knapsack.dp_copies(w, v, 10000, F2, F3)[0], [3, 3]) # дуже великий

    def test_budget_exactly_equals_cost(self):
        r, _ = knapsack.dp_copies([16, 16], [1, 1], 32, F2, F3)
        self.assertEqual(sum(r), 2)
        r, _ = knapsack.dp_copies([16, 16], [1, 1], 31, F2, F3)
        self.assertEqual(sum(r), 1)

    def test_diminishing_returns(self):
        u = [knapsack.utility(10, r, F2, F3) for r in range(4)]
        self.assertTrue(u[0] < u[1] < u[2] < u[3])
        self.assertGreater(u[2] - u[1], u[3] - u[2])


if __name__ == "__main__":
    unittest.main()
