# Зона 3: MLP (forward, loss, backprop, навчання)
import unittest
import numpy as np
from src import mlp


class TestMLP(unittest.TestCase):
    def test_forward_shapes(self):
        p = mlp.init_params(4, 5, 3, 0)
        _, _, probs = mlp.forward(p, np.ones((7, 4)))
        self.assertEqual(probs.shape, (7, 3))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1))

    def test_gradient_check(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(6, 4))
        y = np.array([0, 1, 2, 0, 1, 2])
        cw = np.array([1.0, 2.0, 0.5])
        p = mlp.init_params(4, 5, 3, 1)
        z1, h, probs = mlp.forward(p, X)
        g = mlp.backward(p, X, y, cw, z1, h, probs)
        eps = 1e-6
        for key in ["W1", "W2", "b2"]:
            num = np.zeros_like(p[key])
            it = np.nditer(p[key], flags=["multi_index"])
            for _ in it:
                i = it.multi_index
                old = p[key][i]
                p[key][i] = old + eps
                lp = mlp.loss_fn(mlp.forward(p, X)[2], y, cw)
                p[key][i] = old - eps
                lm = mlp.loss_fn(mlp.forward(p, X)[2], y, cw)
                p[key][i] = old
                num[i] = (lp - lm) / (2 * eps)
            self.assertTrue(np.allclose(num, g[key], atol=1e-5), key)

    def test_learns_simple_task(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(300, 2))
        y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 1).astype(int)
        p, hist = mlp.train(X, y, 8, seed=0, epochs=40)
        self.assertLess(hist[-1], hist[0])
        self.assertGreater(mlp.macro_f1(y, mlp.predict(p, X)), 0.8)

    def test_macro_f1(self):
        self.assertEqual(mlp.macro_f1([0, 1, 2], [0, 1, 2]), 1.0)


if __name__ == "__main__":
    unittest.main()
