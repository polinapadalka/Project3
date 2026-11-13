# Зона 3. Власний багатошаровий перцептрон (бібліотек нейромереж не використовуємо).
#
# Архітектура: вхід (9 ознак) -> прихований шар з ReLU -> 3 виходи -> softmax.
# softmax перетворює виходи на ймовірності трьох класів, що в сумі дають 1.
# Навчання: cross-entropy + backpropagation + оптимізатор Adam, міні-батчі.
import numpy as np


def init_params(n_in, n_hidden, n_out, seed):
    # ініціалізація He: дисперсія 2/n_in добре підходить для ReLU
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden)),
        "b1": np.zeros(n_hidden),
        "W2": rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, n_out)),
        "b2": np.zeros(n_out),
    }


def forward(p, X):
    # прямий прохід
    z1 = X @ p["W1"] + p["b1"]
    h = np.maximum(z1, 0)                    # ReLU
    z2 = h @ p["W2"] + p["b2"]
    z2 = z2 - z2.max(axis=1, keepdims=True)  # віднімаємо максимум, щоб exp не переповнився
    e = np.exp(z2)
    probs = e / e.sum(axis=1, keepdims=True)  # softmax
    return z1, h, probs


def loss_fn(probs, y, cw):
    # cross-entropy: -log(ймовірності правильного класу).
    # cw - ваги класів: рідкісні класи важать більше, інакше модель
    # навчилася б завжди казати "штатна" і мала б поганий macro-F1
    n = len(y)
    return -np.sum(cw[y] * np.log(probs[np.arange(n), y] + 1e-12)) / n


def backward(p, X, y, cw, z1, h, probs):
    # Backpropagation - похідна втрат по кожній вазі через ланцюгове правило.
    # Для softmax + cross-entropy похідна по z2 дуже проста: probs - one_hot(y)
    n = len(y)
    d2 = probs.copy()
    d2[np.arange(n), y] -= 1
    d2 *= cw[y][:, None] / n
    grads = {"W2": h.T @ d2, "b2": d2.sum(axis=0)}
    dh = d2 @ p["W2"].T          # помилка, передана назад у прихований шар
    dz1 = dh * (z1 > 0)          # похідна ReLU: 1, якщо z1 > 0, інакше 0
    grads["W1"] = X.T @ dz1
    grads["b1"] = dz1.sum(axis=0)
    return grads


def train(X, y, n_hidden, seed, epochs=60, lr=0.01, batch=64, l2=1e-4):
    n_out = 3
    p = init_params(X.shape[1], n_hidden, n_out, seed)
    counts = np.bincount(y, minlength=n_out)
    cw = len(y) / (n_out * np.maximum(counts, 1))   # ваги, обернені до частоти класу
    # Adam: m - ковзне середнє градієнта, s - ковзне середнє його квадрата.
    # Кожна вага отримує свій розмір кроку.
    m = {k: np.zeros_like(v) for k, v in p.items()}
    s = {k: np.zeros_like(v) for k, v in p.items()}
    rng = np.random.default_rng(seed)
    step = 0
    history = []
    for ep in range(epochs):
        idx = rng.permutation(len(y))               # нова перестановка кожну епоху
        for start in range(0, len(y), batch):
            b = idx[start:start + batch]
            z1, h, probs = forward(p, X[b])
            g = backward(p, X[b], y[b], cw, z1, h, probs)
            step += 1
            for k in p:
                g[k] = g[k] + l2 * p[k]             # L2-регуляризація проти перенавчання
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                s[k] = 0.999 * s[k] + 0.001 * g[k] ** 2
                mh = m[k] / (1 - 0.9 ** step)       # поправка зсуву на перших кроках
                sh = s[k] / (1 - 0.999 ** step)
                p[k] -= lr * mh / (np.sqrt(sh) + 1e-8)
        history.append(loss_fn(forward(p, X)[2], y, cw))
    return p, history


def predict(p, X):
    return forward(p, X)[2].argmax(axis=1)          # клас з найбільшою ймовірністю


def confusion(y_true, y_pred, n=3):
    # рядок - правильний клас, стовпець - передбачений
    cm = np.zeros((n, n), dtype=int)
    for a, b in zip(y_true, y_pred):
        cm[a][b] += 1
    return cm


def macro_f1(y_true, y_pred, n=3):
    # F1 рахуємо для кожного класу окремо і беремо середнє.
    # На відміну від accuracy, рідкісний клас тут важить стільки ж, як частий.
    cm = confusion(y_true, y_pred, n)
    f1s = []
    for c in range(n):
        tp = cm[c][c]
        fp = cm[:, c].sum() - tp        # сказали "c", а було інше
        fn = cm[c].sum() - tp           # було "c", а сказали інше
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))
