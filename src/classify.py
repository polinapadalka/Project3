# Зона 3. Протокол навчання класифікатора:
#   train.csv -> навчання;
#   valid.csv -> вибір найкращої з кількох ініціалізацій за macro-F1;
#   holdout.csv -> фінальна незалежна оцінка (у команди його немає, і це правильно).
import numpy as np
from src import mlp
from src.features import make_features


def fit_scaler(X):
    # нормалізація до середнього 0 і відхилення 1, щоб усі ознаки мали один масштаб
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    return mean, std


def train_and_select(train_recs, train_y, valid_recs, valid_y, cfg,
                     hidden=16, epochs=60):
    # кількість ініціалізацій - з протоколу навчання в team_config.json (3)
    n_init = cfg["класифікатор"]["протокол_навчання"]["ініціалізацій"]
    seeds = list(range(1, n_init + 1))
    Xtr = make_features(train_recs)
    Xva = make_features(valid_recs)
    mean, std = fit_scaler(Xtr)        # статистики тільки з train, інакше valid "підглядає"
    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std
    ytr = np.array(train_y)
    yva = np.array(valid_y)
    runs = []
    best = None
    for seed in seeds:
        # різні ініціалізації ваг можуть дати різну якість - тому щонайменше три
        p, hist = mlp.train(Xtr, ytr, hidden, seed, epochs=epochs)
        f1 = mlp.macro_f1(yva, mlp.predict(p, Xva))
        runs.append({"seed": seed, "valid_macro_f1": f1, "final_loss": hist[-1]})
        if best is None or f1 > best["f1"]:
            best = {"f1": f1, "params": p, "seed": seed}
    model = {"params": best["params"], "mean": mean, "std": std, "seed": best["seed"]}
    cm = mlp.confusion(yva, mlp.predict(model["params"], Xva))
    return model, runs, cm


def predict_classes(model, recs, cfg):
    X = (make_features(recs) - model["mean"]) / model["std"]
    return [int(c) for c in mlp.predict(model["params"], X)]


def save_model(model, path):
    p = model["params"]
    np.savez(path, W1=p["W1"], b1=p["b1"], W2=p["W2"], b2=p["b2"],
             mean=model["mean"], std=model["std"], seed=model["seed"])


def load_model(path):
    d = np.load(path)
    params = {"W1": d["W1"], "b1": d["b1"], "W2": d["W2"], "b2": d["b2"]}
    return {"params": params, "mean": d["mean"], "std": d["std"], "seed": int(d["seed"])}
