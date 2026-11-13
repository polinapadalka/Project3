# Зона 4. Таблиця 6: ГА проти random search з однаковою кількістю обчислень fitness.
#  - базовий ГА: 4 гени (вікно, вага надійності вузла, коефіцієнт розподілу бюджету, фактор 2-ї копії);
#  - просунутий ГА: + максимальна кількість копій і розмір прихованого шару перцептрона,
#    адаптивна мутація і локальний пошук.
# Кращий результат з двох ГА стає параметрами фінальної системи.
from common import cfg, save_csv, save_json, session, session_labels, mean_run, SEED0
from src.pipeline import default_params
from src.features import load_csv
from src.classify import train_and_select, predict_classes, save_model, load_model
from src import ga, mlp

p = default_params(cfg)
tr, ytr = load_csv("data/train.csv")
va, yva = load_csv("data/valid.csv")
base_model = load_model("results/model.npz")
HIDDEN0 = base_model["params"]["W1"].shape[1]
cache = {HIDDEN0: (base_model, predict_classes(base_model, session, cfg))}


def model_for(hidden):
    # той самий протокол: 3 ініціалізації, вибір за macro-F1 на valid; кешуємо
    if hidden not in cache:
        model, _, _ = train_and_select(tr, ytr, va, yva, cfg, hidden=hidden)
        cache[hidden] = (model, predict_classes(model, session, cfg))
    return cache[hidden]


# fitness - на seed, відмінних від тих, на яких потім рахуємо фінальний Q
FIT_SEEDS = [SEED0 + 100, SEED0 + 101]
EVALS = 44


def q_of(pp, classes):
    q = 0.0
    for sc in ["деградований", "аварійний"]:
        q += mean_run(sc, pp, classes, seeds=FIT_SEEDS)["Q"] / 2
    return q


base_genes = [(24, 96, "int"), (0.0, 1.0, "float"), (0.8, 1.2, "float"), (1.1, 1.8, "float")]
adv_genes = base_genes + [(1, 3, "int"), (4, 32, "int")]


def fit_base(x):
    return q_of(dict(p, window=int(x[0]), mix=x[1], pace=x[2], f2=x[3]), cache[HIDDEN0][1])


def fit_adv(x):
    return q_of(dict(p, window=int(x[0]), mix=x[1], pace=x[2], f2=x[3], rmax=int(x[4])),
                model_for(int(x[5]))[1])


manual_f = fit_base([p["window"], p["mix"], p["pace"], p["f2"]])
print("manual: %.4f" % manual_f)
b_best, b_f, b_hist = ga.run_ga(fit_base, base_genes, pop_size=8, seed=7, max_evals=EVALS)
rb_best, rb_f, rb_hist = ga.random_search(fit_base, base_genes, b_hist[-1]["evals"], seed=7)
print("base GA %.4f, random %.4f" % (b_f, rb_f))
a_best, a_f, a_hist = ga.run_ga(fit_adv, adv_genes, pop_size=8, seed=7, adaptive=True, local=3,
                                max_evals=EVALS)
ra_best, ra_f, ra_hist = ga.random_search(fit_adv, adv_genes, a_hist[-1]["evals"], seed=7)
print("advanced GA %.4f, random %.4f" % (a_f, ra_f))


def random_at(hist, evals):
    return max(r["best"] for r in hist if r["evals"] <= evals)


rows = []
for h in b_hist:
    rows.append({"variant": "базовий ГА", "generation": h["generation"], "evals": h["evals"],
                 "ga_best": round(h["best"], 4), "ga_mean": round(h["mean"], 4),
                 "random_best": round(random_at(rb_hist, h["evals"]), 4), "p_mut": round(h["p_mut"], 3)})
for h in a_hist:
    rows.append({"variant": "просунутий ГА", "generation": h["generation"], "evals": h["evals"],
                 "ga_best": round(h["best"], 4), "ga_mean": round(h["mean"], 4),
                 "random_best": round(random_at(ra_hist, h["evals"]), 4), "p_mut": round(h["p_mut"], 3)})
save_csv("ga_history.csv", rows)

# фінальні параметри - кращий з двох ГА (random search - лише точка порівняння)
if a_f >= b_f:
    final = a_best
else:
    final = list(b_best) + [p["rmax"], HIDDEN0]
hidden = int(final[5])
model = model_for(hidden)[0]       # перцептрон з обраною архітектурою
save_model(model, "results/model.npz")
X_session_f1 = mlp.macro_f1(session_labels, predict_classes(model, session, cfg))
save_json("ga_summary.json", {
    "genes": ["window", "mix", "pace", "f2", "rmax", "hidden"],
    "manual": {"params": [p["window"], p["mix"], p["pace"], p["f2"], p["rmax"], HIDDEN0], "fitness": manual_f},
    "base_ga": {"params": b_best, "fitness": b_f},
    "base_random": {"params": rb_best, "fitness": rb_f},
    "advanced_ga": {"params": a_best, "fitness": a_f},
    "ga": {"params": final, "fitness": max(a_f, b_f), "from": "advanced" if a_f >= b_f else "base"},
    "advanced_random": {"params": ra_best, "fitness": ra_f},
    "evals": EVALS,
    "final_hidden": hidden,
    "final_model_session_macro_f1": X_session_f1,
})
