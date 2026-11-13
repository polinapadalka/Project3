# Зона 3. Навчання і вибір перцептрона, матриця помилок
from common import cfg, save_csv, save_json, session, session_labels
from src.features import load_csv
from src.classify import train_and_select, save_model, predict_classes
from src import mlp

tr, ytr = load_csv("data/train.csv")
va, yva = load_csv("data/valid.csv")
model, runs, cm = train_and_select(tr, ytr, va, yva, cfg)
save_model(model, "results/model.npz")

clf = cfg["класифікатор"]
# session.csv теж має мітки - це додаткова перевірка, але модель за нею НЕ обирається
session_f1 = mlp.macro_f1(session_labels, predict_classes(model, session, cfg))
save_json("classifier_metrics.json", {
    "runs": runs,
    "chosen_seed": model["seed"],
    "valid_macro_f1": max(r["valid_macro_f1"] for r in runs),
    "session_macro_f1": session_f1,
    "F1_min": clf["F1_min"],
    "baseline_majority": clf["базова_лінія_мажоритарний_клас"],
    "baseline_random": clf["базова_лінія_випадковий_вибір"],
})
rows = []
for c in range(3):
    rows.append({"true": c, "pred0": int(cm[c][0]), "pred1": int(cm[c][1]), "pred2": int(cm[c][2])})
save_csv("confusion_matrix.csv", rows)
for r in runs:
    print("seed %d  valid macro-F1 = %.4f" % (r["seed"], r["valid_macro_f1"]))
print("session macro-F1 = %.4f (F1_min = %.2f)" % (session_f1, clf["F1_min"]))
