# Зона 1. Спільне для всіх експериментів
import csv
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import load_csv
from src.pipeline import run_session

with open("team_config.json", encoding="utf-8") as f:
    cfg = json.load(f)
SCEN = list(cfg["реплікація"]["сценарії"])
# офіційний Q рахується на прихованих seed, тому усереднюємо по кількох:
# відкритий seed з конфігу і чотири наступні
SEED0 = cfg["реплікація"]["seed_моделі_реплікації"]
SEEDS = [SEED0 + i for i in range(5)]
os.makedirs("results", exist_ok=True)
session, session_labels = load_csv("data/session.csv")


def save_csv(name, rows):
    with open("results/" + name, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("saved results/" + name)


def save_json(name, data):
    with open("results/" + name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("saved results/" + name)


def session_classes():
    # класи сесії від навченої моделі (run_classifier.py має бути запущений раніше)
    from src.classify import load_model, predict_classes
    model = load_model("results/model.npz")
    return predict_classes(model, session, cfg)


def mean_run(sc, params, classes, seeds=SEEDS):
    acc = {"S": 0.0, "R": 0.0, "T": 0.0, "Q": 0.0}
    for seed in seeds:
        res = run_session(session, cfg, sc, seed, params=params, classes=classes)
        for k in acc:
            acc[k] += res[k] / len(seeds)
    return acc


def ga_params(p):
    # параметри, знайдені ГА (run_ga.py); якщо ГА ще не запускали - ручні
    path = "results/ga_summary.json"
    if not os.path.exists(path):
        return p
    with open(path, encoding="utf-8") as f:
        x = json.load(f)["ga"]["params"]
    out = dict(p, window=int(x[0]), mix=x[1], pace=x[2], f2=x[3])
    if len(x) > 4:
        out["rmax"] = int(x[4])        # просунутий ГА обирає ще й максимум копій
    return out
