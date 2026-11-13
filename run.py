# Повний прогін однією командою:
#   python run.py                      - усі три сценарії
#   python run.py аварійний 1 2 3      - один сценарій на кількох seed
import json
import os
import sys
from src.features import load_csv
from src.classify import train_and_select, save_model, load_model
from src.pipeline import run_session

with open("team_config.json", encoding="utf-8") as f:
    cfg = json.load(f)
model_path = "results/model.npz"
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    os.makedirs("results", exist_ok=True)
    tr, ytr = load_csv("data/train.csv")
    va, yva = load_csv("data/valid.csv")
    model, _, _ = train_and_select(tr, ytr, va, yva, cfg)
    save_model(model, model_path)

session, _ = load_csv("data/session.csv")
scenarios = [sys.argv[1]] if len(sys.argv) > 1 else list(cfg["реплікація"]["сценарії"])
seed0 = cfg["реплікація"]["seed_моделі_реплікації"]
seeds = [int(s) for s in sys.argv[2:]] or [seed0 + i for i in range(5)]
limits = cfg["пороги_втрати_критичних_подій"]

for sc in scenarios:
    qs = []
    for seed in seeds:
        res = run_session(session, cfg, sc, seed, model=model)
        qs.append(res["Q"])
        st = res["store_stats"]
        print("%-12s seed=%-7d S=%.3f R=%.3f T=%.3f Q=%.3f  crit %d/%d  used %d/%d байт" % (
            sc, seed, res["S"], res["R"], res["T"], res["Q"], res["critical_saved"],
            res["critical_total"], st["global_used_bytes"], st["global_budget_bytes"]))
    print("%-12s mean Q = %.4f  (поріг втрати critical: %.2f)" % (sc, sum(qs) / len(qs), limits[sc]))
