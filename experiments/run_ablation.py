# Зона 1. Таблиця 3: вплив блоків системи
from common import cfg, session, save_csv, session_classes, mean_run, ga_params
from src.pipeline import default_params

p = default_params(cfg)
full = ga_params(p)
classes = session_classes()
no_mlp = [0] * len(session)        # усі події "штатні": цінність лише від critical
variants = [("повна система", full, classes),
            ("без перцептрона", full, no_mlp),
            ("без ДП (greedy)", dict(full, method="greedy"), classes),
            ("без розподілу бюджету", dict(full, pace=0), classes),
            ("без ГА (ручні параметри)", p, classes)]
rows = []
for name, pp, cl in variants:
    for sc in ["деградований", "аварійний"]:
        m = mean_run(sc, pp, cl)
        rows.append({"variant": name, "scenario": sc, "S": round(m["S"], 4), "R": round(m["R"], 4),
                     "T": round(m["T"], 4), "Q": round(m["Q"], 4)})
        print("%-26s %-12s R=%.3f Q=%.4f" % (name, sc, m["R"], m["Q"]))
save_csv("ablation.csv", rows)
