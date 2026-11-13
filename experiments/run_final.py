# Зона 1. Фінальний Q повної системи (5 seed на кожен сценарій) + перевірка порогів
from common import cfg, SCEN, SEEDS, session, session_classes, ga_params, save_json
from src.pipeline import default_params, run_session

p = ga_params(default_params(cfg))
classes = session_classes()
limits = cfg["пороги_втрати_критичних_подій"]
final = {}
for sc in SCEN:
    acc = {"S": 0.0, "R": 0.0, "T": 0.0, "Q": 0.0}
    worst = 0.0
    for seed in SEEDS:
        res = run_session(session, cfg, sc, seed, params=p, classes=classes)
        for k in acc:
            acc[k] += res[k] / len(SEEDS)
        worst = max(worst, 1 - res["R"])
    acc["max_lost_critical"] = worst
    acc["threshold"] = limits[sc]
    acc["passed"] = worst <= limits[sc]
    final[sc] = acc
    print("%-12s Q=%.4f R=%.3f  max loss %.3f <= %.2f  %s" % (
        sc, acc["Q"], acc["R"], worst, limits[sc], "OK" if acc["passed"] else "FAIL"))
final["mean_Q"] = sum(final[sc]["Q"] for sc in SCEN) / len(SCEN)
final["params"] = p
save_json("final_Q.json", final)
print("mean Q = %.4f" % final["mean_Q"])
