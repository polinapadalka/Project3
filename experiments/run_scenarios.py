# Зона 2. Таблиця 5: поведінка в трьох сценаріях
from common import cfg, session, SCEN, SEEDS, save_csv, session_classes, ga_params
from src.pipeline import default_params, run_session

p = ga_params(default_params(cfg))
classes = session_classes()
limits = cfg["пороги_втрати_критичних_подій"]
rows = []
for sc in SCEN:
    for seed in SEEDS:
        res = run_session(session, cfg, sc, seed, params=p, classes=classes)
        st = res["store_stats"]
        nodes = st["вузли"].values()
        rows.append({"scenario": sc, "seed": seed, "S": round(res["S"], 4), "R": round(res["R"], 4),
                     "T": round(res["T"], 4), "Q": round(res["Q"], 4),
                     "lost_critical": round(1 - res["R"], 4), "threshold": limits[sc],
                     "assigned": res["assigned_copies"], "placed": res["placed_copies"],
                     "stored": sum(n["stored"] for n in nodes), "failed": sum(n["failed"] for n in nodes),
                     "rejected": st["global_rejected"],
                     "used_bytes": st["global_used_bytes"], "budget_bytes": st["global_budget_bytes"]})
save_csv("scenarios.csv", rows)
