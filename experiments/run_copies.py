# Зона 2. Таблиця 2: вплив максимальної кількості копій
from common import cfg, SCEN, save_csv, session_classes, mean_run
from src.pipeline import default_params

p = default_params(cfg)
classes = session_classes()
rows = []
for sc in SCEN:
    for rmax in [1, 2, 3]:
        m = mean_run(sc, dict(p, rmax=rmax), classes)
        rows.append({"scenario": sc, "max_copies": rmax, "S": round(m["S"], 4),
                     "R": round(m["R"], 4), "T": round(m["T"], 4), "Q": round(m["Q"], 4)})
save_csv("copy_count_experiment.csv", rows)
