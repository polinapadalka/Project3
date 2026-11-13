# Зона 2. Таблиця 1: ДП проти жадібного на однакових вікнах
import time
from common import cfg, session, save_csv, session_classes
from src.pipeline import default_params, event_values, plan_window, window_budget
from src.record import pack_record
from src import knapsack

p = default_params(cfg)
vals = event_values(session_classes(), session, cfg)
all_b = [pack_record(r) for r in session]
W = p["window"]
budget = window_budget(cfg, p)
rows = []
for start in range(0, len(session), W):
    wb, wv = all_b[start:start + W], vals[start:start + W]
    _, w, _, _, _ = plan_window(wb, wv, cfg, p)
    t0 = time.perf_counter()
    r_dp, u_dp = knapsack.dp_copies(w, wv, budget, p["f2"], p["f3"])
    t_dp = time.perf_counter() - t0
    t0 = time.perf_counter()
    r_gr, u_gr = knapsack.greedy_copies(w, wv, budget, p["f2"], p["f3"])
    t_gr = time.perf_counter() - t0
    rows.append({"window": start // W, "u_dp": round(u_dp, 3), "u_greedy": round(u_gr, 3),
                 "bits_dp": sum(r_dp[i] * w[i] for i in range(len(w))),
                 "bits_greedy": sum(r_gr[i] * w[i] for i in range(len(w))),
                 "time_dp_ms": round(t_dp * 1000, 3), "time_greedy_ms": round(t_gr * 1000, 3)})
save_csv("dp_vs_greedy.csv", rows)
better = sum(1 for r in rows if r["u_dp"] > r["u_greedy"] + 1e-9)
print("ДП краще за greedy у %d з %d вікон" % (better, len(rows)))
