# Зона 2, просунутий рівень (D1 #13): оптимальні контрольні точки проти фіксованого кроку.
from common import cfg, session, save_csv
from src.record import pack_record
from src import huffman
from src.checkpoints import optimal_checkpoints, best_fixed_step, fixed_step

W = cfg["реплікація"]["replication_window_records"]
all_b = [pack_record(r) for r in session]
# вага запису - скільки бітів він займає після стиснення таблицею свого вікна
w = []
for start in range(0, len(all_b), W):
    wb = all_b[start:start + W]
    lengths = huffman.code_lengths(b"".join(wb))
    w += [sum(lengths[b] for b in rb) for rb in wb]
snap_cost = 8 * cfg["реплікація"]["max_batch_bytes"]     # знімок стану ~ один повний batch

rows = []
for sc, data in cfg["реплікація"]["сценарії"].items():
    nodes = data["вузли"].values()
    fail_rate = sum(n["failure_probability"] for n in nodes) / len(nodes)
    if fail_rate == 0:
        continue
    cost_dp, points = optimal_checkpoints(w, snap_cost, fail_rate)
    cost_best, step, _ = best_fixed_step(w, snap_cost, fail_rate)
    cost_48, _ = fixed_step(w, snap_cost, fail_rate, W)
    rows.append({"scenario": sc, "fail_rate": round(fail_rate, 4),
                 "dp_cost": round(cost_dp), "dp_checkpoints": len(points),
                 "best_fixed_step": step, "best_fixed_cost": round(cost_best),
                 "fixed_48_cost": round(cost_48),
                 "gain_vs_best_fixed_%": round(100 * (cost_best - cost_dp) / cost_best, 2),
                 "dp_not_worse": cost_dp <= cost_best + 1e-6})
    print("%-12s ДП %d (%d точок), найкращий крок %d -> %d, крок 48 -> %d" % (
        sc, cost_dp, len(points), step, cost_best, cost_48))
save_csv("checkpoints.csv", rows)
