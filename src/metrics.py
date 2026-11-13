# Зона 1. Метрика якості системи (умова проєкту, розділ 4)
#
#   S = B_ref / (B_ref + B_stored)   ощадливість: що більше копій, то менше S
#   R = частка critical-подій, які вдалося відновити
#   T = t_ref / (t_ref + t)          швидкодія: що довше працює, то менше T
#   Q = a*S + b*R + g*T
# Три складові тягнуть у різні боки, тому не можна оптимізувати лише R:
# "все по 3 копії" підніме R, але зіб'є S і з'їсть бюджет пізніх подій.


def compute_q(original, restored, global_used_bytes, t, cfg):
    crit = [rb for rb in original if rb[14] == 1]      # байт 14 запису - critical
    got = set(restored)
    saved = sum(1 for rb in crit if rb in got)
    R = saved / len(crit) if crit else 1.0
    b_ref = 16 * 8 * len(original)                     # сирий журнал у бітах
    # B_stored - усі спроби запису, що списали ресурс, включно з невдалими
    b_stored = 8 * global_used_bytes
    S = b_ref / (b_ref + b_stored) if b_ref + b_stored > 0 else 1.0
    q = cfg["метрика_Q"]
    T = q["t_ref_секунд"] / (q["t_ref_секунд"] + t)
    Q = q["alpha"] * S + q["beta"] * R + q["gamma"] * T
    if crit and saved == 0:
        Q = 0.0          # за умовою: жодної critical-події не відновлено -> Q = 0
    return {"S": S, "R": R, "T": T, "Q": Q, "critical_total": len(crit),
            "critical_saved": saved, "t_s": t}
