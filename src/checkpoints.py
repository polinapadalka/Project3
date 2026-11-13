# Зона 2, просунутий рівень. Оптимальні контрольні точки журналу (задача Б, розділ 3.3).
#
# Журнал з n записів. Snapshot - повний знімок стану, з якого можна почати
# відновлення. Якщо збій стався після запису k, треба взяти останній snapshot
# перед k і "програти" всі записи від нього до k.
#   - часті snapshot: дорого зберігати, але відновлення коротке;
#   - рідкі snapshot: дешево, але відновлення довге.
#
# F[j] - мінімальна сумарна вартість для перших j записів, якщо в j стоїть snapshot:
#   F[0] = 0
#   F[j] = min по i < j з (F[i] + Snapshot(i, j) + RecoveryCost(i, j))
# Snapshot(i, j) = snap_cost - ціна одного знімка;
# RecoveryCost(i, j) = fail_rate * сума по k від i+1 до j бітів, які треба програти
#   до запису k, тобто w[i+1] + ... + w[k] (збій однаково ймовірний після кожного запису).
# Складність O(n^2): для кожного j перебираємо всі i.


def prefix_sums(w):
    p = [0]
    for x in w:
        p.append(p[-1] + x)
    return p


def make_cost(w, fail_rate):
    p = prefix_sums(w)
    pp = prefix_sums(p)       # префіксні суми від префіксних сум

    def recovery(i, j):
        # сума по k = i+1..j з (p[k] - p[i]) рахується за O(1)
        return fail_rate * ((pp[j + 1] - pp[i + 1]) - (j - i) * p[i])
    return recovery


def optimal_checkpoints(w, snap_cost, fail_rate):
    n = len(w)
    recovery = make_cost(w, fail_rate)
    F = [0.0] + [float("inf")] * n
    prev = [0] * (n + 1)              # звідки прийшли - щоб відновити точки
    for j in range(1, n + 1):
        for i in range(j):
            c = F[i] + snap_cost + recovery(i, j)
            if c < F[j]:
                F[j] = c
                prev[j] = i
    points = []
    j = n
    while j > 0:
        points.append(j)
        j = prev[j]
    points.reverse()
    return F[n], points


def cost_of(points, w, snap_cost, fail_rate):
    # вартість довільного набору точок (останньою має бути n)
    recovery = make_cost(w, fail_rate)
    total = 0.0
    i = 0
    for j in points:
        total += snap_cost + recovery(i, j)
        i = j
    return total


def fixed_step(w, snap_cost, fail_rate, step):
    n = len(w)
    points = list(range(step, n, step)) + [n]
    return cost_of(points, w, snap_cost, fail_rate), points


def best_fixed_step(w, snap_cost, fail_rate):
    best = None
    for step in range(1, len(w) + 1):
        c, pts = fixed_step(w, snap_cost, fail_rate, step)
        if best is None or c < best[0]:
            best = (c, step, pts)
    return best
