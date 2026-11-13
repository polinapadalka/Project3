# Зона 2. Багатоваріантний рюкзак (FORMAT_SPEC_P3 §3.5)
#
# Для кожної події обираємо ОДНЕ значення r_i з {0,1,2,3} - це взаємовиключні
# варіанти, а не окремі предмети. Звичайний рюкзак 0/1 тут не підходить.
#
#   maximize   sum u_i(r_i)
#   subject to sum r_i * w_i <= W
#
# u_i(0)=0, u_i(1)=v, u_i(2)=v*f2, u_i(3)=v*f3 - фактори КУМУЛЯТИВНІ:
# 1.55 - це корисність двох копій разом, а не приріст другої.
# Приріст: друга копія +0.55v, третя +0.30v - зменшувана віддача.
# Тому алгоритм не витратить увесь бюджет на кілька найважливіших подій.
import itertools
import numpy as np


def utility(v, r, f2, f3):
    return [0.0, v, v * f2, v * f3][r]


def dp_copies(w, v, budget, f2, f3, rmax=3):
    # DP[i][c] - найбільша корисність для перших i подій, якщо витрачено <= c бітів.
    # Перехід: DP[i][c] = max по r з (DP[i-1][c - r*w_i] + u_i(r)), якщо r*w_i <= c.
    # Зберігаємо тільки поточний рядок (масив best), а вибір r - у таблиці choice,
    # щоб потім відновити сам розв'язок.
    # Складність O(n * W * Rmax): n = 48 подій, W = 5000 бітів, Rmax = 3.
    # Рядок по c рахується векторно numpy - алгоритм той самий, просто швидше.
    n = len(w)
    best = np.zeros(budget + 1)                       # DP[0][c] = 0 для всіх c
    choice = np.zeros((n, budget + 1), dtype=np.int8)
    for i in range(n):
        new = best.copy()                             # варіант r = 0
        for r in range(1, rmax + 1):
            cost = r * w[i]
            if cost > budget:
                break
            cand = np.full(budget + 1, -1.0)          # -1: варіант недопустимий
            cand[cost:] = best[:budget + 1 - cost] + utility(v[i], r, f2, f3)
            better = cand > new
            new[better] = cand[better]
            choice[i][better] = r
        best = new
    # відновлення: йдемо з кінця, на кожному кроці віднімаємо вартість обраного r
    r_list = [0] * n
    c = budget
    for i in range(n - 1, -1, -1):
        r_list[i] = int(choice[i][c])
        c -= r_list[i] * w[i]
    return r_list, float(best[budget])


def greedy_copies(w, v, budget, f2, f3, rmax=3):
    # Жадібний baseline: на кожному кроці додаємо ОДНУ наступну копію
    # (0->1, 1->2 або 2->3) тієї події, де du / cost найбільше.
    # Це локально найкращий вибір, але не гарантує оптимуму:
    # контрприклад w=[6,5,5], v=[13,10,10], W=10 -> greedy 13, ДП 20.
    n = len(w)
    r_list = [0] * n
    used = 0
    while True:
        best_i = -1
        best_ratio = -1.0
        for i in range(n):
            if r_list[i] < rmax and used + w[i] <= budget:
                du = utility(v[i], r_list[i] + 1, f2, f3) - utility(v[i], r_list[i], f2, f3)
                if du / w[i] > best_ratio:
                    best_ratio = du / w[i]
                    best_i = i
        if best_i < 0:          # жодна наступна копія вже не влазить
            break
        r_list[best_i] += 1
        used += w[best_i]
    total = sum(utility(v[i], r_list[i], f2, f3) for i in range(n))
    return r_list, total


def brute_force(w, v, budget, f2, f3, rmax=3):
    # Повний перебір 4^n варіантів - лише для перевірки ДП на малих n (n=6 -> 4096)
    best_u = 0.0
    best_r = [0] * len(w)
    for combo in itertools.product(range(rmax + 1), repeat=len(w)):
        cost = sum(combo[i] * w[i] for i in range(len(w)))
        if cost <= budget:
            u = sum(utility(v[i], combo[i], f2, f3) for i in range(len(w)))
            if u > best_u + 1e-9:
                best_u = u
                best_r = list(combo)
    return best_r, best_u
