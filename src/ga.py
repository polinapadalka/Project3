# Зона 4. Генетичний алгоритм і random search для порівняння.
#
# Особина - список генів. Ген задається (min, max, "int"/"float").
# Цикл: популяція -> fitness -> турнірна селекція -> кросовер -> мутація,
# а найкращі (еліта) переходять у нове покоління без змін.
# Через елітизм найкращий fitness ніколи не погіршується між поколіннями.
import random


def random_gene(g, rng):
    lo, hi, kind = g
    return rng.randint(lo, hi) if kind == "int" else rng.uniform(lo, hi)


def mutate(ind, genes, rng, p_mut):
    # кожен ген мутує з імовірністю p_mut
    child = list(ind)
    for k in range(len(genes)):
        if rng.random() < p_mut:
            lo, hi, kind = genes[k]
            if kind == "int":
                if rng.random() < 0.3:
                    child[k] = rng.randint(lo, hi)        # великий стрибок у будь-яку точку
                else:
                    # маленький крок поруч - для тонкого доналаштування
                    step = max(1, round(rng.gauss(0, (hi - lo) * 0.1)))
                    child[k] = min(hi, max(lo, child[k] + step * rng.choice([-1, 1])))
            else:
                step = (hi - lo) * 0.1
                child[k] = min(hi, max(lo, child[k] + rng.gauss(0, step)))
    return child


def crossover(a, b, rng):
    # рівномірний кросовер: кожен ген беремо від випадкового з двох батьків
    return [a[k] if rng.random() < 0.5 else b[k] for k in range(len(a))]


def tournament(pop, fit, rng, size=3):
    # беремо size випадкових особин, перемагає найкраща з них.
    # Сильні частіше стають батьками, але слабкі теж мають шанс - різноманіття не зникає
    best = None
    for _ in range(size):
        i = rng.randrange(len(pop))
        if best is None or fit[i] > fit[best]:
            best = i
    return pop[best]


def local_search(ind, f, fitness, genes, rng, tries):
    # Просунутий рівень: локальний пошук (hill climbing) навколо найкращої особини.
    # Пробуємо невеликі зміни одного гена; кращий варіант одразу приймаємо.
    for _ in range(tries):
        k = rng.randrange(len(genes))
        lo, hi, kind = genes[k]
        cand = list(ind)
        if kind == "int":
            cand[k] = min(hi, max(lo, cand[k] + rng.choice([-1, 1]) * max(1, (hi - lo) // 20)))
        else:
            cand[k] = min(hi, max(lo, cand[k] + rng.gauss(0, (hi - lo) * 0.05)))
        cf = fitness(cand)
        if cf > f:
            ind, f = cand, cf
    return ind, f


def run_ga(fitness, genes, pop_size=10, generations=8, seed=0, p_mut=0.2, elite=2,
           adaptive=False, local=0, max_evals=None):
    # adaptive=True - адаптивна мутація: якщо найкращий не покращився,
    #   мутацію посилюємо (виходимо з локального оптимуму), інакше послаблюємо;
    # local > 0 - після кожного покоління стільки кроків локального пошуку для лідера;
    # max_evals - зупинка за кількістю обчислень fitness (для чесного порівняння).
    rng = random.Random(seed)
    pop = [[random_gene(g, rng) for g in genes] for _ in range(pop_size)]
    fit = [fitness(ind) for ind in pop]
    evals = pop_size          # рахуємо обчислення fitness - для чесного порівняння з random
    history = [{"generation": 0, "evals": evals, "best": max(fit), "mean": sum(fit) / len(fit),
                "p_mut": p_mut}]
    gen = 0
    while True:
        gen += 1
        if max_evals is None and gen > generations:
            break
        if max_evals is not None and evals >= max_evals:
            break
        best_before = max(fit)
        order = sorted(range(pop_size), key=lambda i: -fit[i])
        new_pop = [pop[i] for i in order[:elite]]          # елітизм
        new_fit = [fit[i] for i in order[:elite]]          # еліту не перераховуємо
        while len(new_pop) < pop_size:
            child = crossover(tournament(pop, fit, rng), tournament(pop, fit, rng), rng)
            child = mutate(child, genes, rng, p_mut)
            new_pop.append(child)
            new_fit.append(fitness(child))
            evals += 1
        if local > 0:
            i = max(range(pop_size), key=lambda i: new_fit[i])
            new_pop[i], new_fit[i] = local_search(new_pop[i], new_fit[i], fitness, genes, rng, local)
            evals += local
        pop, fit = new_pop, new_fit
        if adaptive:
            if max(fit) > best_before + 1e-12:
                p_mut = max(0.05, p_mut * 0.8)
            else:
                p_mut = min(0.6, p_mut * 1.5)
        history.append({"generation": gen, "evals": evals, "best": max(fit),
                        "mean": sum(fit) / len(fit), "p_mut": p_mut})
    i = max(range(pop_size), key=lambda i: fit[i])
    return pop[i], fit[i], history


def random_search(fitness, genes, n_evals, seed=0):
    # те саме число обчислень fitness, але кожна точка просто випадкова
    rng = random.Random(seed)
    best, best_f = None, None
    history = []
    for e in range(1, n_evals + 1):
        ind = [random_gene(g, rng) for g in genes]
        f = fitness(ind)
        if best_f is None or f > best_f:
            best, best_f = ind, f
        history.append({"evals": e, "best": best_f})
    return best, best_f, history
