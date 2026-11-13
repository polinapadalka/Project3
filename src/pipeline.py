# Зона 1 (інтеграція). Наскрізний тракт системи:
#
#  session -> класи (MLP) -> цінність v_i -> Гаффман вікна -> ваги w_i
#  -> ДП: скільки копій (0/1/2/3) -> планувальник: на які вузли
#  -> payload -> AES-256-CTR -> HMAC -> store() на вузли (replication.py)
#  -> репліки, що вижили -> recover() -> S, R, T, Q
#
# Дві оптимізації навмисно розділені: "скільки копій" вирішує ДП,
# "куди їх покласти" - планувальник.
import os
import time
from replication import Replication      # виданий файл, змінювати не можна
from src import huffman, knapsack, scheduler, crypto
from src.record import pack_record
from src.payload import encode_payload
from src.recovery import recover
from src.metrics import compute_q


def default_params(cfg):
    # параметри, які може змінювати ГА; за замовчуванням - з конфігу
    rep = cfg["реплікація"]
    return {
        "window": rep["replication_window_records"],
        "no_comp_threshold": 1.0,
        "f2": rep["second_copy_factor"],
        "f3": rep["third_copy_factor"],
        "rmax": 3,
        "method": "dp",
        "mix": 0.7,
        "pace": 1.0,
        "prediction": False,      # просунуте розширення зони 3
    }


def event_values(classes, recs, cfg):
    # v_i = priority(class_i) + critical_bonus * [critical_i = 1]
    # Клас (від MLP) і critical - різні речі: клас каже "що відбувається",
    # critical - "наскільки важливо зберегти". Обидва впливають на цінність.
    names = cfg["класифікатор"]["класи"]          # "0" -> "штатна подія" тощо
    prio = cfg["пріоритети_класів"]
    vals = []
    for c, rec in zip(classes, recs):
        v = prio[names[str(c)]]
        if rec["critical"] == 1:
            v += cfg["critical_bonus"]
        vals.append(v)
    return vals


def plan_window(win_bytes, vals, cfg, params, budget=None):
    # Одна таблиця Гаффмана на все вікно, ДО рюкзака і включно з подіями,
    # для яких потім вийде r = 0. Якби будувати її за обраними подіями,
    # ваги залежали б від вибору, а вибір - від ваг (циклічна залежність).
    lengths = huffman.code_lengths(b"".join(win_bytes))
    # w_i - скільки бітів займе одна стиснута копія запису i
    w = [sum(lengths[b] for b in rb) for rb in win_bytes]
    if params.get("prediction"):
        # з прогнозом стискаються залишки: вагу оцінюємо за залишками всього вікна
        from src import prediction
        res = prediction.transform(win_bytes)
        res_lengths = huffman.code_lengths(b"".join(res))
        w = [sum(res_lengths[b] for b in rb) for rb in res]
    table_bits = 16 * len(lengths)        # 2 байти на кожен символ таблиці
    # Якщо стиснення разом із таблицею не менше за сирі дані (з порогом від ГА),
    # пишемо без стиснення: кожен запис тоді коштує рівно 128 бітів.
    no_comp = sum(w) + table_bits >= params["no_comp_threshold"] * 128 * len(win_bytes)
    if no_comp:
        w = [128] * len(win_bytes)
    # бюджет задано на вікно з 48 записів; якщо ГА змінює вікно - масштабуємо,
    # інакше ГА "вигравав" би, просто дроблячи вікна
    if budget is None:
        budget = window_budget(cfg, params)
    if params["method"] == "dp":
        r, u = knapsack.dp_copies(w, vals, budget, params["f2"], params["f3"], params["rmax"])
    else:
        r, u = knapsack.greedy_copies(w, vals, budget, params["f2"], params["f3"], params["rmax"])
    return lengths, w, no_comp, r, u


def encode_batch(items, lengths, no_comp, prediction, limit):
    # items - пари (цінність, 16 байтів запису).
    # Будуємо всі допустимі варіанти payload і беремо найменший:
    # Гаффман з таблицею вікна, сирі байти, (за бажанням) залишки прогнозу.
    # Якщо навіть найменший не влазить у limit (буває лише з прогнозом, бо
    # планувальник оцінює розмір наближено), прибираємо найменш цінний запис.
    items = list(items)
    while True:
        records = sorted(rb for _, rb in items)        # у batch - порядок за часом
        options = [encode_payload(records, {}, True)]
        if not no_comp:
            options.append(encode_payload(records, lengths))
            if prediction:
                options.append(encode_payload(records, lengths, prediction=True))
        best = min(options, key=len)
        if len(best) + scheduler.OVERHEAD <= limit or len(items) <= 1:
            return best
        items.remove(min(items))


def window_budget(cfg, params):
    # бюджет задано на вікно з 48 записів; якщо ГА змінює вікно - масштабуємо,
    # інакше ГА "вигравав" би, просто дроблячи вікна
    rep = cfg["реплікація"]
    return round(rep["replication_budget_bits"] * params["window"] / rep["replication_window_records"])


def plan_with_pacing(wb, wv, cfg, params, nodes, free, share):
    # Глобальний бюджет один на весь сеанс. Якщо кожне вікно бере повні
    # 5000 бітів, бюджет закінчується посередині сесії, і останні вікна
    # не записуються зовсім. Тому вікну дозволено витратити лише свою частку
    # share байтів: якщо план дорожчий - зменшуємо бюджет ДП і плануємо знову.
    rep = cfg["реплікація"]
    bits = window_budget(cfg, params)
    while True:
        lengths, w, no_comp, r, u = plan_window(wb, wv, cfg, params, bits)
        batches, placed, sizes = scheduler.place(r, w, wv, nodes, free, rep["max_batch_bytes"],
                                                 len(lengths), no_comp, params["mix"])
        if sum(sizes.values()) <= share or bits == 0:
            return lengths, no_comp, r, u, batches, placed, sizes
        bits = int(bits * 0.85)


def run_session(recs, cfg, scenario, seed, params=None, classes=None, model=None):
    if params is None:
        params = default_params(cfg)
    rep = cfg["реплікація"]
    t0 = time.perf_counter()          # T рахує час обчислень усієї системи
    if classes is None:
        if model:
            from src.classify import predict_classes   # зона 3 підключається пізніше
            classes = predict_classes(model, recs, cfg)
        else:
            classes = [0] * len(recs)                   # режим "без перцептрона"
    vals = event_values(classes, recs, cfg)
    all_bytes = [pack_record(rec) for rec in recs]

    sc = rep["сценарії"][scenario]
    repl = Replication(sc, seed)
    nodes = sc["вузли"]
    # free - скільки місця, на думку планувальника, лишилося на кожному вузлі
    free = {name: nodes[name]["capacity_bytes"] for name in nodes}
    password = cfg["пароль_сеансу"]
    group_id = int(cfg["команда"][1:])                # "T01" -> 1
    salt = os.urandom(16)
    # PBKDF2 (100000 ітерацій) - один раз на сеанс, а не на кожну репліку,
    # інакше система буде повільною і зіпсується компонента T
    k_enc, k_mac = crypto.derive_keys(password, salt)
    nonces = crypto.NonceSource()
    info = {"assigned_copies": 0, "placed_copies": 0, "utility": 0.0,
            "windows": 0, "no_comp_windows": 0}

    win = params["window"]
    n_windows = (len(recs) + win - 1) // win
    for start in range(0, len(recs), win):
        wb = all_bytes[start:start + win]
        wv = vals[start:start + win]
        # частка глобального бюджету: залишок порівну на вікна, що лишилися.
        # Недовитрачене одним вікном автоматично переходить на наступні.
        left = sc["global_replication_budget_bytes"] - repl.stats()["global_used_bytes"]
        if params["pace"] > 0:
            share = left / (n_windows - start // win) * params["pace"]
        else:
            share = float("inf")      # pace = 0: без розподілу бюджету (для ablation)
        lengths, no_comp, r, u, batches, placed, sizes = plan_with_pacing(
            wb, wv, cfg, params, nodes, free, share)
        # ДП не знає про місткість окремих вузлів, тому "призначено" може
        # бути більше, ніж "розміщено" - цю різницю показуємо у звіті
        info["assigned_copies"] += sum(r)
        info["placed_copies"] += sum(placed)
        info["utility"] += u
        info["windows"] += 1
        info["no_comp_windows"] += 1 if no_comp else 0
        # Порядок має значення: невдала спроба теж списує бюджет, тому
        # найцінніші batch записуємо першими - вони точно отримають ресурс
        order = sorted(batches, key=lambda name: -sum(wv[i] for i in batches[name]))
        for name in order:
            if not batches[name]:
                continue
            items = [(wv[i], wb[i]) for i in batches[name]]
            limit = min(rep["max_batch_bytes"], free[name])
            payload = encode_batch(items, lengths, no_comp, params.get("prediction"), limit)
            # порядок: стиснути -> зашифрувати -> HMAC
            replica = crypto.build_replica(group_id, start // win, salt,
                                           nonces.next(), payload, k_enc, k_mac)
            free[name] -= len(replica)   # місце на вузлі витрачається навіть при відмові
            repl.store(name, replica)

    # після сценарію відмов збираємо все, що вижило хоча б на одному вузлі
    restored, rejected = recover(repl.surviving_blobs(), password)
    t = time.perf_counter() - t0
    st = repl.stats()
    res = compute_q(all_bytes, restored, st["global_used_bytes"], t, cfg)
    res.update(info)
    res["store_stats"] = st
    res["rejected"] = rejected
    res["restored_records"] = len(restored)
    return res
