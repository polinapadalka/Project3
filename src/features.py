# Зона 3. Ознаки для перцептрона - мінімальний набір з team_config.json.
#
# Одного запису мало: аномалію (клас 1) ще видно зі стрибка metric_value
# того самого джерела і високої severity, а перехід до інциденту (клас 2) -
# це часова структура: події частішають, джерела і category змінюються,
# event_code ідуть серією сусідніх значень (130, 131, 132...).
# Тому більшість ознак рахується у ковзному вікні часу.
# Беремо ТІЛЬКИ минуле - заглядати в майбутні записи не можна.
import csv
import math
import numpy as np
from src.record import FIELDS

WINDOW_MS = 1000     # ковзне вікно: ~10 подій при середньому інтервалі ~95 мс


def load_csv(path):
    recs = []
    labels = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rec = {name: int(row[name]) for name in FIELDS}
            recs.append(rec)
            labels.append(int(row["label"]) if "label" in row else -1)
    return recs, labels


def slog(x):
    # знаковий логарифм: стрибок на 5000 не "розриває" масштаб інших ознак
    return math.copysign(math.log1p(abs(x)), x)


def make_features(recs, window_ms=WINDOW_MS):
    rows = []
    left = 0          # два вказівники: left - перша подія, що ще у вікні часу
    last_metric = {}  # останнє metric_value кожного джерела
    for j in range(len(recs)):
        cur = recs[j]
        while recs[left]["timestamp_ms"] < cur["timestamp_ms"] - window_ms:
            left += 1
        win = recs[left:j + 1]                 # вікно разом з поточною подією
        cat_changes = 0
        adjacent = 0
        for a in range(1, len(win)):
            if win[a]["category"] != win[a - 1]["category"]:
                cat_changes += 1
            if abs(win[a]["event_code"] - win[a - 1]["event_code"]) == 1:
                adjacent += 1                  # пара із сусідніми event_code
        pairs = max(1, len(win) - 1)
        sources = len(set(r["source_id"] for r in win))
        dt = cur["timestamp_ms"] - recs[j - 1]["timestamp_ms"] if j > 0 else 0
        # зміна metric відносно попередньої події ТОГО САМОГО джерела:
        # у різних джерел різний рівень показника, тому сусідня подія тут не годиться
        src = cur["source_id"]
        d_metric = cur["metric_value"] - last_metric[src] if src in last_metric else 0
        last_metric[src] = cur["metric_value"]
        rows.append([
            dt,                     # інтервал до попередньої події
            len(win),               # кількість подій у ковзному вікні часу
            cat_changes,            # кількість змін category у вікні
            sources,                # кількість різних source_id у вікні
            adjacent / pairs,       # частка пар із сусідніми event_code
            slog(d_metric),         # зміна metric_value того самого джерела
            abs(slog(d_metric)),    # розмір стрибка незалежно від напрямку
            cur["severity"],        # severity
        ])
    # поле critical свідомо НЕ ознака і НЕ мітка: воно про важливість зберігання
    return np.array(rows, dtype=float)
