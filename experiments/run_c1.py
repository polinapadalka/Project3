# Зона 1. Результати 12 базових кейсів C1 (розділ 9.1 умови) -> results/c1_results.json
import json
import os
import time
from common import cfg, session, SEEDS, save_json, session_classes
from src import crypto, huffman, knapsack, mlp
from src.errors import ReplicaError
from src.features import load_csv
from src.payload import encode_payload, decode_payload
from src.record import pack_record, parse_record
from src.pipeline import run_session, default_params, event_values, plan_window, window_budget
from src.classify import load_model, predict_classes

REF = "reference"
limits = cfg["пороги_втрати_критичних_подій"]
results = []


def case(num, text, ok, detail):
    results.append({"case": num, "text": text, "passed": bool(ok), "detail": detail})
    print("%2d. %-4s %s - %s" % (num, "OK" if ok else "FAIL", text, detail))


def roundtrip(recs_bytes, lengths, no_comp=False):
    salt, nonce = os.urandom(16), os.urandom(12)
    k_enc, k_mac = crypto.derive_keys(cfg["пароль_сеансу"], salt)
    rep = crypto.build_replica(1, 0, salt, nonce, encode_payload(recs_bytes, lengths, no_comp), k_enc, k_mac)
    _, p = crypto.open_replica(rep, k_enc, k_mac)
    return decode_payload(p)[0]


def read_ref(name):
    with open(os.path.join(REF, name), "rb") as f:
        return f.read()


def open_ref(data, password):
    h = crypto.parse_header(data)
    k_enc, k_mac = crypto.derive_keys(password, h["salt"])
    _, p = crypto.open_replica(data, k_enc, k_mac)
    return decode_payload(p)


def error_code(data, password):
    try:
        open_ref(data, password)
        return "OK"
    except ReplicaError as e:
        return e.code


# 1. один запис
one = pack_record(session[0])
case(1, "Один запис кодується і відновлюється побайтово",
     roundtrip([one], huffman.code_lengths(one)) == [one], "16 байтів збігаються")

# 2. порожній журнал
try:
    empty_ok = roundtrip([], {}) == [] and run_session([], cfg, "штатний", SEEDS[0])["restored_records"] == 0
except Exception as e:
    empty_ok = False
case(2, "Порожній журнал не спричиняє аварійного завершення", empty_ok, "repl + повний тракт з 0 записів")

# 3. NO_COMPRESSION
raw32 = [pack_record(r) for r in session[:32]]
case(3, "NO_COMPRESSION відновлює сирі записи точно", roundtrip(raw32, {}, True) == raw32, "32 записи")

# 4. канонічні коди тільки з довжин
codes_ok = True
for name in sorted(os.listdir(REF)):
    if name.startswith("ref_") and name.endswith(".json"):
        with open(os.path.join(REF, name), encoding="utf-8") as f:
            d = json.load(f)
        table = {t["символ"]: t["довжина"] for t in d["таблиця_кодів"]}
        codes = huffman.canonical_codes(table)
        for t in d["таблиця_кодів"]:
            if format(codes[t["символ"]], "0%db" % t["довжина"]) != t["код"]:
                codes_ok = False
case(4, "Канонічні коди відновлюються тільки з довжин", codes_ok, "коди збігаються з еталонними таблицями")

# 5. вироджений алфавіт
deg = huffman.code_lengths(bytes([1]) * 16)
deg_ok = deg == {1: 1} and huffman.canonical_codes(deg) == {1: 0}
with open(os.path.join(REF, "ref_03_degenerate.json"), encoding="utf-8") as f:
    d3 = json.load(f)
deg_ok = deg_ok and [parse_record(r) for r in open_ref(read_ref("ref_03_degenerate.bin"), d3["пароль"])[0]] == d3["записи"]
case(5, "Вироджений алфавіт має код довжини 1", deg_ok, "length=1, code=0, ref_03 читається")

# 6. ДП >= жадібний на всіх виданих наборах
p = default_params(cfg)
budget = window_budget(cfg, p)
model = load_model("results/model.npz")
worse = 0
windows = 0
for path in ["data/train.csv", "data/valid.csv", "data/session.csv"]:
    recs, _ = load_csv(path)
    vals = event_values(predict_classes(model, recs, cfg), recs, cfg)
    all_b = [pack_record(r) for r in recs]
    for start in range(0, len(recs), p["window"]):
        wb, wv = all_b[start:start + p["window"]], vals[start:start + p["window"]]
        _, w, _, _, _ = plan_window(wb, wv, cfg, p)
        u_dp = knapsack.dp_copies(w, wv, budget, p["f2"], p["f3"])[1]
        u_gr = knapsack.greedy_copies(w, wv, budget, p["f2"], p["f3"])[1]
        windows += 1
        if u_dp + 1e-9 < u_gr:
            worse += 1
case(6, "ДП дає корисність не меншу за жадібний baseline на всіх виданих наборах",
     worse == 0, "%d вікон train/valid/session, гірше - %d" % (windows, worse))

# 7, 8. сценарії
classes = session_classes()
times = []
for num, sc in [(7, "штатний"), (8, "деградований")]:
    losses = []
    for seed in SEEDS:
        res = run_session(session, cfg, sc, seed, classes=classes)
        losses.append(1 - res["R"])
        times.append(res["t_s"])
    text = ("У штатному сценарії відновлено всі критичні події" if num == 7 else
            "У деградованому сценарії втрата критичних подій не перевищує поріг")
    case(num, text, max(losses) <= limits[sc], "макс. втрата %.3f, поріг %.2f, %d seed" % (max(losses), limits[sc], len(SEEDS)))

# 9. перцептрон
with open("results/classifier_metrics.json", encoding="utf-8") as f:
    clf = json.load(f)
case(9, "Перцептрон: macro-F1 >= F1_min (holdout рахує викладач)",
     clf["valid_macro_f1"] >= clf["F1_min"],
     "valid %.3f, session %.3f, F1_min %.2f" % (clf["valid_macro_f1"], clf["session_macro_f1"], clf["F1_min"]))

# 10, 11. відхилення пошкоджених реплік
with open(os.path.join(REF, "expected_failures.json"), encoding="utf-8") as f:
    expected = json.load(f)
pw = cfg["пароль_сеансу"]
got = {name: error_code(read_ref(name), pw) for name in expected}
ok10 = all(got[n] == expected[n]["очікуваний_код"] for n in ["bad_01_tampered_byte.bin", "bad_02_tampered_tag.bin"])
case(10, "Змінений шифротекст або HMAC відхиляється", ok10, "bad_01, bad_02 -> E_HMAC")
wrong_pw = error_code(read_ref("ref_04_session.bin"), "wrong-password") == "E_HMAC"
ok11 = wrong_pw and all(got[n] == expected[n]["очікуваний_код"] for n in
                        ["bad_03_truncated.bin", "bad_04_bad_magic.bin", "bad_05_reserved.bin"])
case(11, "Невірний пароль, обрізана репліка, поганий magic і reserved відхиляються", ok11,
     "невірний пароль -> E_HMAC, bad_03/04/05 -> E_LENGTH/E_MAGIC/E_PAYLOAD")

# 12. усі еталонні пакети + ліміт часу
all_ok = all(got[n] == expected[n]["очікуваний_код"] for n in expected)
for name in sorted(os.listdir(REF)):
    if name.startswith("ref_") and name.endswith(".json"):
        with open(os.path.join(REF, name), encoding="utf-8") as f:
            d = json.load(f)
        recs, cls = open_ref(read_ref(name[:-5] + ".bin"), d["пароль"])
        if [parse_record(r) for r in recs] != d["записи"] or cls != d["class_table"]:
            all_ok = False
t0 = time.perf_counter()
for sc in cfg["реплікація"]["сценарії"]:
    run_session(session, cfg, sc, SEEDS[0], model=model)
full_t = time.perf_counter() - t0
limit = cfg["ліміт_часу_повного_прогону_секунд"]
case(12, "Усі еталонні пакети читаються; повний прогін укладається в ліміт часу",
     all_ok and full_t < limit, "6 ref + 7 bad, 3 сценарії за %.1f с (ліміт %d с)" % (full_t, limit))

passed = sum(1 for r in results if r["passed"])
print("C1: %d / 12" % passed)
save_json("c1_results.json", {"passed": passed, "total": 12, "cases": results})
