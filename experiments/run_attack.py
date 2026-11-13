# Зона 4. Сценарій атаки (розділ 3.7 умови) -> results/attack.json
#  1) ГА відновлює ключ навмисно слабкого шифру (не AES);
#  2) модифікація байтів справжніх реплік сеансу - відновлювач має відхилити за HMAC.
import os
from common import cfg, session, save_json
from src import attack, ga, crypto
from src.errors import ReplicaError
from src.record import pack_record
from src.payload import encode_payload
from src import huffman

# --- 1. відновлення ключа слабкого шифру ---
stats = attack.char_stats(attack.make_training_text(500, 99))   # окремий навчальний набір
text = attack.make_training_text(200, 1)                         # перехоплений відкритий текст
key = bytes([17, 200, 3, 99, 250, 64])                           # секрет, якого атакуючий не знає
ct = attack.weak_encrypt(text, key)
genes = [(0, 255, "int")] * attack.KEY_LEN
fit = lambda k: attack.key_fitness(k, ct, stats)
runs = []
for seed in range(1, 9):
    best, f, hist = ga.run_ga(fit, genes, pop_size=40, generations=150, seed=seed, p_mut=0.3)
    _, f_rs, _ = ga.random_search(fit, genes, hist[-1]["evals"], seed=seed)
    found = bytes(best) == key
    runs.append({"seed": seed, "ga_found_key": found, "ga_fitness": f, "random_fitness": f_rs,
                 "evals": hist[-1]["evals"]})
    print("seed %d: ГА %s (fitness %.3f), random search %.3f, %d обчислень" % (
        seed, "знайшов ключ" if found else "не знайшов", f, f_rs, hist[-1]["evals"]))

# --- 2. атака на цілісність реплік ---
salt = os.urandom(16)
k_enc, k_mac = crypto.derive_keys(cfg["пароль_сеансу"], salt)
nonces = crypto.NonceSource()
replicas = []
for start in range(0, len(session), 48):
    wb = [pack_record(r) for r in session[start:start + 48]]
    p = encode_payload(wb, huffman.code_lengths(b"".join(wb)))
    replicas.append(crypto.build_replica(1, start // 48, salt, nonces.next(), p, k_enc, k_mac))
tried = 0
rejected = 0
for rep in replicas:
    for pos in range(43, len(rep), 37):          # байти шифротексту і тегу
        tried += 1
        try:
            crypto.open_replica(attack.tamper(rep, pos), k_enc, k_mac)
        except ReplicaError as e:
            if e.code == "E_HMAC":
                rejected += 1
print("змінено байтів: %d, відхилено за HMAC: %d" % (tried, rejected))

save_json("attack.json", {
    "weak_cipher": "побайтовий Віженер, ключ 6 байтів",
    "key_recovery": runs,
    "ga_success_rate": sum(1 for r in runs if r["ga_found_key"]) / len(runs),
    "tamper": {"modified": tried, "rejected_by_hmac": rejected},
})
