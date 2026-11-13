# Зони відповідальності (команда з 4 осіб)

Типових зон в умові п'ять, тому «Стиснення» та «Інтеграція/відновлення» об'єднані в одну.
Об'єднання погодити з викладачем на Зрізі 0.

| Зона | Учасник | Рівень | Файли | Тести | C1 |
|---|---|---|---|---|---|
| 1. Стиснення + інтеграція/відновлення | Падалка Поліна | базовий | errors.py, record.py, huffman.py, payload.py, recovery.py, metrics.py, pipeline.py, run.py, reference/, experiments/common.py, run_ablation.py, run_final.py, run_all.py | test_zone1_record.py, test_zone1_huffman.py, test_zone1_payload.py, test_reference.py, test_integration.py | 1-5, 7, 8, 11, 12 |
| 2. ДП / реплікація | Мовчан Анастасія | базовий | knapsack.py, scheduler.py, replication.py (виданий, без змін), experiments/run_dp_vs_greedy.py, run_copies.py, run_scenarios.py | test_zone2_dp.py, test_zone2_scheduler.py | 6 |
| 3. Перцептрон | Степанченко Марія | базовий | data/, features.py, mlp.py, classify.py, experiments/run_classifier.py | test_zone3_features.py, test_zone3_mlp.py | 9 |
| 4. Захист + ГА + атака | Гіда Олена | базовий | crypto.py, ga.py, attack.py, experiments/run_ga.py, run_compress_order.py | test_zone4_crypto.py, test_zone4_ga.py | 10, 11 |

Рев'ю за колом: 1 -> 2 -> 3 -> 4 -> 1.

Інтерфейси:
- `pack_record(dict) -> bytes`, `parse_record(bytes) -> dict`
- `huffman.code_lengths(bytes) -> {symbol: length}`
- `encode_payload(records, lengths, no_compression) -> bytes`, `decode_payload(bytes) -> (records, classes)`
- `make_features(recs) -> масив ознак`, `predict_classes(model, recs, cfg) -> [0/1/2]`
- `dp_copies(w, v, budget, f2, f3, rmax) -> (r_list, utility)`, `greedy_copies(...)` - те саме
- `place(r, w, v, nodes, free, max_batch, n_sym, no_comp, mix) -> (batches, placed, sizes)`
- `derive_keys(password, salt)`, `build_replica(...)`, `open_replica(bytes, k_enc, k_mac)`
- `recover(replicas, password) -> (records, rejected)`
- `Replication(scenario, seed).store(node, blob)`, `.surviving_blobs()`, `.stats()` - з replication.py
