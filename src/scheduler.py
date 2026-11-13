# Зона 2. Жадібний планувальник вузлів (евристика, не доказово оптимальний)
#
# ДП сказав, СКІЛЬКИ копій зробити кожній події. Тут вирішуємо, КУДИ:
#  - копії однієї події - лише на різних вузлах (r=3 означає A, B і C),
#    бо дві копії на одному вузлі зникнуть разом, якщо вузол відмовить;
#  - на кожному вузлі не більше одного batch на вікно, batch <= max_batch_bytes;
#  - ДП не знає місткості окремих вузлів, тому частина копій може не влізти.
from src.payload import payload_size

OVERHEAD = 43 + 32   # заголовок репліки + HMAC


def batch_encoding(n_records, bits, n_sym, no_comp):
    # Повертає (розмір репліки, чи писати сирими байтами).
    # Таблиця Гаффмана коштує 2 байти на символ у КОЖНОМУ batch, тому для
    # маленького batch сирі 16 байтів на запис часто дешевші за стиснення.
    raw = OVERHEAD + payload_size(n_records, 0, 0, True)
    if no_comp:
        return raw, True
    comp = OVERHEAD + payload_size(n_records, n_sym, bits, False)
    if raw <= comp:
        return raw, True
    return comp, False


def batch_bytes(n_records, bits, n_sym, no_comp):
    return batch_encoding(n_records, bits, n_sym, no_comp)[0]


def place(r_list, w, v, nodes, free, max_batch, n_sym, no_comp, mix):
    # nodes - словник вузлів зі сценарію ("A": {...}), free - залишкова місткість,
    # mix - наскільки надійність вузла важливіша за вільне місце (0..1).
    # free тут не змінюється: план можна будувати кілька разів і обрати один.
    names = sorted(nodes)
    rel = {name: nodes[name]["reliability_weight"] for name in names}
    cap = {name: nodes[name]["capacity_bytes"] for name in names}
    batch = {name: [] for name in names}
    bits = {name: 0 for name in names}

    # Порядок - за приростом очікуваної відновлюваності на один біт (v_i / w_i):
    # спершу події, де кожен байт зберігає найбільше цінності. Якщо місця
    # забракне, недоотримають ресурс найменш вигідні.
    order = sorted(range(len(r_list)), key=lambda i: (-v[i] / w[i], i))
    placed = [0] * len(r_list)
    for i in order:
        if r_list[i] == 0:
            continue
        ok = []
        for name in names:
            size = batch_bytes(len(batch[name]) + 1, bits[name] + w[i], n_sym, no_comp)
            if size <= max_batch and size <= free[name]:
                # Кожна наступна копія підвищує шанс зберегти подію на rel_вузла,
                # помножене на шанс, що всі попередні копії втрачено. Цей множник
                # однаковий для всіх вузлів, тому вузли порівнюємо за reliability_weight
                # з поправкою на залишкову місткість.
                frac = (free[name] - size) / cap[name]
                score = mix * rel[name] + (1 - mix) * frac
                ok.append((-score, name))
        ok.sort()                                     # найкращі вузли першими
        for k in range(min(r_list[i], len(ok))):     # кожна копія - на інший вузол
            name = ok[k][1]
            batch[name].append(i)
            bits[name] += w[i]
            placed[i] += 1

    sizes = {}
    for name in names:
        batch[name].sort()      # у batch записи йдуть у порядку часу
        sizes[name] = batch_bytes(len(batch[name]), bits[name], n_sym, no_comp) if batch[name] else 0
    return batch, placed, sizes
