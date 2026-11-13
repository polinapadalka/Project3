# Зона 1. Канонічний Гаффман (FORMAT_SPEC_P3 §3.2)
#
# Ідея: частіші байти отримують коротші коди. У репліку пишемо лише пари
# (символ, довжина коду), а не дерево. Канонічний алгоритм за довжинами
# однозначно відновлює ті самі коди і в кодері, і в декодері.
import heapq
from src.errors import ReplicaError

MAX_LEN = 15     # максимальна довжина коду за специфікацією


def count_freq(data):
    freq = {}
    for b in data:
        freq[b] = freq.get(b, 0) + 1
    return freq


def build_lengths(freq):
    # Дерево будуємо через чергу з пріоритетами (бінарна купа):
    # щоразу дістаємо два найрідші вузли за O(log n) і зливаємо їх.
    # Нам потрібні лише глибини листків, тому замість дерева тримаємо
    # список символів у кожному вузлі і при злитті +1 до їхньої довжини.
    # Складність O(n log n), n - кількість різних символів (до 256).
    if len(freq) == 0:
        return {}
    if len(freq) == 1:
        # вироджений алфавіт: дерево з одного листка мало б глибину 0,
        # тому специфікація фіксує length = 1, code = 0
        return {list(freq.keys())[0]: 1}
    heap = []
    order = 0        # лічильник, щоб при рівних частотах порядок був детермінований
    for sym in sorted(freq):
        heapq.heappush(heap, (freq[sym], order, [sym]))
        order += 1
    lengths = {sym: 0 for sym in freq}
    while len(heap) > 1:
        f1, _, s1 = heapq.heappop(heap)
        f2, _, s2 = heapq.heappop(heap)
        for sym in s1 + s2:
            lengths[sym] += 1      # усі символи піддерева опустилися на рівень нижче
        heapq.heappush(heap, (f1 + f2, order, s1 + s2))
        order += 1
    return lengths


def code_lengths(data):
    freq = count_freq(data)
    lengths = build_lengths(freq)
    # Length-limited: якщо якийсь код довший за 15, зменшуємо частоти вдвічі
    # (не нижче 1) і будуємо дерево знову. Дерево стає "рівнішим", таблиця
    # лишається повною (Kraft = 1). Різні символи при цьому не зливаються.
    while len(lengths) > 0 and max(lengths.values()) > MAX_LEN:
        for sym in freq:
            freq[sym] = (freq[sym] + 1) // 2
        lengths = build_lengths(freq)
    return lengths


def check_lengths(lengths):
    # Перевірка таблиці з репліки, яку ми не створювали самі
    if len(lengths) == 0:
        return
    for sym in lengths:
        if not (0 <= sym <= 255) or not (1 <= lengths[sym] <= MAX_LEN):
            raise ReplicaError("E_CODE", "bad code length")
    if len(lengths) == 1:
        if list(lengths.values())[0] != 1:
            raise ReplicaError("E_CODE", "single symbol must have length 1")
        return
    # Нерівність Крафта: сума 2^(-l) по всіх кодах.
    # < 1 - у дереві "дірки" (неповний код), > 1 - такий префіксний код неможливий.
    # Множимо на 2^15, щоб рахувати в цілих числах без похибок float.
    total = 0
    for sym in lengths:
        total += 2 ** (MAX_LEN - lengths[sym])
    if total != 2 ** MAX_LEN:
        raise ReplicaError("E_CODE", "Kraft sum != 1")


def canonical_codes(lengths):
    # Стандартний канонічний алгоритм (як у DEFLATE):
    # bl_count[l] - скільки кодів довжини l;
    # next_code[l] - перший код довжини l;
    # символи однієї довжини отримують коди підряд у порядку зростання символу.
    check_lengths(lengths)
    bl_count = [0] * (MAX_LEN + 1)
    for sym in lengths:
        bl_count[lengths[sym]] += 1
    next_code = [0] * (MAX_LEN + 2)
    code = 0
    for bits in range(1, MAX_LEN + 1):
        # після всіх кодів довжини bits-1 переходимо на рівень нижче (зсув вліво)
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code
    codes = {}
    for sym in sorted(lengths):
        n = lengths[sym]
        codes[sym] = next_code[n]
        next_code[n] += 1
    return codes


def encode(data, lengths):
    # Біти пакуємо MSB-first: перший біт коду - старший біт байта
    codes = canonical_codes(lengths)
    out = bytearray()
    acc = 0          # накопичувач бітів
    nbits = 0        # скільки бітів зараз у накопичувачі
    for b in data:
        if b not in codes:
            raise ReplicaError("E_CODE", "symbol not in table")
        acc = (acc << lengths[b]) | codes[b]
        nbits += lengths[b]
        while nbits >= 8:            # набрався повний байт - виштовхуємо його
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
        acc &= (1 << nbits) - 1      # лишаємо тільки ще не записані біти
    if nbits > 0:
        out.append((acc << (8 - nbits)) & 0xFF)   # хвіст доповнюємо нулями
    return bytes(out)


def decode(stream, lengths, symbol_count):
    # Читаємо по одному біту, поки (довжина, код) не знайдеться в таблиці.
    # Код префіксний, тому перше ж співпадіння - правильний символ.
    codes = canonical_codes(lengths)
    table = {}
    for sym in codes:
        table[(lengths[sym], codes[sym])] = sym
    out = bytearray()
    code = 0
    clen = 0
    pos = 0
    total_bits = len(stream) * 8
    # зупиняємося рівно після symbol_count символів, а не в кінці потоку
    while len(out) < symbol_count:
        if pos >= total_bits:
            raise ReplicaError("E_CODE", "bitstream ended too early")
        bit = (stream[pos // 8] >> (7 - pos % 8)) & 1
        pos += 1
        code = (code << 1) | bit
        clen += 1
        if (clen, code) in table:
            out.append(table[(clen, code)])
            code = 0
            clen = 0
        elif clen >= MAX_LEN:
            raise ReplicaError("E_CODE", "unknown code")
    # Після останнього символу дозволено лише нульовий padding останнього байта.
    # Зайвий повний байт або ненульові біти - помилка (bad_07_extra_bitstream).
    if (pos + 7) // 8 != len(stream):
        raise ReplicaError("E_PAYLOAD", "extra bytes in bitstream")
    while pos < total_bits:
        if (stream[pos // 8] >> (7 - pos % 8)) & 1:
            raise ReplicaError("E_PAYLOAD", "non-zero padding")
        pos += 1
    return bytes(out)
