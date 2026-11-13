# Зона 4. Модуль атаки. AES не атакуємо - лише навмисно слабкий навчальний шифр.
#
# Слабкий шифр: побайтовий Віженер - до кожного байта додаємо байт ключа
# за модулем 256, ключ із 6 байтів повторюється.
# ГА шукає ключ: fitness - наскільки розшифрований текст схожий на звичайний
# журнал за частотами символів з окремого навчального набору.
# Повний перебір - 256^6 ~ 2.8 * 10^14 ключів, ГА знаходить за кілька тисяч обчислень,
# бо кожен байт ключа можна "доводити" окремо, а кросовер збирає вдалі байти разом.
import math
import random
import numpy as np

KEY_LEN = 6


def make_training_text(n_lines, seed):
    # навчальний набір: текстові рядки журналу
    rng = random.Random(seed)
    lines = []
    t = 1000
    for _ in range(n_lines):
        t += rng.randint(20, 900)
        lines.append("t=%d src=%d code=%d sev=%d cat=%d ok\n" % (
            t, rng.randint(1, 4095), rng.randint(100, 7999), rng.randint(0, 7), rng.randint(0, 15)))
    return "".join(lines).encode()


def weak_encrypt(data, key):
    return bytes((data[i] + key[i % len(key)]) % 256 for i in range(len(data)))


def weak_decrypt(data, key):
    return bytes((data[i] - key[i % len(key)]) % 256 for i in range(len(data)))


def char_stats(train_text):
    # log-частоти символів; +1 до кожної частоти, щоб не було log(0)
    freq = [1] * 256
    for b in train_text:
        freq[b] += 1
    total = sum(freq)
    return [math.log(f / total) for f in freq]


def key_fitness(key, ct, stats):
    # середня log-ймовірність розшифрованого тексту.
    # Правильний ключ дає "звичайні" символи логу -> найбільше значення.
    c = np.frombuffer(ct, dtype=np.uint8).astype(int)
    k = np.resize(np.array(key, dtype=int), len(c))
    return float(np.mean(np.array(stats)[(c - k) % 256]))


def tamper(replica, pos):
    # атака на цілісність: змінюємо один біт репліки.
    # Правильний відновлювач має відхилити її з кодом E_HMAC.
    b = bytearray(replica)
    b[pos] ^= 0x01
    return bytes(b)
