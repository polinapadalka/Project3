# Зона 4. Захист репліки (FORMAT_SPEC_P3 §2, §4, §5)
#
#   HEADER 43 | CIPHERTEXT | HMAC-SHA256 32
#
# Порядок: стиснути -> зашифрувати -> HMAC (Encrypt-then-MAC).
# Стиснення йде ДО шифрування, бо шифротекст схожий на випадкові байти
# і майже не стискається.
import hashlib
import hmac
import os
import struct
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from src.errors import ReplicaError

MAGIC = b"LOG3"
VERSION = 1
HEADER_SIZE = 43     # 4 magic + 1 version + 2 group + 4 seq + 16 salt + 12 nonce + 4 len
TAG_SIZE = 32
PBKDF2_ITER = 100000


def derive_keys(password, salt):
    # З пароля і солі виводимо 64 байти, ділимо на два НЕЗАЛЕЖНІ ключі:
    # K_enc - для шифрування, K_mac - для HMAC. Один ключ на дві задачі
    # використовувати не можна.
    # 100000 ітерацій навмисно повільні - так перебір паролів стає дорогим.
    master = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITER, dklen=64)
    return master[0:32], master[32:64]


def aes_ctr(key, nonce, data):
    # AES-256-CTR: шифруємо лічильники і XOR-имо результат з даними.
    # Блок лічильника: nonce(12) || uint32_BE(i), i починається з 1.
    # Шифрування і розшифрування - одна й та сама операція.
    # Якщо nonce повториться з тим самим ключем, потік ключа теж повториться,
    # і XOR двох шифротекстів розкриє XOR відкритих текстів.
    n_blocks = (len(data) + 15) // 16
    counters = b""
    for i in range(1, n_blocks + 1):
        counters += nonce + struct.pack(">I", i)
    enc = Cipher(algorithms.AES(key), modes.ECB()).encryptor()   # сам блок AES
    stream = enc.update(counters) + enc.finalize()
    out = bytearray(len(data))
    for i in range(len(data)):
        out[i] = data[i] ^ stream[i]
    return bytes(out)


class NonceSource:
    # Унікальний nonce у межах одного K_enc: випадковий префікс + лічильник.
    # Лічильник гарантує, що в межах сеансу nonce ніколи не повториться.
    def __init__(self):
        self.prefix = os.urandom(4)
        self.counter = 0
        self.used = set()

    def next(self):
        self.counter += 1
        nonce = self.prefix + struct.pack(">Q", self.counter)
        if nonce in self.used:
            raise RuntimeError("nonce reuse")
        self.used.add(nonce)
        return nonce


def build_replica(group_id, batch_seq, salt, nonce, payload, k_enc, k_mac):
    ct = aes_ctr(k_enc, nonce, payload)
    header = MAGIC + struct.pack(">BHI", VERSION, group_id, batch_seq) + salt + nonce + struct.pack(">I", len(ct))
    # HMAC рахуємо над заголовком І шифротекстом: підміна будь-якого поля
    # заголовка (наприклад, nonce чи довжини) теж буде помічена
    tag = hmac.new(k_mac, header + ct, hashlib.sha256).digest()
    return header + ct + tag


def parse_header(data):
    # кроки 1-3 порядку приймання - лише дешеві структурні перевірки
    if len(data) < HEADER_SIZE + TAG_SIZE:
        raise ReplicaError("E_SHORT")
    if data[0:4] != MAGIC or data[4] != VERSION:
        raise ReplicaError("E_MAGIC")
    group_id, batch_seq = struct.unpack(">HI", data[5:11])
    salt = data[11:27]
    nonce = data[27:39]
    ct_len = struct.unpack(">I", data[39:43])[0]
    if len(data) != HEADER_SIZE + ct_len + TAG_SIZE:
        raise ReplicaError("E_LENGTH")
    return {"group_id": group_id, "batch_seq": batch_seq, "salt": salt,
            "nonce": nonce, "ct_len": ct_len}


def open_replica(data, k_enc, k_mac):
    # Порядок приймання: довжина -> magic/version -> точна довжина -> HMAC -> decrypt
    h = parse_header(data)
    body = data[:HEADER_SIZE + h["ct_len"]]
    tag = data[HEADER_SIZE + h["ct_len"]:]
    good = hmac.new(k_mac, body, hashlib.sha256).digest()
    # compare_digest порівнює за сталий час: звичайне == зупиняється на першому
    # різному байті, і за часом відповіді можна було б підбирати tag
    if not hmac.compare_digest(good, tag):
        # STOP: змінені чи чужі дані не розшифровуємо і не розбираємо взагалі
        raise ReplicaError("E_HMAC")
    payload = aes_ctr(k_enc, h["nonce"], data[HEADER_SIZE:HEADER_SIZE + h["ct_len"]])
    return h, payload
