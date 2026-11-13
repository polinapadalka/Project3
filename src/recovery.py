# Зона 1. Відновлювач: verify -> decrypt -> decode -> dedup -> журнал
from src.errors import ReplicaError
from src import crypto
from src.payload import decode_payload


def recover(replicas, password):
    # replicas - усі репліки, що вижили на будь-якому з вузлів
    keys = {}            # PBKDF2 дорогий, тому рахуємо один раз на кожну сіль
    seen = set()         # множина: швидка перевірка "цей запис уже бачили?"
    restored = []
    rejected = {}        # скільки реплік відкинуто і з яким кодом
    for data in replicas:
        try:
            h = crypto.parse_header(data)
            if h["salt"] not in keys:
                keys[h["salt"]] = crypto.derive_keys(password, h["salt"])
            k_enc, k_mac = keys[h["salt"]]
            _, payload = crypto.open_replica(data, k_enc, k_mac)
            records, _ = decode_payload(payload, extensions=True)   # наші репліки можуть мати PREDICTION
        except ReplicaError as e:
            # ловимо тільки свої контрольовані помилки, а не будь-який Exception:
            # справжній баг у коді має падати, а не тихо ховатися
            rejected[e.code] = rejected.get(e.code, 0) + 1
            continue
        for rec in records:
            # Дублікати - це нормально: одна подія може вижити в 3 копіях.
            # Ідентичність запису - його 16 байтів (FORMAT_SPEC_P3 §1),
            # timestamp_ms у сеансі унікальний, тож різні події не зіллються.
            if rec not in seen:
                seen.add(rec)
                restored.append(rec)
    # перші 4 байти - timestamp big-endian, тому звичайне сортування байтів
    # дає саме порядок за часом
    restored.sort()
    return restored, rejected
