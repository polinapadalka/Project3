import json
import os
import struct

with open(os.path.join(os.path.dirname(__file__), "..", "team_config.json"), encoding="utf-8") as f:
    CFG = json.load(f)


def rec(ts=1000, src=5, code=300, metric=-42, cat=3, sev=2, crit=0, flags=1):
    return {"timestamp_ms": ts, "source_id": src, "event_code": code, "metric_value": metric,
            "category": cat, "severity": sev, "critical": crit, "record_flags": flags}


def seal(payload, password="pw", salt=b"s" * 16, nonce=b"n" * 12):
    from src import crypto      # тут, щоб helpers працював ще до появи crypto.py
    k_enc, k_mac = crypto.derive_keys(password, salt)
    return crypto.build_replica(1, 0, salt, nonce, payload, k_enc, k_mac), (k_enc, k_mac)


def raw_payload(record_count, flags, reserved, table, symbol_count, stream):
    out = struct.pack(">HBBH", record_count, flags, reserved, len(table))
    for sym, ln in table:
        out += struct.pack(">BB", sym, ln)
    out += struct.pack(">II", symbol_count, len(stream)) + stream
    return out


def flip_byte(data, pos):
    b = bytearray(data)
    b[pos] ^= 0x01
    return bytes(b)
