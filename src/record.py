# Зона 1. Запис журналу - рівно 16 байтів, big-endian (FORMAT_SPEC_P3 §1)
#
#  зсув  розмір  поле
#   0     4      timestamp_ms   uint32   -> I
#   4     2      source_id      uint16   -> H
#   6     2      event_code     uint16   -> H
#   8     4      metric_value   int32    -> i (зі знаком)
#   12    1      category       uint8    -> B
#   13    1      severity       uint8    -> B
#   14    1      critical       uint8    -> B
#   15    1      record_flags   uint8    -> B
# ">" у форматі означає big-endian. Разом 4+2+2+4+1+1+1+1 = 16.
import struct
from src.errors import ReplicaError

RECORD_FMT = ">IHHiBBBB"
RECORD_SIZE = 16
FIELDS = ["timestamp_ms", "source_id", "event_code", "metric_value",
          "category", "severity", "critical", "record_flags"]

# допустимі діапазони - тип поля може вмістити більше, але специфікація звужує
LIMITS = {
    "timestamp_ms": (0, 2**32 - 1),
    "source_id": (1, 4095),
    "event_code": (0, 2**16 - 1),
    "metric_value": (-2**31, 2**31 - 1),
    "category": (0, 15),
    "severity": (0, 7),
    "critical": (0, 1),
    "record_flags": (0, 3),
}


def check_record(rec):
    # некоректний запис відкидаємо одразу, а не сподіваємося, що далі "розберуться"
    for name in FIELDS:
        lo, hi = LIMITS[name]
        if not (lo <= rec[name] <= hi):
            raise ReplicaError("E_PAYLOAD", "bad field " + name)


def pack_record(rec):
    check_record(rec)
    values = [rec[name] for name in FIELDS]
    return struct.pack(RECORD_FMT, *values)


def parse_record(data):
    if len(data) != RECORD_SIZE:
        raise ReplicaError("E_PAYLOAD", "record must be 16 bytes")
    values = struct.unpack(RECORD_FMT, data)
    rec = {}
    for i in range(len(FIELDS)):
        rec[FIELDS[i]] = values[i]
    check_record(rec)
    return rec
