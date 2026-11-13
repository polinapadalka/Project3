# Зона 1. Навантаження репліки - те, що потім шифрується (FORMAT_SPEC_P3 §3)
#
#   record_count  2 байти
#   flags         1
#   reserved      1   (має бути 0)
#   table_nsym    2
#   table         2 x table_nsym   пари (symbol, length), symbol за зростанням
#   symbol_count  4   завжди 16 x record_count
#   bitstream_len 4
#   bitstream     bitstream_len
#   class_table   record_count     лише якщо flags bit2
import struct
from src.errors import ReplicaError
from src.record import RECORD_SIZE, parse_record
from src import huffman

FLAG_PREDICTION = 1        # bit0 - командне розширення (src/prediction.py)
FLAG_MULTI_COPY_META = 2   # bit1 - теж розширення
FLAG_CLASS_TABLE = 4       # bit2 - є таблиця класів
FLAG_NO_COMPRESSION = 8    # bit3 - записи лежать сирими байтами


def encode_payload(records, lengths, no_compression=False, class_table=None, prediction=False):
    # records - список 16-байтових записів одного batch;
    # lengths - таблиця Гаффмана ВСЬОГО вікна (вона може мати символи,
    # яких у цьому batch немає - це нормально, таблиця все одно повна).
    # prediction=True - розширення: стискаємо не записи, а залишки прогнозу,
    # таблиця тоді будується за залишками саме цього batch
    flags = 0
    if prediction and not no_compression and len(records) > 0:
        from src import prediction as prediction_mod     # розширення зони 3
        records = prediction_mod.transform(records)
        lengths = huffman.code_lengths(b"".join(records))
        flags |= FLAG_PREDICTION
    data = b"".join(records)
    if no_compression or len(records) == 0:
        if len(records) > 0:
            flags |= FLAG_NO_COMPRESSION
        table = {}           # для сирих байтів таблиця не потрібна
        stream = data
    else:
        table = lengths
        stream = huffman.encode(data, table)
    if class_table is not None:
        flags |= FLAG_CLASS_TABLE

    out = bytearray()
    out += struct.pack(">HBBH", len(records), flags, 0, len(table))
    for sym in sorted(table):
        out += struct.pack(">BB", sym, table[sym])
    out += struct.pack(">II", len(data), len(stream))
    out += stream
    if class_table is not None:
        out += bytes(class_table)
    return bytes(out)


def payload_size(n_records, n_sym, bits, no_compression=False):
    # Точний розмір payload без побудови. Потрібен планувальнику, щоб
    # заздалегідь знати, чи batch вміститься в max_batch_bytes.
    # 14 = 2+1+1+2 (заголовок) + 4+4 (symbol_count, bitstream_len)
    if no_compression:
        return 14 + 16 * n_records
    return 14 + 2 * n_sym + (bits + 7) // 8


def decode_payload(p, extensions=False):
    # Строгий розбір: кожне поле і кожен інваріант перевіряємо явно.
    # Сюди потрапляє лише payload, що вже пройшов HMAC.
    # extensions=True - приймаємо наше розширення PREDICTION; для чужих
    # реплік і еталонних пакетів діє лише базовий формат.
    if len(p) < 6:
        raise ReplicaError("E_PAYLOAD", "payload too short")
    record_count, flags, reserved, nsym = struct.unpack(">HBBH", p[0:6])
    if reserved != 0 or flags & 0xF0:
        raise ReplicaError("E_PAYLOAD", "reserved bits")
    if flags & FLAG_MULTI_COPY_META:
        raise ReplicaError("E_PAYLOAD", "unknown extension")
    if flags & FLAG_PREDICTION and (not extensions or flags & FLAG_NO_COMPRESSION):
        # для базового декодера PREDICTION - невідоме розширення
        raise ReplicaError("E_PAYLOAD", "unknown extension")
    pos = 6
    if len(p) < pos + 2 * nsym + 8:
        raise ReplicaError("E_PAYLOAD", "payload too short")
    lengths = {}
    last = -1
    for i in range(nsym):
        sym, ln = p[pos], p[pos + 1]
        pos += 2
        if sym <= last:      # символи мають іти строго за зростанням, без повторів
            raise ReplicaError("E_PAYLOAD", "table not sorted")
        last = sym
        lengths[sym] = ln
    symbol_count, stream_len = struct.unpack(">II", p[pos:pos + 8])
    pos += 8
    # основний інваріант базового формату: кожен запис - 16 байтів-символів
    if symbol_count != RECORD_SIZE * record_count:
        raise ReplicaError("E_PAYLOAD", "symbol_count != 16*record_count")

    has_classes = flags & FLAG_CLASS_TABLE
    expected = pos + stream_len + (record_count if has_classes else 0)
    if len(p) != expected:
        raise ReplicaError("E_PAYLOAD", "bad payload length")
    stream = p[pos:pos + stream_len]
    pos += stream_len

    if flags & FLAG_NO_COMPRESSION:
        if nsym != 0 or stream_len != symbol_count:
            raise ReplicaError("E_PAYLOAD", "bad raw block")
        data = stream
    elif record_count == 0:
        # порожній журнал - коректна репліка без таблиці й без даних
        if nsym != 0 or stream_len != 0:
            raise ReplicaError("E_PAYLOAD", "bad empty payload")
        data = b""
    else:
        data = huffman.decode(stream, lengths, symbol_count)

    if flags & FLAG_PREDICTION:
        from src import prediction as prediction_mod
        chunks = [data[i * 16:(i + 1) * 16] for i in range(record_count)]
        data = b"".join(prediction_mod.inverse(chunks))   # залишки -> записи

    records = []
    for i in range(record_count):
        chunk = data[i * 16:(i + 1) * 16]
        parse_record(chunk)          # перевірка діапазонів полів
        records.append(chunk)

    classes = None
    if has_classes:
        classes = list(p[pos:pos + record_count])
        for c in classes:
            if c > 2:                # 3..255 зарезервовано
                raise ReplicaError("E_PAYLOAD", "bad class value")
    return records, classes
