# Зона 1, просунутий рівень: адаптивний Гаффман проти канонічного.
# Канонічний мусить передати таблицю (2 байти на символ), адаптивний - ні,
# тому на малих блоках адаптивний вигідніший, а на великих різниця зменшується.
import time
from common import session, save_csv
from src.record import pack_record
from src import huffman
from src import adaptive_huffman as ah

all_b = [pack_record(r) for r in session]
rows = []
for size in [4, 8, 16, 32, 48, 96, 1200]:
    raw = canon = adapt = 0
    t_c = t_a = 0.0
    ok = True
    for start in range(0, len(all_b), size):
        data = b"".join(all_b[start:start + size])
        t0 = time.perf_counter()
        lengths = huffman.code_lengths(data)
        canon += 2 * len(lengths) + len(huffman.encode(data, lengths))
        t_c += time.perf_counter() - t0
        t0 = time.perf_counter()
        stream, _ = ah.encode(data)
        adapt += len(stream)
        t_a += time.perf_counter() - t0
        ok = ok and ah.decode(stream, len(data)) == data      # побайтове відновлення
        raw += len(data)
    rows.append({"block_records": size, "raw_bytes": raw, "canonical_bytes": canon,
                 "adaptive_bytes": adapt, "adaptive_gain_%": round(100 * (canon - adapt) / canon, 2),
                 "canonical_ms": round(t_c * 1000, 1), "adaptive_ms": round(t_a * 1000, 1),
                 "roundtrip_ok": ok})
    print("блок %4d записів: канонічний %d, адаптивний %d байтів" % (size, canon, adapt))
save_csv("adaptive_huffman.csv", rows)
