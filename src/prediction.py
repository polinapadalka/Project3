# Зона 3, просунутий рівень (задача Б). Передбачення полів і кодування залишків.
#
# Сусідні події схожі: час росте маленькими кроками, джерело часто повторює
# свою category і близьке metric_value, коди йдуть серіями. Тому замість
# самого значення кодуємо залишок = значення - прогноз. Залишки здебільшого
# малі, байти стають повторюваними, і Гаффман стискає краще.
#
#   timestamp_ms  -> різниця з попередньою подією
#   event_code    -> різниця з кодом попередньої події (mod 2^16)
#   metric_value  -> різниця з попереднім metric ТОГО САМОГО джерела (mod 2^32)
#   category      -> різниця з попередньою category того самого джерела (mod 16)
#   решта полів   -> без змін
# Перша подія (і перша подія кожного джерела) кодується як є.
# Перетворення оборотне: декодер іде в тому самому порядку і має ті самі прогнози.
# Це командне розширення формату (flags bit0 PREDICTION), крос-сумісність - лише базовий формат.
import struct

FMT = ">IHHIBBBB"      # як запис, але metric - беззнаковий, бо зберігаємо залишок mod 2^32


def transform(records):
    out = []
    prev_ts = None
    prev_code = None
    last_metric = {}
    last_cat = {}
    for rb in records:
        ts, src, code, metric, cat, sev, crit, flags = struct.unpack(">IHHiBBBB", rb)
        d_ts = ts - prev_ts if prev_ts is not None else ts
        d_code = (code - prev_code) & 0xFFFF if prev_code is not None else code
        d_metric = (metric - last_metric[src]) & 0xFFFFFFFF if src in last_metric else metric & 0xFFFFFFFF
        d_cat = (cat - last_cat[src]) & 0x0F if src in last_cat else cat
        out.append(struct.pack(FMT, d_ts & 0xFFFFFFFF, src, d_code, d_metric, d_cat, sev, crit, flags))
        prev_ts, prev_code = ts, code
        last_metric[src] = metric
        last_cat[src] = cat
    return out


def inverse(residuals):
    out = []
    prev_ts = None
    prev_code = None
    last_metric = {}
    last_cat = {}
    for rb in residuals:
        d_ts, src, d_code, d_metric, d_cat, sev, crit, flags = struct.unpack(FMT, rb)
        ts = (prev_ts + d_ts) & 0xFFFFFFFF if prev_ts is not None else d_ts
        code = (prev_code + d_code) & 0xFFFF if prev_code is not None else d_code
        m = (last_metric[src] + d_metric) & 0xFFFFFFFF if src in last_metric else d_metric
        metric = m - 2**32 if m >= 2**31 else m          # назад у знакове int32
        cat = (last_cat[src] + d_cat) & 0x0F if src in last_cat else d_cat
        out.append(struct.pack(">IHHiBBBB", ts, src, code, metric, cat, sev, crit, flags))
        prev_ts, prev_code = ts, code
        last_metric[src] = metric
        last_cat[src] = cat
    return out
