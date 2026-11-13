# Зона 3, просунутий рівень (D1 #14, #15): передбачення полів і кодування залишків.
from common import cfg, session, SCEN, save_csv, session_classes, mean_run, ga_params
from src.record import pack_record
from src.payload import encode_payload, decode_payload
from src.pipeline import default_params
from src import huffman

W = cfg["реплікація"]["replication_window_records"]
all_b = [pack_record(r) for r in session]
base = pred = 0
roundtrip = True
for start in range(0, len(all_b), W):
    wb = all_b[start:start + W]
    lengths = huffman.code_lengths(b"".join(wb))
    p0 = encode_payload(wb, lengths)
    p1 = encode_payload(wb, lengths, prediction=True)
    base += len(p0)
    pred += len(p1)
    roundtrip = roundtrip and decode_payload(p1, extensions=True)[0] == wb   # D1 #14
print("payload без прогнозу %d, з прогнозом %d байтів, відновлення побайтове: %s" % (base, pred, roundtrip))

# D1 #15: зміна Q без просунутого прогнозу
p = ga_params(default_params(cfg))
classes = session_classes()
rows = [{"part": "розмір payload вікон", "without": base, "with": pred,
         "change_%": round(100 * (pred - base) / base, 2), "roundtrip_ok": roundtrip}]
for sc in SCEN:
    q0 = mean_run(sc, dict(p, prediction=False), classes)
    q1 = mean_run(sc, dict(p, prediction=True), classes)
    rows.append({"part": "Q " + sc, "without": round(q0["Q"], 4), "with": round(q1["Q"], 4),
                 "change_%": round(100 * (q1["Q"] - q0["Q"]) / q0["Q"], 2), "roundtrip_ok": ""})
    print("%-12s Q без прогнозу %.4f, з прогнозом %.4f" % (sc, q0["Q"], q1["Q"]))
save_csv("prediction.csv", rows)
