# Зона 4. Таблиця 4: стиснення до і після шифрування
from common import session, save_csv
from src.record import pack_record
from src.payload import encode_payload
from src import huffman, crypto

all_b = [pack_record(r) for r in session]
raw = b"".join(all_b)
key, nonce = b"k" * 32, b"n" * 12

c_then_e = len(crypto.aes_ctr(key, nonce, encode_payload(all_b, huffman.code_lengths(raw))))
enc = crypto.aes_ctr(key, nonce, raw)
enc_lengths = huffman.code_lengths(enc)
e_then_c = len(huffman.encode(enc, enc_lengths)) + 2 * len(enc_lengths)

save_csv("compress_order.csv", [
    {"order": "raw (без стиснення)", "bytes": len(raw)},
    {"order": "compress -> encrypt", "bytes": c_then_e},
    {"order": "encrypt -> compress", "bytes": e_then_c}])
