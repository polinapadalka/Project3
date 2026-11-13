# Зона 1. Контрольовані коди помилок (FORMAT_SPEC_P3 §5)
# Будь-яка погана репліка має закінчуватися саме цим винятком з кодом,
# а не випадковим IndexError чи struct.error.
#   E_SHORT   - репліка коротша за заголовок + HMAC
#   E_MAGIC   - не "LOG3" або не та версія
#   E_LENGTH  - довжина не збігається з ciphertext_len
#   E_HMAC    - не пройшла перевірка цілісності (або невірний пароль)
#   E_PAYLOAD - порушено структуру або інваріанти навантаження
#   E_CODE    - неможлива таблиця Гаффмана або код, якого немає в таблиці


class ReplicaError(Exception):
    def __init__(self, code, msg=""):
        super().__init__(code + (": " + msg if msg else ""))
        self.code = code
