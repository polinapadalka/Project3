# Проєкт 3. Розумна захищена система пріоритетної реплікації журналів подій

Команда T01 з 4 осіб, розподіл зон - у `docs/zones.md`.

## Встановлення

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Запуск

```bash
python run.py                          # повний тракт, 3 сценарії x 5 seed
python run.py аварійний 1 2 3          # один сценарій на вибраних seed
python experiments/run_all.py          # усі таблиці звіту -> results/
```

Кожну таблицю можна перерахувати й окремо: `python experiments/run_dp_vs_greedy.py` тощо.
`run_classifier.py` треба запустити першим - він навчає модель, яку використовують інші скрипти.

## Тести

```bash
python -m unittest discover -s tests -t .
# або
pytest
```

`tests/test_reference.py` перевіряє еталонні репліки з `reference/` і коди відмов з `expected_failures.json`.

## Структура

```
src/            код системи (власник кожного файла - у docs/zones.md)
replication.py  видана модель реплікації (не змінюється)
team_config.json  виданий конфіг команди
data/           train.csv, valid.csv, session.csv, session.bin
reference/      еталонні та пошкоджені репліки
tests/          unit, negative, reference та integration тести
experiments/    run_*.py - по скрипту на кожну таблицю, run_all.py - усі разом
results/        згенеровані результати (не редагувати вручну)
docs/           zones.md, decisions.md (лог рішень)
```

## Тракт

```
session -> ознаки -> MLP -> value -> Huffman (таблиця вікна) -> DP 0/1/2/3 копій
(з розподілом глобального бюджету між вікнами) -> жадібний планувальник вузлів
-> AES-256-CTR -> HMAC -> replication.py (вузли A/B/C з відмовами)
-> репліки, що вижили -> HMAC -> decrypt -> decode -> dedup -> Q
```
