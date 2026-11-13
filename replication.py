#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replication.py — еталонна модель реплікації для Проєкту 3.

ЦЕЙ ФАЙЛ ВИДАЄТЬСЯ ГОТОВИМ І НЕ Є ПРЕДМЕТОМ РЕАЛІЗАЦІЇ.
Змінювати його не можна: результати команд перестануть бути порівнянними.
Опис моделі — REPLICATION_SPEC.md.

Вузли A, B, C. Для кожного вузла — незалежний двостанний процес GOOD/BAD
і ОКРЕМИЙ детермінований генератор випадкових чисел, похідний від
загального seed. Тому зміна кількості операцій на одному вузлі не змінює
послідовність відмов іншого.

Перехід у BAD на поточній операції означає, що саме ця операція є першою
невдалою. Імовірність переходу:

    p_gb = loss / (loss + B * (1 - loss))

де B — середня довжина пачки відмов. Довжина пачки геометрична з
середнім B.

Модель НЕ приймає ідентифікатори записів і не довіряє заявленому складу
репліки. Фактичний склад визначає evaluator, розбираючи збережені blob.
"""

import hashlib
import random

NODES = ("A", "B", "C")


def _node_seed(seed, node):
    h = hashlib.sha256(("%d:%s" % (seed, node)).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


class _Node:
    def __init__(self, cfg, seed, name):
        self.name = name
        self.capacity = int(cfg["capacity_bytes"])
        self.loss = float(cfg["failure_probability"])
        self.B = float(cfg["mean_failure_burst"])
        self.rng = random.Random(_node_seed(seed, name))
        self.p_gb = (self.loss / (self.loss + self.B * (1.0 - self.loss))
                     if self.loss > 0 else 0.0)
        self.in_burst = 0
        self.used = 0
        self.attempts = 0
        self.stored = 0
        self.failed = 0
        self.rejected = 0
        self.blobs = []

    def next_is_bad(self):
        if self.in_burst > 0:
            self.in_burst -= 1
            return True
        if self.loss > 0 and self.rng.random() < self.p_gb:
            length = 1
            if self.B > 1.0:
                while self.rng.random() > 1.0 / self.B:
                    length += 1
            self.in_burst = length - 1
            return True
        return False


class Replication:
    def __init__(self, scenario, seed):
        """scenario — словник зі секції "реплікація"->"сценарії" конфігурації."""
        self.global_budget = int(scenario["global_replication_budget_bytes"])
        self.global_used = 0
        self.global_rejected = 0
        self.nodes = {n: _Node(scenario["вузли"][n], seed, n) for n in NODES}

    def store(self, node, blob):
        """Спроба записати зашифрований blob на вузол.

        Повертає True, якщо копію збережено; False — якщо операцію
        відхилено через бюджет (байти не списуються) або копію втрачено
        через стан BAD (байти списуються).
        """
        nd = self.nodes[node]
        size = len(blob)
        if (self.global_used + size > self.global_budget
                or nd.used + size > nd.capacity):
            nd.rejected += 1
            self.global_rejected += 1
            return False
        nd.used += size
        self.global_used += size
        nd.attempts += 1
        if nd.next_is_bad():
            nd.failed += 1
            return False
        nd.stored += 1
        nd.blobs.append(bytes(blob))
        return True

    def surviving_blobs_by_node(self):
        return {n: list(self.nodes[n].blobs) for n in NODES}

    def surviving_blobs(self):
        out = []
        for n in NODES:
            out.extend(self.nodes[n].blobs)
        return out

    def stats(self):
        return {
            "global_used_bytes": self.global_used,
            "global_budget_bytes": self.global_budget,
            "global_rejected": self.global_rejected,
            "вузли": {n: {"used_bytes": nd.used, "capacity_bytes": nd.capacity,
                          "attempts": nd.attempts, "stored": nd.stored,
                          "failed": nd.failed, "rejected": nd.rejected}
                      for n, nd in self.nodes.items()},
        }
