# Зона 1, просунутий рівень. Адаптивний Гаффман (алгоритм FGK).
#
# Канонічний Гаффман двопрохідний: спершу рахує частоти всього блоку, потім
# кодує, і мусить передати таблицю довжин (2 байти на символ).
# Адаптивний - однопрохідний: кодер і декодер однаково оновлюють дерево
# після кожного символу, тому таблиця не передається взагалі.
#
# Нового символу ще немає в дереві: передаємо код спеціального листка NYT
# ("not yet transmitted") і 8 бітів самого символу.
#
# Властивість братства (sibling property): якщо пронумерувати вузли order
# знизу вгору, ваги не спадають. Тому перед збільшенням ваги вузол міняється
# місцем з "лідером" - вузлом з найбільшим order серед вузлів тієї самої ваги.
#
# Це командне розширення для експерименту, не частина формату LOG3.

MAX_ORDER = 512          # 256 листків + 255 внутрішніх + NYT


class Node:
    def __init__(self, order, parent=None, symbol=None):
        self.weight = 0
        self.order = order
        self.parent = parent
        self.left = None
        self.right = None
        self.symbol = symbol


class Tree:
    def __init__(self):
        self.root = Node(MAX_ORDER)
        self.nyt = self.root
        self.leaf = {}                       # символ -> листок
        self.by_order = {MAX_ORDER: self.root}

    def code_of(self, node):
        # шлях від кореня: ліво = 0, право = 1
        bits = []
        while node.parent is not None:
            bits.append(0 if node.parent.left is node else 1)
            node = node.parent
        bits.reverse()
        return bits

    def swap(self, a, b):
        # міняємо місцями два піддерева разом з їхніми номерами order
        pa, pb = a.parent, b.parent
        a_left = pa.left is a
        b_left = pb.left is b
        if a_left:
            pa.left = b
        else:
            pa.right = b
        if b_left:
            pb.left = a
        else:
            pb.right = a
        a.parent, b.parent = pb, pa
        a.order, b.order = b.order, a.order
        self.by_order[a.order] = a
        self.by_order[b.order] = b

    def update(self, sym):
        if sym not in self.leaf:
            # NYT ділиться на новий NYT (ліворуч) і листок символу (праворуч)
            old = self.nyt
            new_nyt = Node(old.order - 2, old)
            new_leaf = Node(old.order - 1, old, sym)
            old.left, old.right = new_nyt, new_leaf
            self.by_order[new_nyt.order] = new_nyt
            self.by_order[new_leaf.order] = new_leaf
            self.nyt = new_nyt
            self.leaf[sym] = new_leaf
            node = new_leaf
        else:
            node = self.leaf[sym]
        while node is not None:
            # вузли однієї ваги йдуть підряд за order, тому лідера шукаємо вгору
            leader = node
            o = node.order + 1
            while o in self.by_order and self.by_order[o].weight == node.weight:
                leader = self.by_order[o]
                o += 1
            if leader is not node and leader is not node.parent:
                self.swap(node, leader)
            node.weight += 1
            node = node.parent


def encode(data):
    tree = Tree()
    bits = []
    for b in data:
        if b in tree.leaf:
            bits += tree.code_of(tree.leaf[b])
        else:
            bits += tree.code_of(tree.nyt)
            for k in range(7, -1, -1):
                bits.append((b >> k) & 1)
        tree.update(b)
    out = bytearray((len(bits) + 7) // 8)   # нульовий padding у кінці
    for i in range(len(bits)):
        if bits[i]:
            out[i // 8] |= 0x80 >> (i % 8)
    return bytes(out), len(bits)


def decode(stream, symbol_count):
    tree = Tree()
    out = bytearray()
    pos = 0

    def bit():
        nonlocal pos
        if pos >= len(stream) * 8:
            raise ValueError("bitstream ended too early")
        v = (stream[pos // 8] >> (7 - pos % 8)) & 1
        pos += 1
        return v

    while len(out) < symbol_count:
        node = tree.root
        while node.left is not None:          # спускаємось до листка
            node = node.right if bit() else node.left
        if node is tree.nyt:
            sym = 0
            for _ in range(8):
                sym = (sym << 1) | bit()
        else:
            sym = node.symbol
        out.append(sym)
        tree.update(sym)
    return bytes(out)
