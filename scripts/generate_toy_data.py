#!/usr/bin/env python3
import random

random.seed(42)
n = 200
p_in = 0.08
p_out = 0.01

with open("data/toy.edgelist", "w", encoding="utf-8") as fe:
    for i in range(n):
        for j in range(i + 1, n):
            same = (i < n // 2 and j < n // 2) or (i >= n // 2 and j >= n // 2)
            p = p_in if same else p_out
            if random.random() < p:
                fe.write(f"{i} {j}\n")

with open("data/toy.attrs", "w", encoding="utf-8") as fa, open("data/toy.labels", "w", encoding="utf-8") as fl:
    for i in range(n):
        c = 0 if i < n // 2 else 1
        x1 = random.gauss(1.0 if c == 0 else -1.0, 0.6)
        x2 = random.gauss(1.0 if c == 1 else -1.0, 0.6)
        x3 = random.gauss(0.0, 1.0)
        fa.write(f"{i} {x1:.6f} {x2:.6f} {x3:.6f}\n")
        fl.write(f"{i} {c}\n")
