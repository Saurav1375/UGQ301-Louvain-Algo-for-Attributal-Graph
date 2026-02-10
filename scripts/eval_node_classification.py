#!/usr/bin/env python3
import argparse
import math
import random


def read_vectors(path):
    vec = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            vec[u] = [float(x) for x in parts[1:]]
    return vec


def read_labels(path):
    y = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, l = line.split()
            y[int(u)] = int(l)
    return y


def softmax(logits):
    m = max(logits)
    ex = [math.exp(z - m) for z in logits]
    s = sum(ex) + 1e-12
    return [v / s for v in ex]


def matvec(x, W, b):
    c = len(b)
    out = [b[j] for j in range(c)]
    for i, xi in enumerate(x):
        row = W[i]
        for j in range(c):
            out[j] += xi * row[j]
    return out


def train_linear(X, y, ncls, epochs=60, lr=0.1, reg=1e-4):
    n = len(X)
    d = len(X[0])
    W = [[0.0 for _ in range(ncls)] for _ in range(d)]
    b = [0.0 for _ in range(ncls)]

    for _ in range(epochs):
        dW = [[0.0 for _ in range(ncls)] for _ in range(d)]
        db = [0.0 for _ in range(ncls)]

        for i in range(n):
            p = softmax(matvec(X[i], W, b))
            p[y[i]] -= 1.0
            for c in range(ncls):
                db[c] += p[c]
            for k, xk in enumerate(X[i]):
                row = dW[k]
                for c in range(ncls):
                    row[c] += xk * p[c]

        inv_n = 1.0 / max(1, n)
        for k in range(d):
            for c in range(ncls):
                dW[k][c] = dW[k][c] * inv_n + reg * W[k][c]
                W[k][c] -= lr * dW[k][c]
        for c in range(ncls):
            b[c] -= lr * db[c] * inv_n

    return W, b


def predict(x, W, b):
    z = matvec(x, W, b)
    best_c = 0
    best_v = z[0]
    for c in range(1, len(z)):
        if z[c] > best_v:
            best_v = z[c]
            best_c = c
    return best_c


def accuracy(X, y, W, b):
    hit = 0
    for xi, yi in zip(X, y):
        if predict(xi, W, b) == yi:
            hit += 1
    return hit / max(1, len(y))


def mean_std(vals):
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / len(vals)
    return m, math.sqrt(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--train-ratio", type=float, default=0.5)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.1)
    args = ap.parse_args()

    vec = read_vectors(args.vectors)
    lab = read_labels(args.labels)

    ids = sorted(set(vec.keys()) & set(lab.keys()))
    X = [vec[u] for u in ids]
    y_raw = [lab[u] for u in ids]

    uniq = sorted(set(y_raw))
    mp = {c: i for i, c in enumerate(uniq)}
    y = [mp[c] for c in y_raw]
    ncls = len(uniq)

    random.seed(args.seed)
    idx = list(range(len(ids)))
    scores = []

    for _ in range(args.runs):
        random.shuffle(idx)
        tr = int(len(idx) * args.train_ratio)
        tr_idx = idx[:tr]
        te_idx = idx[tr:]

        Xtr = [X[i] for i in tr_idx]
        ytr = [y[i] for i in tr_idx]
        Xte = [X[i] for i in te_idx]
        yte = [y[i] for i in te_idx]

        W, b = train_linear(Xtr, ytr, ncls, epochs=args.epochs, lr=args.lr)
        scores.append(accuracy(Xte, yte, W, b))

    m, s = mean_std(scores)
    print(f"accuracy_mean {m:.6f}")
    print(f"accuracy_std {s:.6f}")
    print(f"n_eval_nodes {len(ids)}")


if __name__ == "__main__":
    main()
