#!/usr/bin/env python3
import argparse
import math
import random
from collections import defaultdict


def read_graph(path):
    adj = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v = map(int, line.split())
            adj[u].append(v)
            adj[v].append(u)
    return sorted(adj.keys()), adj


def weighted_choice(items, weights):
    s = sum(weights)
    if s <= 0:
        return random.choice(items)
    r = random.random() * s
    cur = 0.0
    for it, w in zip(items, weights):
        cur += w
        if cur >= r:
            return it
    return items[-1]


def node2vec_next(prev, cur, adj, nbr_set, p, q):
    nbrs = adj[cur]
    weights = []
    for x in nbrs:
        if x == prev:
            weights.append(1.0 / p)
        elif x in nbr_set[prev]:
            weights.append(1.0)
        else:
            weights.append(1.0 / q)
    return weighted_choice(nbrs, weights)


def generate_walks(nodes, adj, walk_length, num_walks, method, p, q):
    walks = []
    nbr_set = {u: set(adj[u]) for u in nodes}
    for _ in range(num_walks):
        order = nodes[:]
        random.shuffle(order)
        for start in order:
            walk = [start]
            while len(walk) < walk_length:
                cur = walk[-1]
                if not adj[cur]:
                    break
                if len(walk) == 1 or method == "deepwalk":
                    nxt = random.choice(adj[cur])
                else:
                    nxt = node2vec_next(walk[-2], cur, adj, nbr_set, p, q)
                walk.append(nxt)
            walks.append(walk)
    return walks


def build_pairs(walks, window, node2idx):
    pairs = []
    for w in walks:
        ids = [node2idx[u] for u in w]
        for i, c in enumerate(ids):
            l = max(0, i - window)
            r = min(len(ids), i + window + 1)
            for j in range(l, r):
                if j != i:
                    pairs.append((c, ids[j]))
    return pairs


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def add_scaled(dst, src, s):
    for i in range(len(dst)):
        dst[i] += s * src[i]


def sigmoid(x):
    if x > 8:
        return 0.9997
    if x < -8:
        return 0.0003
    return 1.0 / (1.0 + math.exp(-x))


def build_neg_table(freq, size=200000):
    power = 0.75
    items = list(freq.items())
    z = sum((c ** power) for _, c in items)
    table = []
    for i, c in items:
        cnt = max(1, int((c ** power) / z * size))
        table.extend([i] * cnt)
    random.shuffle(table)
    return table


def train_skipgram(walks, dim, window, epochs, neg_k, lr, seed):
    random.seed(seed)
    nodes = sorted({u for w in walks for u in w})
    node2idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)

    freq = defaultdict(int)
    for w in walks:
        for u in w:
            freq[node2idx[u]] += 1

    W_in = [[(random.random() - 0.5) / dim for _ in range(dim)] for _ in range(n)]
    W_out = [[0.0 for _ in range(dim)] for _ in range(n)]

    pairs = build_pairs(walks, window, node2idx)
    neg_table = build_neg_table(freq)

    for ep in range(epochs):
        random.shuffle(pairs)
        alpha = lr * (1.0 - ep / max(1, epochs))

        for c, o in pairs:
            vc = W_in[c][:]

            s = dot(vc, W_out[o])
            g = (1.0 - sigmoid(s)) * alpha
            add_scaled(W_in[c], W_out[o], g)
            add_scaled(W_out[o], vc, g)

            for _ in range(neg_k):
                ni = random.choice(neg_table)
                if ni == o:
                    continue
                s2 = dot(vc, W_out[ni])
                g2 = (0.0 - sigmoid(s2)) * alpha
                add_scaled(W_in[c], W_out[ni], g2)
                add_scaled(W_out[ni], vc, g2)

    emb = []
    for i in range(n):
        v = [W_in[i][j] + W_out[i][j] for j in range(dim)]
        norm = math.sqrt(sum(x * x for x in v)) + 1e-12
        emb.append([x / norm for x in v])
    return nodes, emb


def save_vectors(path, nodes, emb):
    with open(path, "w", encoding="utf-8") as f:
        for u, v in zip(nodes, emb):
            f.write(str(u))
            for x in v:
                f.write(f" {x:.8f}")
            f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edgelist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["deepwalk", "node2vec"], required=True)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--walk-length", type=int, default=40)
    ap.add_argument("--num-walks", type=int, default=10)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--neg", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.025)
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    nodes, adj = read_graph(args.edgelist)
    walks = generate_walks(nodes, adj, args.walk_length, args.num_walks, args.method, args.p, args.q)
    nodes_out, emb = train_skipgram(walks, args.dim, args.window, args.epochs, args.neg, args.lr, args.seed)
    save_vectors(args.out, nodes_out, emb)


if __name__ == "__main__":
    main()
