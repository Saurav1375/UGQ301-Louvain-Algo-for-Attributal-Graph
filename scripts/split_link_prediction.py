#!/usr/bin/env python3
import argparse
import random
from pathlib import Path


def read_edges(path):
    edges = set()
    nodes = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v = map(int, line.split())
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            edges.add((a, b))
            nodes.add(a)
            nodes.add(b)
    return sorted(nodes), list(edges)


def split_edges(nodes, edges, test_ratio, seed):
    rnd = random.Random(seed)
    deg = {u: 0 for u in nodes}
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    target_test = int(len(edges) * test_ratio)
    rnd.shuffle(edges)

    test = []
    train = []
    for u, v in edges:
        if len(test) < target_test and deg[u] > 1 and deg[v] > 1:
            test.append((u, v))
            deg[u] -= 1
            deg[v] -= 1
        else:
            train.append((u, v))

    return train, test


def sample_negatives(nodes, edge_set, n_samples, seed):
    rnd = random.Random(seed + 1337)
    node_list = list(nodes)
    n = len(node_list)
    neg = set()

    while len(neg) < n_samples:
        u = node_list[rnd.randrange(n)]
        v = node_list[rnd.randrange(n)]
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in edge_set or (a, b) in neg:
            continue
        neg.add((a, b))

    return list(neg)


def write_edges(path, edges):
    with open(path, "w", encoding="utf-8") as f:
        for u, v in edges:
            f.write(f"{u} {v}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edgelist", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--test-pos-out", required=True)
    ap.add_argument("--test-neg-out", required=True)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--neg-mult", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    nodes, edges = read_edges(args.edgelist)
    train, test_pos = split_edges(nodes, edges, args.test_ratio, args.seed)

    train_set = set(train)
    full_set = set(edges)
    n_neg = int(len(test_pos) * args.neg_mult)
    test_neg = sample_negatives(nodes, full_set, n_neg, args.seed)

    Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
    write_edges(args.train_out, train)
    write_edges(args.test_pos_out, test_pos)
    write_edges(args.test_neg_out, test_neg)

    print(f"n_nodes {len(nodes)}")
    print(f"n_edges_full {len(edges)}")
    print(f"n_edges_train {len(train)}")
    print(f"n_edges_test_pos {len(test_pos)}")
    print(f"n_edges_test_neg {len(test_neg)}")


if __name__ == "__main__":
    main()
