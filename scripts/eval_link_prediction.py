#!/usr/bin/env python3
import argparse
import math


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


def read_edges(path):
    edges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v = map(int, line.split())
            edges.append((u, v))
    return edges


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cosine(a, b):
    da = math.sqrt(sum(x * x for x in a)) + 1e-12
    db = math.sqrt(sum(x * x for x in b)) + 1e-12
    return dot(a, b) / (da * db)


def edge_score(u, v, vec, metric):
    if u not in vec or v not in vec:
        return None
    if metric == "cosine":
        return cosine(vec[u], vec[v])
    return dot(vec[u], vec[v])


def auc(scores_labels):
    n = len(scores_labels)
    if n == 0:
        return 0.0
    arr = sorted(scores_labels, key=lambda x: x[0])
    n_pos = sum(lbl for _, lbl in arr)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    rank_sum_pos = 0.0
    i = 0
    rank = 1
    while i < n:
        j = i + 1
        while j < n and arr[j][0] == arr[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        pos_in_tie = sum(arr[k][1] for k in range(i, j))
        rank_sum_pos += avg_rank * pos_in_tie
        rank += (j - i)
        i = j

    u_stat = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u_stat / (n_pos * n_neg)


def average_precision(scores_labels):
    arr = sorted(scores_labels, key=lambda x: x[0], reverse=True)
    n_pos = sum(lbl for _, lbl in arr)
    if n_pos == 0:
        return 0.0

    tp = 0
    ap = 0.0
    for i, (_, lbl) in enumerate(arr, start=1):
        if lbl == 1:
            tp += 1
            ap += tp / i
    return ap / n_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--test-pos", required=True)
    ap.add_argument("--test-neg", required=True)
    ap.add_argument("--metric", choices=["dot", "cosine"], default="dot")
    args = ap.parse_args()

    vec = read_vectors(args.vectors)
    pos = read_edges(args.test_pos)
    neg = read_edges(args.test_neg)

    scores_labels = []
    used_pos = 0
    used_neg = 0

    for u, v in pos:
        s = edge_score(u, v, vec, args.metric)
        if s is None:
            continue
        scores_labels.append((s, 1))
        used_pos += 1

    for u, v in neg:
        s = edge_score(u, v, vec, args.metric)
        if s is None:
            continue
        scores_labels.append((s, 0))
        used_neg += 1

    auc_v = auc(scores_labels)
    ap_v = average_precision(scores_labels)

    print(f"link_auc {auc_v:.6f}")
    print(f"link_ap {ap_v:.6f}")
    print(f"n_test_pos_used {used_pos}")
    print(f"n_test_neg_used {used_neg}")


if __name__ == "__main__":
    main()
