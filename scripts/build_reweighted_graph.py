#!/usr/bin/env python3
import argparse
import heapq
import math
from collections import defaultdict


def read_graph(path):
    edges = []
    edge_set = set()
    nodes = set()
    neighbors = defaultdict(set)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            v = int(parts[1])
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in edge_set:
                continue
            edge_set.add((a, b))
            edges.append((a, b))
            neighbors[a].add(b)
            neighbors[b].add(a)
            nodes.add(a)
            nodes.add(b)

    return sorted(nodes), edges, edge_set, neighbors


def read_attributes(path):
    sparse = {}
    norms = {}
    inverted = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            node = int(parts[0])
            feats = []
            norm_sq = 0.0
            for idx, raw in enumerate(parts[1:]):
                val = float(raw)
                if val == 0.0:
                    continue
                feats.append((idx, val))
                norm_sq += val * val
            sparse[node] = feats
            norms[node] = math.sqrt(norm_sq)
            for idx, val in feats:
                inverted[idx].append((node, val))

    return sparse, norms, inverted


def dot_sparse(feats_a, feats_b):
    if len(feats_a) > len(feats_b):
        feats_a, feats_b = feats_b, feats_a
    map_b = dict(feats_b)
    res = 0.0
    for idx, val in feats_a:
        res += val * map_b.get(idx, 0.0)
    return res


def similarity01(u, v, sparse, norms):
    feats_u = sparse.get(u, [])
    feats_v = sparse.get(v, [])
    nu = norms.get(u, 0.0)
    nv = norms.get(v, 0.0)
    if nu == 0.0 or nv == 0.0:
        return 0.0
    cos = dot_sparse(feats_u, feats_v) / (nu * nv)
    if cos < -1.0:
        cos = -1.0
    if cos > 1.0:
        cos = 1.0
    return 0.5 * (cos + 1.0)


def reweight_existing(edges, sparse, norms, alpha):
    weighted = {}
    for u, v in edges:
        sim = similarity01(u, v, sparse, norms)
        weighted[(u, v)] = alpha + (1.0 - alpha) * sim
    return weighted


def augment_edges(nodes, neighbors, edge_set, sparse, norms, inverted, alpha, k):
    weighted = {}
    for u in nodes:
        accum = defaultdict(float)
        feats_u = sparse.get(u, [])
        nu = norms.get(u, 0.0)
        if nu == 0.0:
            continue

        for dim, val_u in feats_u:
            for v, val_v in inverted[dim]:
                if v == u:
                    continue
                accum[v] += val_u * val_v

        topk = []
        for v, dot in accum.items():
            if v in neighbors[u]:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in edge_set:
                continue
            nv = norms.get(v, 0.0)
            if nv == 0.0:
                continue
            cos = dot / (nu * nv)
            if cos < -1.0:
                cos = -1.0
            if cos > 1.0:
                cos = 1.0
            sim = 0.5 * (cos + 1.0)
            if sim <= 0.0:
                continue
            heapq.heappush(topk, (sim, a, b))
            if len(topk) > k:
                heapq.heappop(topk)

        for sim, a, b in topk:
            cur = weighted.get((a, b), -1.0)
            w = (1.0 - alpha) * sim
            if w > cur:
                weighted[(a, b)] = w

    return weighted


def write_weighted(path, weighted):
    with open(path, "w", encoding="utf-8") as f:
        for u, v in sorted(weighted.keys()):
            f.write(f"{u} {v} {weighted[(u, v)]:.10f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edgelist", required=True)
    ap.add_argument("--attributes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["preserve", "augment"], default="preserve")
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--knn-k", type=int, default=5)
    args = ap.parse_args()

    if args.alpha < 0.0 or args.alpha > 1.0:
        raise SystemExit("alpha must be in [0, 1]")
    if args.mode == "augment" and args.knn_k <= 0:
        raise SystemExit("knn-k must be positive in augment mode")

    nodes, edges, edge_set, neighbors = read_graph(args.edgelist)
    sparse, norms, inverted = read_attributes(args.attributes)

    weighted = reweight_existing(edges, sparse, norms, args.alpha)

    added = 0
    if args.mode == "augment":
        extra = augment_edges(nodes, neighbors, edge_set, sparse, norms, inverted, args.alpha, args.knn_k)
        added = len(extra)
        weighted.update(extra)

    write_weighted(args.out, weighted)

    print(f"mode {args.mode}")
    print(f"n_nodes {len(nodes)}")
    print(f"n_edges_input {len(edges)}")
    print(f"n_edges_output {len(weighted)}")
    print(f"n_edges_added {added}")


if __name__ == "__main__":
    main()
