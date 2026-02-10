#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def write_edgelist(edges, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for u, v in sorted(edges):
            f.write(f"{u} {v}\n")


def write_attributes(attrs, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for u in sorted(attrs.keys()):
            vals = attrs[u]
            f.write(str(u))
            for x in vals:
                f.write(f" {x}")
            f.write("\n")


def write_labels(labels, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for u in sorted(labels.keys()):
            f.write(f"{u} {labels[u]}\n")


def prep_blogcatalog(root: Path, out_dir: Path):
    data_dir = root / "BlogCatalog3" / "BlogCatalog-dataset" / "data"
    edges_csv = data_dir / "edges.csv"
    group_edges_csv = data_dir / "group-edges.csv"

    raw_edges = set()
    raw_nodes = set()
    with open(edges_csv, "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if len(row) < 2:
                continue
            a = int(row[0])
            b = int(row[1])
            if a == b:
                continue
            u, v = (a, b) if a < b else (b, a)
            raw_edges.add((u, v))
            raw_nodes.add(u)
            raw_nodes.add(v)

    id_map = {nid: i for i, nid in enumerate(sorted(raw_nodes))}
    edges = {(id_map[u], id_map[v]) for (u, v) in raw_edges}

    node_groups = defaultdict(set)
    all_groups = set()
    with open(group_edges_csv, "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if len(row) < 2:
                continue
            n = int(row[0])
            g = int(row[1])
            if n in id_map:
                node_groups[id_map[n]].add(g)
                all_groups.add(g)

    groups_sorted = sorted(all_groups)
    g2i = {g: i for i, g in enumerate(groups_sorted)}
    d = len(groups_sorted)

    attrs = {}
    labels = {}
    for u in range(len(id_map)):
        vec = [0] * d
        gs = sorted(node_groups.get(u, []))
        for g in gs:
            vec[g2i[g]] = 1
        attrs[u] = vec
        if gs:
            labels[u] = g2i[gs[0]]

    ds_dir = out_dir / "blogcatalog"
    ds_dir.mkdir(parents=True, exist_ok=True)
    write_edgelist(edges, ds_dir / "edgelist.txt")
    write_attributes(attrs, ds_dir / "attributes.txt")
    write_labels(labels, ds_dir / "labels.txt")


def parse_cite_line(line):
    line = line.strip()
    if not line.startswith("Cite(") or ")=" not in line:
        return None
    body = line[len("Cite("):line.index(")=")]
    a, b = body.split(",")
    return int(a), int(b)


def parse_topic_line(line):
    line = line.strip()
    if not line or ")=" not in line or "(" not in line:
        return None
    left = line[:line.index("(")]
    body = line[line.index("(") + 1: line.index(")=")]
    return int(body), left


def parse_paper_features(line):
    line = line.strip()
    if not line:
        return None
    parts = line.split(";")
    if len(parts) < 3:
        return None
    pid = int(parts[0])
    feats = parts[2].split(",") if parts[2] else []
    pairs = []
    for t in feats:
        if not t:
            continue
        k, v = t.split(":")
        pairs.append((k, float(v)))
    return pid, pairs


def prep_cora(root: Path, out_dir: Path, hash_dim: int):
    cora_dir = root / "CoRA_Raw"
    citations = cora_dir / "citations.txt"
    topics = cora_dir / "topics.txt"
    papers = cora_dir / "papers_dataset.txt"

    raw_edges = set()
    nodes_in_edges = set()
    with open(citations, "r", encoding="utf-8") as f:
        for line in f:
            p = parse_cite_line(line)
            if p is None:
                continue
            a, b = p
            if a == b:
                continue
            u, v = (a, b) if a < b else (b, a)
            raw_edges.add((u, v))
            nodes_in_edges.add(u)
            nodes_in_edges.add(v)

    id_map = {nid: i for i, nid in enumerate(sorted(nodes_in_edges))}
    edges = {(id_map[u], id_map[v]) for (u, v) in raw_edges}

    topic2i = {}
    labels = {}
    with open(topics, "r", encoding="utf-8") as f:
        for line in f:
            p = parse_topic_line(line)
            if p is None:
                continue
            pid, topic = p
            if pid not in id_map:
                continue
            if topic not in topic2i:
                topic2i[topic] = len(topic2i)
            labels[id_map[pid]] = topic2i[topic]

    attrs = {u: [0.0] * hash_dim for u in range(len(id_map))}
    with open(papers, "r", encoding="utf-8") as f:
        for line in f:
            p = parse_paper_features(line)
            if p is None:
                continue
            pid, feats = p
            if pid not in id_map:
                continue
            u = id_map[pid]
            vec = attrs[u]
            for key, val in feats:
                h = hash(key) % hash_dim
                vec[h] += val

    ds_dir = out_dir / "cora"
    ds_dir.mkdir(parents=True, exist_ok=True)
    write_edgelist(edges, ds_dir / "edgelist.txt")
    write_attributes(attrs, ds_dir / "attributes.txt")
    write_labels(labels, ds_dir / "labels.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--outdir", default="data/real")
    ap.add_argument("--cora-hash-dim", type=int, default=256)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    prep_blogcatalog(root, out)
    prep_cora(root, out, args.cora_hash_dim)

    print(f"Prepared datasets in {out}")


if __name__ == "__main__":
    main()
