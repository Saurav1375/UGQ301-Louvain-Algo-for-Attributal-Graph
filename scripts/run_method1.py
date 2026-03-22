#!/usr/bin/env python3
import argparse
import csv
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(cmd, cwd=None):
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=cwd)
    return time.time() - t0


def run_capture(cmd, cwd=None):
    out = subprocess.check_output(cmd, text=True, cwd=cwd)
    return out


def parse_metrics(output):
    metrics = {}
    for line in output.strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        key, value = parts
        try:
            if "." in value or "e" in value.lower():
                metrics[key] = float(value)
            else:
                metrics[key] = int(value)
        except ValueError:
            continue
    return metrics


def evaluate_node(vectors, labels, epochs, runs):
    out = run_capture([
        "python3", "scripts/eval_node_classification.py",
        "--vectors", str(vectors),
        "--labels", labels,
        "--epochs", str(epochs),
        "--runs", str(runs),
    ], cwd=REPO_ROOT)
    metrics = parse_metrics(out)
    return float(metrics.get("accuracy_mean", 0.0)), float(metrics.get("accuracy_std", 0.0))


def evaluate_link(vectors, pos, neg, metric):
    out = run_capture([
        "python3", "scripts/eval_link_prediction.py",
        "--vectors", str(vectors),
        "--test-pos", str(pos),
        "--test-neg", str(neg),
        "--metric", metric,
    ], cwd=REPO_ROOT)
    metrics = parse_metrics(out)
    return float(metrics.get("link_auc", 0.0)), float(metrics.get("link_ap", 0.0))


def maybe_prepare_link_split(edgelist, outdir, test_ratio, neg_mult, seed):
    train = outdir / "lp_train.edgelist"
    pos = outdir / "lp_test_pos.txt"
    neg = outdir / "lp_test_neg.txt"
    run([
        "python3", "scripts/split_link_prediction.py",
        "--edgelist", edgelist,
        "--train-out", str(train),
        "--test-pos-out", str(pos),
        "--test-neg-out", str(neg),
        "--test-ratio", str(test_ratio),
        "--neg-mult", str(neg_mult),
        "--seed", str(seed),
    ], cwd=REPO_ROOT)
    return train, pos, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-name", default="dataset")
    ap.add_argument("--edgelist", required=True)
    ap.add_argument("--attributes", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mode", choices=["preserve", "augment"], default="preserve")
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--a", type=float, default=0.01)
    ap.add_argument("--eval-epochs", type=int, default=10)
    ap.add_argument("--eval-runs", type=int, default=3)
    ap.add_argument("--with-link-pred", action="store_true")
    ap.add_argument("--lp-test-ratio", type=float, default=0.1)
    ap.add_argument("--lp-neg-mult", type=float, default=1.0)
    ap.add_argument("--lp-seed", type=int, default=42)
    ap.add_argument("--lp-metric", choices=["dot", "cosine"], default="dot")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_graph = args.edgelist
    test_pos = None
    test_neg = None
    if args.with_link_pred:
        train_graph, test_pos, test_neg = maybe_prepare_link_split(
            args.edgelist, outdir, args.lp_test_ratio, args.lp_neg_mult, args.lp_seed
        )

    run(["make", "recpart_weighted", "hi2vec"], cwd=REPO_ROOT)

    weighted_graph = outdir / "weighted_graph.txt"
    hierarchy = outdir / "hierarchy.txt"
    vectors = outdir / "vectors.txt"

    method = f"method1-{args.mode}"

    t0 = time.time()
    run([
        "python3", "scripts/build_reweighted_graph.py",
        "--edgelist", str(train_graph),
        "--attributes", args.attributes,
        "--out", str(weighted_graph),
        "--mode", args.mode,
        "--alpha", str(args.alpha),
        "--knn-k", str(args.knn_k),
    ], cwd=REPO_ROOT)
    t_weight = time.time() - t0

    t1 = time.time()
    run(["./recpart_weighted", str(weighted_graph), str(hierarchy), "1"], cwd=REPO_ROOT)
    run(["./hi2vec", str(args.dim), str(args.a), str(hierarchy), str(vectors)], cwd=REPO_ROOT)
    t_embed = time.time() - t1

    t2 = time.time()
    acc_mean, acc_std = evaluate_node(vectors, args.labels, args.eval_epochs, args.eval_runs)
    link_auc = ""
    link_ap = ""
    if args.with_link_pred:
        auc, ap_score = evaluate_link(vectors, test_pos, test_neg, args.lp_metric)
        link_auc = f"{auc:.6f}"
        link_ap = f"{ap_score:.6f}"
    t_eval = time.time() - t2

    row = {
        "dataset": args.dataset_name,
        "method": method,
        "alpha": f"{args.alpha:.6f}",
        "knn_k": args.knn_k,
        "weight_build_time_sec": f"{t_weight:.6f}",
        "embed_time_sec": f"{t_embed:.6f}",
        "eval_time_sec": f"{t_eval:.6f}",
        "total_time_sec": f"{(t_weight + t_embed + t_eval):.6f}",
        "node_accuracy_mean": f"{acc_mean:.6f}",
        "node_accuracy_std": f"{acc_std:.6f}",
        "link_auc": link_auc,
        "link_ap": link_ap,
        "weighted_graph": str(weighted_graph),
        "vectors": str(vectors),
    }

    out_csv = outdir / "results.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(f"results_csv {out_csv}")
    print(f"dataset {args.dataset_name}")
    print(f"method {method}")
    print(f"node_accuracy_mean {acc_mean:.6f}")
    if link_auc:
        print(f"link_auc {link_auc}")


if __name__ == "__main__":
    main()
