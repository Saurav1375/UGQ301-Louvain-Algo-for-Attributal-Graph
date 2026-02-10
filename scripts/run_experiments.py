#!/usr/bin/env python3
import argparse
import subprocess
import time
from pathlib import Path


def run(cmd):
    t0 = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edgelist", required=True)
    ap.add_argument("--attributes", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--a", type=float, default=0.01)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--lambda-attr", type=float, default=0.2)
    ap.add_argument("--walk-length", type=int, default=20)
    ap.add_argument("--num-walks", type=int, default=4)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--neg", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.025)
    ap.add_argument("--eval-epochs", type=int, default=30)
    ap.add_argument("--eval-runs", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    summary = []

    t = run(["make"])
    summary.append(("compile", t, "-"))

    h_base = out / "hier_base.txt"
    v_base = out / "vec_base.txt"
    t1 = run(["./recpart", args.edgelist, str(h_base), "1"])
    t2 = run(["./hi2vec", str(args.dim), str(args.a), str(h_base), str(v_base)])
    t3 = run([
        "python3", "scripts/eval_node_classification.py",
        "--vectors", str(v_base),
        "--labels", args.labels,
        "--epochs", str(args.eval_epochs),
        "--runs", str(args.eval_runs),
    ])
    summary.append(("louvainNE", t1 + t2 + t3, str(v_base)))

    h_attr = out / "hier_attr.txt"
    v_attr = out / "vec_attr.txt"
    t1 = run(["./recpart_attr", args.edgelist, str(h_attr), args.attributes, str(args.lambda_attr), "4"])
    t2 = run(["./hi2vec_attr", str(args.dim), str(args.a), str(args.beta), str(h_attr), args.attributes, str(v_attr)])
    t3 = run([
        "python3", "scripts/eval_node_classification.py",
        "--vectors", str(v_attr),
        "--labels", args.labels,
        "--epochs", str(args.eval_epochs),
        "--runs", str(args.eval_runs),
    ])
    summary.append(("attr-louvainNE", t1 + t2 + t3, str(v_attr)))

    v_dw = out / "vec_deepwalk.txt"
    t1 = run([
        "python3", "scripts/embed_baselines.py",
        "--edgelist", args.edgelist,
        "--out", str(v_dw),
        "--method", "deepwalk",
        "--dim", str(args.dim),
        "--walk-length", str(args.walk_length),
        "--num-walks", str(args.num_walks),
        "--window", str(args.window),
        "--epochs", str(args.epochs),
        "--neg", str(args.neg),
        "--lr", str(args.lr),
    ])
    t2 = run([
        "python3", "scripts/eval_node_classification.py",
        "--vectors", str(v_dw),
        "--labels", args.labels,
        "--epochs", str(args.eval_epochs),
        "--runs", str(args.eval_runs),
    ])
    summary.append(("deepwalk", t1 + t2, str(v_dw)))

    v_n2v = out / "vec_node2vec.txt"
    t1 = run([
        "python3", "scripts/embed_baselines.py",
        "--edgelist", args.edgelist,
        "--out", str(v_n2v),
        "--method", "node2vec",
        "--p", "1.0",
        "--q", "0.5",
        "--dim", str(args.dim),
        "--walk-length", str(args.walk_length),
        "--num-walks", str(args.num_walks),
        "--window", str(args.window),
        "--epochs", str(args.epochs),
        "--neg", str(args.neg),
        "--lr", str(args.lr),
    ])
    t2 = run([
        "python3", "scripts/eval_node_classification.py",
        "--vectors", str(v_n2v),
        "--labels", args.labels,
        "--epochs", str(args.eval_epochs),
        "--runs", str(args.eval_runs),
    ])
    summary.append(("node2vec", t1 + t2, str(v_n2v)))

    print("method,total_time_sec,vectors")
    for m, tt, v in summary:
        print(f"{m},{tt:.3f},{v}")


if __name__ == "__main__":
    main()
