#!/usr/bin/env python3
import argparse
import csv
import subprocess
import time
from pathlib import Path


def run(cmd):
    t0 = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - t0


def run_capture(cmd):
    t0 = time.time()
    out = subprocess.check_output(cmd, text=True)
    return time.time() - t0, out


def parse_metrics(output):
    m = {}
    for line in output.strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        k, v = parts
        try:
            if "." in v or "e" in v.lower():
                m[k] = float(v)
            else:
                m[k] = int(v)
        except ValueError:
            continue
    return m


def evaluate_node(vectors, labels, eval_epochs, eval_runs):
    _, out = run_capture([
        "python3", "scripts/eval_node_classification.py",
        "--vectors", str(vectors),
        "--labels", labels,
        "--epochs", str(eval_epochs),
        "--runs", str(eval_runs),
    ])
    met = parse_metrics(out)
    return float(met.get("accuracy_mean", 0.0)), float(met.get("accuracy_std", 0.0))


def evaluate_link(vectors, test_pos, test_neg, metric):
    _, out = run_capture([
        "python3", "scripts/eval_link_prediction.py",
        "--vectors", str(vectors),
        "--test-pos", str(test_pos),
        "--test-neg", str(test_neg),
        "--metric", metric,
    ])
    met = parse_metrics(out)
    return float(met.get("link_auc", 0.0)), float(met.get("link_ap", 0.0))


def maybe_prepare_link_split(edgelist, outdir, test_ratio, neg_mult, seed):
    train = outdir / "lp_train.edgelist"
    test_pos = outdir / "lp_test_pos.txt"
    test_neg = outdir / "lp_test_neg.txt"

    run([
        "python3", "scripts/split_link_prediction.py",
        "--edgelist", edgelist,
        "--train-out", str(train),
        "--test-pos-out", str(test_pos),
        "--test-neg-out", str(test_neg),
        "--test-ratio", str(test_ratio),
        "--neg-mult", str(neg_mult),
        "--seed", str(seed),
    ])

    return train, test_pos, test_neg


def append_global_csv(global_csv, rows):
    global_csv = Path(global_csv)
    global_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = global_csv.exists()
    fields = list(rows[0].keys())

    with open(global_csv, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-name", default="dataset")
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

    ap.add_argument("--with-link-pred", action="store_true")
    ap.add_argument("--lp-test-ratio", type=float, default=0.1)
    ap.add_argument("--lp-neg-mult", type=float, default=1.0)
    ap.add_argument("--lp-seed", type=int, default=42)
    ap.add_argument("--lp-metric", choices=["dot", "cosine"], default="dot")

    ap.add_argument("--global-csv", default="")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    split_train = args.edgelist
    split_pos = None
    split_neg = None
    if args.with_link_pred:
        split_train, split_pos, split_neg = maybe_prepare_link_split(
            args.edgelist,
            out,
            args.lp_test_ratio,
            args.lp_neg_mult,
            args.lp_seed,
        )

    rows = []

    t_compile = run(["make"])
    rows.append({
        "dataset": args.dataset_name,
        "method": "compile",
        "total_time_sec": f"{t_compile:.6f}",
        "embed_time_sec": f"{0.0:.6f}",
        "node_accuracy_mean": "",
        "node_accuracy_std": "",
        "link_auc": "",
        "link_ap": "",
        "vectors": "-",
    })

    def run_method(method_name, embed_cmds, vec_path):
        t_embed = 0.0
        for cmd in embed_cmds:
            t_embed += run(cmd)

        t_eval_start = time.time()
        acc_m, acc_s = evaluate_node(vec_path, args.labels, args.eval_epochs, args.eval_runs)
        link_auc = ""
        link_ap = ""
        if args.with_link_pred:
            auc_v, ap_v = evaluate_link(vec_path, split_pos, split_neg, args.lp_metric)
            link_auc = f"{auc_v:.6f}"
            link_ap = f"{ap_v:.6f}"
        t_eval = time.time() - t_eval_start

        total = t_embed + t_eval
        rows.append({
            "dataset": args.dataset_name,
            "method": method_name,
            "total_time_sec": f"{total:.6f}",
            "embed_time_sec": f"{t_embed:.6f}",
            "node_accuracy_mean": f"{acc_m:.6f}",
            "node_accuracy_std": f"{acc_s:.6f}",
            "link_auc": link_auc,
            "link_ap": link_ap,
            "vectors": str(vec_path),
        })

    h_base = out / "hier_base.txt"
    v_base = out / "vec_base.txt"
    run_method(
        "louvainNE",
        [
            ["./recpart", str(split_train), str(h_base), "1"],
            ["./hi2vec", str(args.dim), str(args.a), str(h_base), str(v_base)],
        ],
        v_base,
    )

    h_attr = out / "hier_attr.txt"
    v_attr = out / "vec_attr.txt"
    run_method(
        "attr-louvainNE",
        [
            ["./recpart_attr", str(split_train), str(h_attr), args.attributes, str(args.lambda_attr), "4"],
            ["./hi2vec_attr", str(args.dim), str(args.a), str(args.beta), str(h_attr), args.attributes, str(v_attr)],
        ],
        v_attr,
    )

    v_dw = out / "vec_deepwalk.txt"
    run_method(
        "deepwalk",
        [[
            "python3", "scripts/embed_baselines.py",
            "--edgelist", str(split_train),
            "--out", str(v_dw),
            "--method", "deepwalk",
            "--dim", str(args.dim),
            "--walk-length", str(args.walk_length),
            "--num-walks", str(args.num_walks),
            "--window", str(args.window),
            "--epochs", str(args.epochs),
            "--neg", str(args.neg),
            "--lr", str(args.lr),
        ]],
        v_dw,
    )

    v_n2v = out / "vec_node2vec.txt"
    run_method(
        "node2vec",
        [[
            "python3", "scripts/embed_baselines.py",
            "--edgelist", str(split_train),
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
        ]],
        v_n2v,
    )

    out_csv = out / "results.csv"
    fields = list(rows[0].keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    if args.global_csv:
        append_global_csv(args.global_csv, rows)

    print(f"results_csv {out_csv}")
    print("dataset,method,total_time_sec,embed_time_sec,node_accuracy_mean,link_auc")
    for r in rows:
        print(
            f"{r['dataset']},{r['method']},{r['total_time_sec']},"
            f"{r['embed_time_sec']},{r['node_accuracy_mean']},{r['link_auc']}"
        )


if __name__ == "__main__":
    main()
