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
    out = subprocess.check_output(cmd, text=True)
    return out


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


def evaluate_node(vec, labels, eval_epochs, eval_runs):
    out = run_capture([
        "python3", "scripts/eval_node_classification.py",
        "--vectors", str(vec),
        "--labels", labels,
        "--epochs", str(eval_epochs),
        "--runs", str(eval_runs),
    ])
    m = parse_metrics(out)
    return float(m.get("accuracy_mean", 0.0)), float(m.get("accuracy_std", 0.0))


def evaluate_link(vec, pos, neg, metric):
    out = run_capture([
        "python3", "scripts/eval_link_prediction.py",
        "--vectors", str(vec),
        "--test-pos", str(pos),
        "--test-neg", str(neg),
        "--metric", metric,
    ])
    m = parse_metrics(out)
    return float(m.get("link_auc", 0.0)), float(m.get("link_ap", 0.0))


def parse_grid(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def write_heatmap_svg(rows, lvals, bvals, key, out_path):
    val = {(float(r["lambda"]), float(r["beta"])): float(r[key]) for r in rows}
    if not val:
        return
    vmin = min(val.values())
    vmax = max(val.values())

    w = 70 + 70 * len(bvals)
    h = 70 + 50 * len(lvals)

    def color(x):
        if vmax <= vmin:
            t = 0.5
        else:
            t = (x - vmin) / (vmax - vmin)
        r = int(255 * (1 - t))
        g = int(180 * t + 40)
        b = int(255 * (1 - t))
        return f"rgb({r},{g},{b})"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        f'<text x="10" y="20" font-size="14">{key} heatmap</text>',
    ]

    for j, beta in enumerate(bvals):
        x = 70 + j * 70
        lines.append(f'<text x="{x+10}" y="45" font-size="11">b={beta:.2f}</text>')

    for i, lam in enumerate(lvals):
        y = 60 + i * 50
        lines.append(f'<text x="5" y="{y+28}" font-size="11">l={lam:.2f}</text>')
        for j, beta in enumerate(bvals):
            x = 70 + j * 70
            v = val.get((lam, beta), 0.0)
            lines.append(f'<rect x="{x}" y="{y}" width="62" height="40" fill="{color(v)}" stroke="black"/>')
            lines.append(f'<text x="{x+6}" y="{y+24}" font-size="10">{v:.3f}</text>')

    lines.append("</svg>")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--edgelist", required=True)
    ap.add_argument("--attributes", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--a", type=float, default=0.01)
    ap.add_argument("--lambdas", default="0.0,0.05,0.1,0.2,0.4")
    ap.add_argument("--betas", default="0.0,0.1,0.2,0.3,0.5")

    ap.add_argument("--eval-epochs", type=int, default=5)
    ap.add_argument("--eval-runs", type=int, default=1)

    ap.add_argument("--with-link-pred", action="store_true")
    ap.add_argument("--lp-test-ratio", type=float, default=0.1)
    ap.add_argument("--lp-neg-mult", type=float, default=1.0)
    ap.add_argument("--lp-seed", type=int, default=42)
    ap.add_argument("--lp-metric", choices=["dot", "cosine"], default="dot")
    ap.add_argument("--score-node-weight", type=float, default=0.7)
    ap.add_argument("--score-link-weight", type=float, default=0.3)
    ap.add_argument("--select-by", choices=["node_accuracy", "link_auc", "weighted_score"], default="weighted_score")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    run(["make"])

    train_graph = args.edgelist
    lp_pos = None
    lp_neg = None
    if args.with_link_pred:
        train_graph = str(out / "lp_train.edgelist")
        lp_pos = out / "lp_test_pos.txt"
        lp_neg = out / "lp_test_neg.txt"
        run([
            "python3", "scripts/split_link_prediction.py",
            "--edgelist", args.edgelist,
            "--train-out", train_graph,
            "--test-pos-out", str(lp_pos),
            "--test-neg-out", str(lp_neg),
            "--test-ratio", str(args.lp_test_ratio),
            "--neg-mult", str(args.lp_neg_mult),
            "--seed", str(args.lp_seed),
        ])

    lambdas = parse_grid(args.lambdas)
    betas = parse_grid(args.betas)

    rows = []
    for lam in lambdas:
        hier = out / f"hier_l{lam:.3f}".replace(".", "p")
        hier = Path(str(hier) + ".txt")
        run(["./recpart_attr", train_graph, str(hier), args.attributes, str(lam), "4"])

        for beta in betas:
            tag = f"l{lam:.3f}_b{beta:.3f}".replace(".", "p")
            vec = out / f"vec_{tag}.txt"

            t0 = time.time()
            run(["./hi2vec_attr", str(args.dim), str(args.a), str(beta), str(hier), args.attributes, str(vec)])
            embed_t = time.time() - t0

            acc_m, acc_s = evaluate_node(vec, args.labels, args.eval_epochs, args.eval_runs)
            auc_v = ""
            ap_v = ""
            if args.with_link_pred:
                auc, ap = evaluate_link(vec, lp_pos, lp_neg, args.lp_metric)
                auc_v = f"{auc:.6f}"
                ap_v = f"{ap:.6f}"
                weighted = args.score_node_weight * acc_m + args.score_link_weight * auc
            else:
                weighted = acc_m

            rows.append({
                "dataset": args.dataset_name,
                "lambda": f"{lam:.6f}",
                "beta": f"{beta:.6f}",
                "embed_time_sec": f"{embed_t:.6f}",
                "node_accuracy_mean": f"{acc_m:.6f}",
                "node_accuracy_std": f"{acc_s:.6f}",
                "link_auc": auc_v,
                "link_ap": ap_v,
                "weighted_score": f"{weighted:.6f}",
                "vectors": str(vec),
            })
    if args.select_by == "node_accuracy":
        rows.sort(key=lambda r: float(r["node_accuracy_mean"]), reverse=True)
    elif args.select_by == "link_auc":
        rows.sort(key=lambda r: float(r["link_auc"]) if r["link_auc"] else -1.0, reverse=True)
    else:
        rows.sort(key=lambda r: float(r["weighted_score"]), reverse=True)

    out_csv = out / "sweep_results.csv"
    fields = list(rows[0].keys()) if rows else []
    if fields:
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

        write_heatmap_svg(rows, lambdas, betas, "node_accuracy_mean", out / "sweep_node_accuracy.svg")
        if args.with_link_pred:
            write_heatmap_svg(rows, lambdas, betas, "link_auc", out / "sweep_link_auc.svg")

    print(f"sweep_csv {out_csv}")
    if rows:
        best = rows[0]
        print(f"best_lambda {best['lambda']}")
        print(f"best_beta {best['beta']}")
        print(f"best_node_accuracy {best['node_accuracy_mean']}")
        print(f"best_link_auc {best['link_auc']}")
        print(f"best_weighted_score {best['weighted_score']}")


if __name__ == "__main__":
    main()
