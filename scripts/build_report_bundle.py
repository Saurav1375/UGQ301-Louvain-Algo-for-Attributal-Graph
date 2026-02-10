#!/usr/bin/env python3
import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path


def copy_if_exists(src, dst_dir, runs_root):
    p = Path(src)
    if p.exists():
        rel = p.relative_to(runs_root)
        safe_name = "__".join(rel.parts)
        dst = dst_dir / safe_name
        shutil.copy2(p, dst)
        return dst
    return None


def read_csv_rows(path):
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def markdown_table(rows, columns):
    if not rows:
        return "_No data available._\n"
    header = "| " + " | ".join(columns) + " |\n"
    sep = "|" + "|".join(["---" for _ in columns]) + "|\n"
    body = []
    for r in rows:
        body.append("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |\n")
    return header + sep + "".join(body)


def filter_methods(rows):
    return [r for r in rows if r.get("method") and r.get("method") != "compile"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs_real")
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else runs_root / f"report_bundle_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    copied = []
    files_to_copy = [
        runs_root / "benchmark_results.csv",
        runs_root / "tuned_summary.csv",
        runs_root / "cora/results.csv",
        runs_root / "blogcatalog/results.csv",
        runs_root / "cora_tuned/results.csv",
        runs_root / "blogcatalog_tuned/results.csv",
        runs_root / "cora_sweep/sweep_results.csv",
        runs_root / "blogcatalog_sweep/sweep_results.csv",
        runs_root / "cora_sweep_dense/sweep_results.csv",
        runs_root / "blogcatalog_sweep_dense/sweep_results.csv",
        runs_root / "figures/time_comparison.svg",
        runs_root / "figures/node_accuracy.svg",
        runs_root / "figures/link_auc.svg",
        runs_root / "figures_tuned/time_comparison.svg",
        runs_root / "figures_tuned/node_accuracy.svg",
        runs_root / "figures_tuned/link_auc.svg",
    ]

    for p in files_to_copy:
        cp = copy_if_exists(p, outdir, runs_root)
        if cp is not None:
            copied.append(cp.name)

    bench = filter_methods(read_csv_rows(runs_root / "benchmark_results.csv"))
    tuned_sum = read_csv_rows(runs_root / "tuned_summary.csv")
    cora_dense = read_csv_rows(runs_root / "cora_sweep_dense/sweep_results.csv")[:10]
    blog_dense = read_csv_rows(runs_root / "blogcatalog_sweep_dense/sweep_results.csv")[:10]

    summary_md = outdir / "SUMMARY.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Report Bundle Summary\n\n")
        f.write(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`\n\n")

        f.write("## Included Files\n\n")
        for name in copied:
            f.write(f"- `{name}`\n")
        if not copied:
            f.write("- _No files were copied._\n")
        f.write("\n")

        f.write("## Benchmark Results\n\n")
        f.write(markdown_table(
            bench,
            ["dataset", "method", "total_time_sec", "embed_time_sec", "node_accuracy_mean", "link_auc", "link_ap"],
        ))
        f.write("\n")

        f.write("## Selected Tuned Params\n\n")
        f.write(markdown_table(
            tuned_sum,
            ["dataset", "best_lambda", "best_beta", "selection", "node_accuracy", "link_auc", "weighted_score"],
        ))
        f.write("\n")

        f.write("## Top Dense Sweep Rows (Cora)\n\n")
        f.write(markdown_table(
            cora_dense,
            ["lambda", "beta", "node_accuracy_mean", "link_auc", "weighted_score", "embed_time_sec"],
        ))
        f.write("\n")

        f.write("## Top Dense Sweep Rows (BlogCatalog)\n\n")
        f.write(markdown_table(
            blog_dense,
            ["lambda", "beta", "node_accuracy_mean", "link_auc", "weighted_score", "embed_time_sec"],
        ))

    print(f"bundle_dir {outdir}")
    print(f"summary_md {summary_md}")


if __name__ == "__main__":
    main()
