#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def load_rows(paths):
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                if r.get("method") == "compile":
                    continue
                rows.append(r)
    return rows


def save_csv(rows, out_path):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def grouped(rows, metric):
    datasets = sorted(set(r["dataset"] for r in rows))
    methods = ["louvainNE", "attr-louvainNE", "deepwalk", "node2vec"]

    data = {}
    for ds in datasets:
        for m in methods:
            data[(ds, m)] = 0.0

    for r in rows:
        ds = r["dataset"]
        m = r["method"]
        v = r.get(metric, "")
        if v == "":
            continue
        data[(ds, m)] = float(v)

    return datasets, methods, data


def plot_grouped_bar(rows, metric, title, out_svg):
    datasets, methods, data = grouped(rows, metric)
    if not datasets:
        return

    vals = [data[(d, m)] for d in datasets for m in methods]
    vmax = max(vals) if vals else 1.0
    if vmax <= 0:
        vmax = 1.0

    colors = {
        "louvainNE": "#1f77b4",
        "attr-louvainNE": "#d62728",
        "deepwalk": "#2ca02c",
        "node2vec": "#ff7f0e",
    }

    w = 140 + 220 * len(datasets)
    h = 420
    left = 70
    top = 50
    chart_w = w - left - 20
    chart_h = 280
    y0 = top + chart_h

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">']
    lines.append(f'<text x="20" y="24" font-size="18">{title}</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{y0}" stroke="black"/>')
    lines.append(f'<line x1="{left}" y1="{y0}" x2="{left+chart_w}" y2="{y0}" stroke="black"/>')

    for i in range(6):
        v = vmax * i / 5.0
        y = y0 - chart_h * i / 5.0
        lines.append(f'<line x1="{left}" y1="{y}" x2="{left+chart_w}" y2="{y}" stroke="#ddd"/>')
        lines.append(f'<text x="8" y="{y+4}" font-size="10">{v:.2f}</text>')

    ds_gap = chart_w / max(1, len(datasets))
    bar_w = ds_gap / 6.0

    for di, ds in enumerate(datasets):
        x_base = left + di * ds_gap + ds_gap * 0.12
        lines.append(f'<text x="{x_base+2*bar_w}" y="{y0+18}" font-size="12">{ds}</text>')
        for mi, m in enumerate(methods):
            v = data[(ds, m)]
            bh = (v / vmax) * chart_h
            x = x_base + mi * (bar_w + 6)
            y = y0 - bh
            lines.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bh}" fill="{colors[m]}"/>')
            lines.append(f'<text x="{x}" y="{y-4}" font-size="9">{v:.3f}</text>')

    lx = left
    ly = y0 + 50
    for i, m in enumerate(methods):
        x = lx + i * 140
        lines.append(f'<rect x="{x}" y="{ly}" width="14" height="14" fill="{colors[m]}"/>')
        lines.append(f'<text x="{x+20}" y="{ly+12}" font-size="12">{m}</text>')

    lines.append("</svg>")
    Path(out_svg).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.inputs)
    save_csv(rows, out / "combined_results.csv")

    plot_grouped_bar(rows, "total_time_sec", "Total Runtime (sec)", out / "time_comparison.svg")
    plot_grouped_bar(rows, "node_accuracy_mean", "Node Classification Accuracy", out / "node_accuracy.svg")
    plot_grouped_bar(rows, "link_auc", "Link Prediction AUC", out / "link_auc.svg")

    print(f"plots_outdir {out}")


if __name__ == "__main__":
    main()
