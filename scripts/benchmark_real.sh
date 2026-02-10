#!/usr/bin/env bash
set -euo pipefail

mkdir -p runs_real
mkdir -p runs_real/figures

rm -f runs_real/benchmark_results.csv

python3 scripts/prepare_real_datasets.py --outdir data/real --cora-hash-dim 256
make

python3 scripts/run_experiments.py \
  --dataset-name cora \
  --edgelist data/real/cora/edgelist.txt \
  --attributes data/real/cora/attributes.txt \
  --labels data/real/cora/labels.txt \
  --outdir runs_real/cora \
  --dim 32 --a 0.01 --beta 0.3 --lambda-attr 0.2 \
  --walk-length 5 --num-walks 1 --window 3 --epochs 1 --neg 2 \
  --eval-epochs 5 --eval-runs 1 \
  --with-link-pred --lp-test-ratio 0.1 --lp-neg-mult 1.0 --lp-seed 42 \
  --global-csv runs_real/benchmark_results.csv

python3 scripts/run_experiments.py \
  --dataset-name blogcatalog \
  --edgelist data/real/blogcatalog/edgelist.txt \
  --attributes data/real/blogcatalog/attributes.txt \
  --labels data/real/blogcatalog/labels.txt \
  --outdir runs_real/blogcatalog \
  --dim 32 --a 0.01 --beta 0.3 --lambda-attr 0.2 \
  --walk-length 5 --num-walks 1 --window 3 --epochs 1 --neg 2 \
  --eval-epochs 5 --eval-runs 1 \
  --with-link-pred --lp-test-ratio 0.1 --lp-neg-mult 1.0 --lp-seed 42 \
  --global-csv runs_real/benchmark_results.csv

python3 scripts/plot_results.py \
  --inputs runs_real/cora/results.csv runs_real/blogcatalog/results.csv \
  --outdir runs_real/figures
