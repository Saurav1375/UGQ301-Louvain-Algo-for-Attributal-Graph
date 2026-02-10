#!/usr/bin/env bash
set -euo pipefail

mkdir -p runs_real

python3 scripts/prepare_real_datasets.py --outdir data/real --cora-hash-dim 256
make

python3 scripts/run_experiments.py \
  --edgelist data/real/cora/edgelist.txt \
  --attributes data/real/cora/attributes.txt \
  --labels data/real/cora/labels.txt \
  --outdir runs_real/cora \
  --dim 32 --a 0.01 --beta 0.3 --lambda-attr 0.2 \
  --walk-length 5 --num-walks 1 --window 3 --epochs 1 --neg 2 \
  --eval-epochs 5 --eval-runs 1

python3 scripts/run_experiments.py \
  --edgelist data/real/blogcatalog/edgelist.txt \
  --attributes data/real/blogcatalog/attributes.txt \
  --labels data/real/blogcatalog/labels.txt \
  --outdir runs_real/blogcatalog \
  --dim 32 --a 0.01 --beta 0.3 --lambda-attr 0.2 \
  --walk-length 5 --num-walks 1 --window 3 --epochs 1 --neg 2 \
  --eval-epochs 5 --eval-runs 1
