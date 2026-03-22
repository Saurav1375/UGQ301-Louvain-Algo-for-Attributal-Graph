# Attributed LouvainNE (Fast, Non-GNN)

This repository contains:
- Original LouvainNE baselines (`recpart`, `hi2vec`, `renum`)
- Attributed extension integrated **inside** the LouvainNE pipeline:
  - `recpart_attr`: attributes influence hierarchy construction (community moves)
  - `hi2vec_attr`: attributes influence embedding generation
- From-scratch baselines for comparison:
  - `DeepWalk`
  - `Node2Vec`
- Evaluation/benchmark scripts for timing + node classification accuracy.

## Build

```bash
make
```

## Input formats

### Graph edgelist
Undirected, unweighted, one edge per line:

```text
u v
```

### Attributes file
One node per line:

```text
node_id f1 f2 ... fD
```

All lines must have the same attribute dimension `D`.

### Labels file (for node classification)

```text
node_id class_id
```

## Baseline LouvainNE

```bash
./recpart edgelist.txt hierarchy.txt 1
./hi2vec 128 0.01 hierarchy.txt vectors.txt
```

## Attributed LouvainNE

Attributes are integrated at two internal stages:
1. **Hierarchy construction** via attributed Louvain gain in `recpart_attr`
2. **Embedding generation** via attribute projection in `hi2vec_attr`

```bash
./recpart_attr edgelist.txt hierarchy_attr.txt attributes.txt 0.2 4
./hi2vec_attr 128 0.01 0.3 hierarchy_attr.txt attributes.txt vectors_attr.txt
```

Parameters:
- `lambda` (4th arg of `recpart_attr`): attribute weight in partition gain
- `beta` (3rd arg of `hi2vec_attr`): attribute injection strength in embedding
- `a`: hierarchy damping factor (same role as LouvainNE)

## DeepWalk / Node2Vec baselines

```bash
python3 scripts/embed_baselines.py --edgelist edgelist.txt --out vec_dw.txt --method deepwalk --dim 128
python3 scripts/embed_baselines.py --edgelist edgelist.txt --out vec_n2v.txt --method node2vec --p 1.0 --q 0.5 --dim 128
```

## Evaluate node classification accuracy

```bash
python3 scripts/eval_node_classification.py --vectors vectors_attr.txt --labels labels.txt --train-ratio 0.5 --runs 5
```

## End-to-end benchmark runner

```bash
python3 scripts/run_experiments.py \
  --edgelist edgelist.txt \
  --attributes attributes.txt \
  --labels labels.txt \
  --outdir runs \
  --dim 128 --a 0.01 --beta 0.3 --lambda-attr 0.2 \
  --with-method1 --method1-alpha 0.8 --method1-knn-k 5 --method1-modes preserve,augment
```

This runs:
- LouvainNE
- Attributed LouvainNE
- Method 1 preserve
- Method 1 augment
- DeepWalk
- Node2Vec

and writes `results.csv` (time + node classification + optional link prediction).

## SOP Method 1: Structural Reweighting

Method 1 is implemented as a separate weighted-graph pipeline that preserves the LouvainNE architecture and injects attributes into edge weights before hierarchy construction.

### Edge-preserving variant

Reweight only existing edges:

```bash
python3 scripts/build_reweighted_graph.py \
  --edgelist edgelist.txt \
  --attributes attributes.txt \
  --out weighted_graph.txt \
  --mode preserve \
  --alpha 0.8

./recpart_weighted weighted_graph.txt hierarchy.txt 1
./hi2vec 128 0.01 hierarchy.txt vectors.txt
```

For an existing edge `(i,j)`, the new weight is:

```text
w(i,j) = alpha * 1 + (1 - alpha) * sim(x_i, x_j)
```

### Edge-augmenting variant

Add top-`k` semantic neighbors in attribute space:

```bash
python3 scripts/build_reweighted_graph.py \
  --edgelist edgelist.txt \
  --attributes attributes.txt \
  --out weighted_graph_aug.txt \
  --mode augment \
  --alpha 0.8 \
  --knn-k 5

./recpart_weighted weighted_graph_aug.txt hierarchy_aug.txt 1
./hi2vec 128 0.01 hierarchy_aug.txt vectors_aug.txt
```

For a non-edge added by augmentation, the weight is:

```text
w(i,j) = (1 - alpha) * sim(x_i, x_j)
```

### End-to-end Method 1 runner

```bash
python3 scripts/run_method1.py \
  --dataset-name cora \
  --edgelist data/real/cora/edgelist.txt \
  --attributes data/real/cora/attributes.txt \
  --labels data/real/cora/labels.txt \
  --outdir runs_real/cora_method1 \
  --mode preserve \
  --alpha 0.8 \
  --with-link-pred
```

## Link prediction evaluation

Generate a train/test split and evaluate AUC/AP:

```bash
python3 scripts/split_link_prediction.py \
  --edgelist edgelist.txt \
  --train-out lp_train.edgelist \
  --test-pos-out lp_test_pos.txt \
  --test-neg-out lp_test_neg.txt \
  --test-ratio 0.1 --neg-mult 1.0 --seed 42

python3 scripts/eval_link_prediction.py \
  --vectors vectors.txt \
  --test-pos lp_test_pos.txt \
  --test-neg lp_test_neg.txt \
  --metric dot
```

To run link prediction inside the benchmark runner:

```bash
python3 scripts/run_experiments.py \
  --dataset-name cora \
  --edgelist data/real/cora/edgelist.txt \
  --attributes data/real/cora/attributes.txt \
  --labels data/real/cora/labels.txt \
  --outdir runs_real/cora \
  --with-link-pred
```

## Hyperparameter sweep (`lambda`, `beta`)

```bash
python3 scripts/sweep_attr_params.py \
  --dataset-name cora \
  --edgelist data/real/cora/edgelist.txt \
  --attributes data/real/cora/attributes.txt \
  --labels data/real/cora/labels.txt \
  --outdir runs_real/cora_sweep \
  --lambdas 0.0,0.05,0.1,0.2,0.4 \
  --betas 0.0,0.1,0.2,0.3,0.5 \
  --with-link-pred
```

Outputs:
- `sweep_results.csv`
- `sweep_node_accuracy.svg`
- `sweep_link_auc.svg` (if link prediction enabled)

## CSV export + plotting for report/SOP

Run both Cora and BlogCatalog benchmarks and auto-generate plots:

```bash
./scripts/benchmark_real.sh
```

Outputs:
- `runs_real/benchmark_results.csv` (combined CSV)
- `runs_real/cora/results.csv`
- `runs_real/blogcatalog/results.csv`
- `runs_real/figures/time_comparison.svg`
- `runs_real/figures/node_accuracy.svg`
- `runs_real/figures/link_auc.svg`

## Notes on complexity

- Structural LouvainNE behavior remains near original design.
- Attribute integration is lightweight (no GNN/message passing/backprop over graph).
- Extra cost is local vector arithmetic during community scoring and vector output.
