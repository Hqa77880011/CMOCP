# CMOCP

Implementation of **Online Condition-Aware Conformal Predictive Maintenance under Sensor Gaps and Unseen Operating Regimes**.

## Installation

Use Python 3.11+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

Create a CSV with `path,key,label,split,group,rpm,load`. `path` is relative to the CSV, `key` selects a MAT variable, and NPY inputs may be one vector or `channels × samples`. `label` is consecutive from zero, with healthy label zero. Use `train`, `val`, `cal`, and `test` for `split`; `group` is an original recording or bearing. Group reuse across partitions is rejected. Row order defines stream order, and windows preserve recording order.

For the primary CWRU protocol assign one load to each of training, validation, calibration, and held-out test, rotating all four roles in separate manifests. Assign recordings before windowing. Do not divide overlapping windows from one recording across partitions.

```bash
cmocp prepare --manifest recordings.csv --output bearing.npz --window 2048 --hop 1024
```

## Training and inference

```bash
cmocp train --data bearing.npz --output outputs/classifier --epochs 50 --seed 0
```

## Evaluation

```bash
cmocp evaluate --data bearing.npz --model outputs/classifier/model.pt --output outputs/full --variant full --missing 0.3 --gap block --spikes 0.01 --delay 25 --seed 0
```

## Ablations

Supported controls: `aps`, `raps`, `aci`, `regime`, `no_regime`, `no_mask`, `no_online`, `no_gate`. Reuse the same inputs, seed and budget. For retrained controls, evaluate the corresponding saved model.

```bash
cmocp evaluate --data bearing.npz --model outputs/classifier/model.pt --output outputs/aps --variant aps --missing 0.3 --gap block --spikes 0.01 --delay 25 --seed 0
```

## Outputs

Preparation writes `bearing.npz`; training writes `model.pt` and `history.json`. Evaluation writes `decisions.json`, `metrics.json` and `coverage.png` to its output folder.

## Implementation and functional validation

This update reuses the manuscript-based classifier/calibrator, aligns the randomized RAPS score with the paper's explicit `u` convention, derives class taxonomy from training labels, and fixes NumPy boolean serialization in decision output. Original recording manifests, training artifacts and paper experiment runs are not supplied here.

| Paper module / metric | Entry | Command / output |
| --- | --- | --- |
| Group-disjoint preparation / train-only signal scaling | `prepare` | `cmocp prepare ...` |
| Fixed classifier | `Classifier`, `train` | `cmocp train ...` |
| RAPS, context-mask-recency weighting, delayed update/gate | `scores`, `evaluate` | `cmocp evaluate ...` |
| Group/marginal coverage, set size, selective risk, maintenance cost | `evaluate`, `wilson` | `metrics.json`, `decisions.json` |
| Functional validation | `smoke.py` | `python smoke.py` |

`python smoke.py` passed on Python 3.13.5 / torch 2.7.1+cu118 CPU, seed 0, torch/BLAS threads 1. It reads raw NPY recordings through a manifest, prepares four partitions, trains one epoch, checks nonzero classifier gradients/updates and identical checkpoint predictions, runs full/APS/no-online streams, verifies delay timing and hand-computed RAPS/Wilson/coverage values. Synthetic signals validate behavior only. Original CWRU load rotations and long shifted-stream results remain unverified; backbone capacity and fixed kernel scales are documented implementation choices rather than recovered original experiment settings.
