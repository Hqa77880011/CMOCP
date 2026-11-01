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

Supported controls: `aps`, `raps`, `aci`, `regime`, `no_regime`, `no_mask`, `no_online`, `no_gate`, `no_blending`. Reuse the same inputs, seed and budget. For retrained controls, evaluate the corresponding saved model.

```bash
cmocp evaluate --data bearing.npz --model outputs/classifier/model.pt --output outputs/aps --variant aps --missing 0.3 --gap block --spikes 0.01 --delay 25 --seed 0
```

## Outputs

Preparation writes `bearing.npz`; training writes `model.pt` and `history.json`. Evaluation writes `decisions.json`, `metrics.json` and `coverage.png` to its output folder.

## Calibration and metrics

The classifier remains fixed during stream evaluation. Class taxonomy comes from training labels; RAPS scores, context/mask/recency weights and delayed feedback drive the online gate. `decisions.json` records the local-distribution weight at each step; `no_blending` fixes it to one. Metrics include marginal/group coverage, set size, selective risk and maintenance cost.

## Official CWRU acquisition, load rotations and long drift

The executable protocol uses the university's [12k drive-end fault table](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data)
and [normal baseline records](https://engineering.case.edu/bearingdatacenter/normal-baseline-data).
It downloads original MAT files, discovers their actual drive-end/RPM keys,
retains source URLs in a local catalog, and registers ten classes: healthy
(label zero), inner-race/ball/outer-race at 0.007/0.014/0.021 inches. Outer-race
records use the declared 6 o'clock table entries. Raw files, prepared arrays and
trained weights stay in ignored `data/` and `outputs/` folders.

```bash
python -m cmocp.protocol download --output data/cwru
python -m cmocp.protocol rotations --catalog data/cwru/catalog.json --output data/rotations
python -m cmocp.protocol run --data data/rotations --output outputs/rotations --epochs 50 --seed 0
```

Four cyclic assignments give every load exactly one turn as train, validation,
calibration and test. `rotations --all-permutations` produces all 24 role
permutations instead. Assignment precedes windowing and each original recording
is atomic in this protocol. `run` fits a separate classifier for each fold,
then evaluates its fixed checkpoint with matched variants, missing fractions and
label delays. `summary.json` indexes the executed fold/settings; metrics are
computed from the evaluated stream. The default matrix is missing
fractions 0.1/0.3/0.5 by delays 0/10/25/50 and the registered calibration controls.
Use `--variants`, `--missing`, `--delays`, `--epochs` and `--bootstrap` to run a
small functional subset.

The long-drift protocol is separate from atomic load rotation. It trains on the
first half of load-0 records; a two-window source-sample guard separates the
held-out load-0 prefix. Loads 1/2 supply validation/calibration, and load 3
supplies the unseen-regime suffix. Windows are interleaved across source records
while preserving order within each record. No window is repeated to manufacture
a long stream. A request exceeding available distinct windows raises an error.

```bash
python -m cmocp.protocol drift --catalog data/cwru/catalog.json --output data/drift.npz --prefix-windows 160 --total-windows 1200
python -m cmocp.protocol run --data data/drift.npz --output outputs/drift --epochs 50 --variants full aps raps aci no_blending --missing 0.3 --delays 0 10 25 50 --corruption-after 320 --seed 0
```

The first transition is the actual seen-to-unseen load change at window 160;
block missingness/spikes begin at window 320. `--corruption-after` is also
available to `cmocp evaluate` and defaults to zero for the original compound
shift evaluation. Decision records expose actual `observed_fraction`.
For a custom seen-regime manifest, `start`/`stop` are absolute source-sample
bounds. `cmocp prepare --guarded-seen` permits only train/test sharing of the
same physical recording/key with at least two full window lengths between
segments. Overlap, including identical starts, is rejected. The default preparer
continues to reject source-record reuse. Signal normalization uses training
samples only; context/embedding normalization uses calibration covariates
without labels and is then frozen for the stream.

## Checks

```bash
python smoke.py
```

Synthetic stream checks cover grouped preparation, classifier updates, frozen replay, feedback timing, guarded source intervals and RAPS/Wilson/coverage arithmetic.
