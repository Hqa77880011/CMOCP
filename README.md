# CMOCP

Implementation of **Online Condition-Aware Conformal Predictive Maintenance under Sensor Gaps and Unseen Operating Regimes**.

CMOCP calibrates a fixed diagnostic CNN. RAPS scores are weighted by operating context, classifier embedding, observation-mask Jaccard similarity, and recency. Effective sample size blends this local distribution with the global reference. Verified delayed labels update the requested error level; a separate support, distance, confidence, and singleton gate determines inspection.

## Installation and data

```bash
python -m venv .venv
python -m pip install -e .
```

Use Python 3.11+. Activate the environment before subsequent commands. Obtain the 12 kHz drive-end recordings from the [CWRU bearing data center](https://engineering.case.edu/bearingdatacenter). The paper's primary task has ten classes: normal, inner-race, ball, and outer-race faults at the three selected sizes. Keep outer-race orientation and channel choices fixed. Paderborn is available from its [bearing data center](https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter); map shared health states explicitly and split by bearing.

Create a CSV with `path,key,label,split,group,rpm,load`. `path` is relative to the CSV, `key` selects a MAT variable, and NPY inputs may be one vector or `channels × samples`. `label` is consecutive from zero, with healthy label zero. Use `train`, `val`, `cal`, and `test` for `split`; `group` is an original recording or bearing. Group reuse across partitions is rejected. Row order defines stream order, and windows preserve recording order.

For the primary CWRU protocol assign one load to each of training, validation, calibration, and held-out test, rotating all four roles in separate manifests. Assign recordings before windowing. Do not divide overlapping windows from one recording across partitions.

```bash
cmocp prepare --manifest recordings.csv --output bearing.npz --window 2048 --hop 1024
cmocp train --data bearing.npz --output outputs/classifier --epochs 50 --seed 0
```

Preparation fits channel normalization on training only. The two-convolution CNN receives the zero-masked signal plus its mask and emits a 64-dimensional embedding. Training uses Adam at 0.001 and clean training windows; the lowest validation cross-entropy checkpoint is saved. The calibration layer never modifies this classifier.

## Compound-shift comparison

```bash
cmocp evaluate --data bearing.npz --model outputs/classifier/model.pt --output outputs/full --variant full --missing 0.3 --gap block --spikes 0.01 --delay 25
cmocp evaluate --data bearing.npz --model outputs/classifier/model.pt --output outputs/raps --variant raps --missing 0.3 --gap block --spikes 0.01 --delay 25
cmocp evaluate --data bearing.npz --model outputs/classifier/model.pt --output outputs/aci --variant aci --missing 0.3 --gap block --spikes 0.01 --delay 25
```

All variants share deterministic corruption, CNN outputs, randomized RAPS draws, and stream order for a fixed seed. `aps` removes the rank penalty; `raps` fixes global calibration; `aci` uses global calibration with delayed online updates; `regime` uses context/recency weighting with a fixed target and no mask term. Ablations are `no_regime`, `no_mask`, `no_online`, and `no_gate`. `no_gate` accepts the top class even when the prediction set is ambiguous.

Run `--missing 0.1`, `0.3`, or `0.5` with `--gap random|block|channel`. Random and block gaps select the requested sample fraction after integer rounding; channel dropout rounds the requested number of complete channels. Spikes have signed amplitude six training standard deviations. `--noise` adds independent Gaussian noise in normalized units. Calibration remains clean, while held-out test windows receive the declared corruption. `--delay 0|10|25|50` controls feedback arrivals. Delay zero is processed after the current decision; it cannot affect that decision.

Defaults follow the paper's calibration settings: buffer 1200, minimum ESS 80, safe ESS 40, error level 0.1, feedback step 0.005 clipped to [0.01,0.30], mask weight 2, and recency weight 0.002. RAPS uses rank penalty 0.01 above rank three. Context and embedding coordinates are normalized on calibration only. `--distance` is the maximum nearest-context Euclidean distance and `--confidence` the singleton confidence threshold; choose both on validation conditions. The default values 10 and 0.5 are implementation settings. The empirical blended quantile follows the manuscript; it does not add an exchangeability correction or claim conditional validity under shift.

## Outputs and interpretation

Each evaluation writes:

- `decisions.json`: prediction set, action, true label for retrospective scoring, ESS, alpha, inspection priority, buffer size, and measured calibration/gating latency per window.
- `metrics.json`: balanced accuracy, Macro-F1, coverage, record-bootstrap coverage interval, class/regime/joint coverage with Wilson intervals, worst-group shortfall, set size, selective risk, rejection, fault misses, decision cost, recovery, update frequency, latency, and traced Python memory.
- `coverage.png`: rolling empirical coverage against the requested target.

The mutually exclusive decision cost is ten per faulty window accepted as healthy, three per other accepted error, and one per inspection. Selective risk is undefined and written `null` when nothing is accepted. Group coverage includes all observed class, operating-context, and joint groups within the chosen corruption family. Compare distinct corruption runs together to study family robustness. Memory measures Python allocations inside calibration, excluding native PyTorch classifier memory and the already computed fixed-backbone arrays.

Recovery is the first endpoint of 20 consecutive rolling windows within ±0.02 of target, relative to the start of the shifted test stream; the rolling width is `--rolling`, default 50. Use a new ordered manifest for a distinct shift episode. Bootstrap resamples recording groups and preserves every window in each sampled group. Run five seeds and retain per-seed outputs for paired comparisons. No experimental numbers are included.
