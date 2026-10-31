import argparse
import csv
import json
import time
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn


class Classifier(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(channels * 2, 32, 9, padding=4), nn.ReLU(),
                                     nn.MaxPool1d(4), nn.Conv1d(32, 64, 7, padding=3), nn.ReLU(),
                                     nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Linear(64, classes)

    def forward(self, x, mask):
        hidden = self.encoder(torch.cat([x * mask, mask], 1))
        return self.head(hidden), hidden


def scores(probability, randomizer, penalty=.01, rank_start=3):
    order = np.argsort(-probability)
    ranked = probability[order]
    ranked_scores = np.cumsum(ranked) - (1 - randomizer) * ranked
    ranked_scores += penalty * np.maximum(np.arange(1, len(ranked) + 1) - rank_start, 0)
    result = np.empty_like(ranked_scores)
    result[order] = ranked_scores
    return result


def prepare(args):
    rows = list(csv.DictReader(open(args.manifest, encoding='utf-8')))
    groups, xs, ys, splits, contexts, identities = {}, [], [], [], [], []
    for row in rows:
        if row['split'] not in ('train', 'val', 'cal', 'test'):
            raise ValueError('split must be train, val, cal or test')
        if row['group'] in groups and groups[row['group']] != row['split']:
            raise ValueError('A source recording occurs in more than one partition')
        groups[row['group']] = row['split']
        path = Path(args.manifest).parent / row['path']
        signal = np.asarray(loadmat(path)[row['key']] if path.suffix == '.mat' else np.load(path)).squeeze()
        if signal.ndim == 1:
            signal = signal[None]
        if signal.ndim != 2 or not np.isfinite(signal).all():
            raise ValueError('Signals must have shape channels,samples')
        for t in range(0, signal.shape[-1] - args.window + 1, args.hop):
            xs.append(signal[:, t:t + args.window])
            ys.append(int(row['label']))
            splits.append(row['split'])
            contexts.append([float(row['rpm']), float(row['load'])])
            identities.append(row['group'])
    x, split = np.asarray(xs, dtype=np.float32), np.asarray(splits)
    if set(split) != {'train', 'val', 'cal', 'test'}:
        raise ValueError('Four nonempty partitions are required')
    mean = x[split == 'train'].mean((0, 2), keepdims=True)
    std = x[split == 'train'].std((0, 2), keepdims=True).clip(1e-6)
    np.savez_compressed(args.output, x=(x - mean) / std, y=ys, split=split,
                        context=contexts, group=identities, mean=mean, std=std)


def train(args):
    torch.manual_seed(args.seed)
    d = np.load(args.data)
    x, y = torch.tensor(d['x']), torch.tensor(d['y'], dtype=torch.long)
    classes = np.unique(d['y'][d['split'] == 'train'])
    if not np.array_equal(classes, np.arange(len(classes))) or not np.isin(d['y'], classes).all():
        raise ValueError('Training labels must define the consecutive registered taxonomy for every partition')
    model = Classifier(x.shape[1], len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    train_ids = torch.tensor(np.flatnonzero(d['split'] == 'train'))
    val_ids = torch.tensor(np.flatnonzero(d['split'] == 'val'))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    history, best = [], float('inf')
    for epoch in range(args.epochs):
        model.train()
        for b in train_ids[torch.randperm(len(train_ids))].split(args.batch):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x[b], torch.ones_like(x[b]))[0], y[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = sum(nn.functional.cross_entropy(model(x[b], torch.ones_like(x[b]))[0], y[b], reduction='sum').item()
                      for b in val_ids.split(args.batch)) / len(val_ids)
        history.append({'epoch': epoch + 1, 'val_loss': val})
        if val < best:
            best = val
            torch.save({'state': model.state_dict(), 'channels': x.shape[1], 'classes': len(classes)}, out / 'model.pt')
    (out / 'history.json').write_text(json.dumps(history, indent=2))


def wilson(success, total):
    z = 1.95996398454
    rate = success / total
    center = (rate + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [float(center - half), float(center + half)]


def evaluate(args):
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    d = np.load(args.data)
    checkpoint = torch.load(args.model, weights_only=False, map_location='cpu')
    model = Classifier(checkpoint['channels'], checkpoint['classes'])
    model.load_state_dict(checkpoint['state'])
    model.eval()
    ids = np.flatnonzero(np.isin(d['split'], ['cal', 'test']))
    probabilities, embeddings, masks = [], [], []
    with torch.no_grad():
        for i in ids:
            x = d['x'][i].copy()
            mask = np.ones_like(x)
            if d['split'][i] == 'test':
                count = round(args.missing * x.size)
                if args.gap == 'random':
                    mask.flat[rng.choice(x.size, count, replace=False)] = 0
                elif args.gap == 'block':
                    length = round(args.missing * x.shape[-1])
                    for channel in range(len(x)):
                        start = rng.integers(x.shape[-1] - length + 1)
                        mask[channel, start:start + length] = 0
                else:
                    count = round(args.missing * len(x))
                    mask[rng.choice(len(x), count, replace=False)] = 0
                spike_ids = rng.choice(x.size, round(args.spikes * x.size), replace=False)
                x.flat[spike_ids] += rng.choice([-6., 6.], len(spike_ids))
                x += rng.normal(0, args.noise, x.shape)
            logits, hidden = model(torch.tensor(x[None], dtype=torch.float32), torch.tensor(mask[None], dtype=torch.float32))
            probabilities.append(logits.softmax(-1)[0].numpy())
            embeddings.append(hidden[0].numpy())
            masks.append(mask.ravel().astype(bool))
    p, h, masks = np.asarray(probabilities), np.asarray(embeddings), np.asarray(masks)
    calibration = d['split'][ids] == 'cal'
    context = np.concatenate([d['context'][ids], h], axis=1)
    context = (context - context[calibration].mean(0)) / context[calibration].std(0).clip(1e-6)
    y = d['y'][ids]
    candidate = np.stack([scores(v, rng.random(), 0 if args.variant == 'aps' else .01) for v in p])
    buffer = [{'score': candidate[i, y[i]], 'z': context[i], 'mask': masks[i], 'time': -1}
              for i in np.flatnonzero(calibration)][-args.buffer:]
    if not buffer:
        raise ValueError('Calibration partition is empty')
    alpha, queue, records = args.alpha, [], []
    tracemalloc.start()
    for tick, i in enumerate(np.flatnonzero(~calibration)):
        begin = time.perf_counter()
        zs = np.stack([v['z'] for v in buffer])
        ms = np.stack([v['mask'] for v in buffer])
        distance = np.square(zs - context[i]).sum(1)
        union = np.logical_or(ms, masks[i]).sum(1)
        jaccard = np.where(union > 0, 1 - np.logical_and(ms, masks[i]).sum(1) / union.clip(1), 0.)
        logw = -.5 * distance - 2 * jaccard - .002 * (tick - np.array([v['time'] for v in buffer]))
        if args.variant == 'no_regime':
            logw += .5 * distance
        if args.variant in ('no_mask', 'regime'):
            logw += 2 * jaccard
        if args.variant in ('aps', 'raps', 'aci'):
            logw[:] = 0
        w = np.exp(logw - logw.max())
        w /= w.sum()
        neff = 1 / np.square(w).sum()
        blend = 1. if args.variant == 'no_blending' else min(1., neff / args.minimum)
        w = blend * w + (1 - blend) / len(w)
        order = np.argsort([v['score'] for v in buffer])
        threshold_index = min(np.searchsorted(w[order].cumsum(), 1 - alpha), len(w) - 1)
        q = buffer[order[threshold_index]]['score']
        prediction_set = np.flatnonzero(candidate[i] <= q).tolist()
        accepted = (len(prediction_set) == 1 and neff >= args.safe and
                    np.sqrt(distance.min()) <= args.distance and p[i].max() >= args.confidence)
        if args.variant == 'no_gate':
            accepted = True
        action = int(p[i].argmax()) if accepted else -1
        covered = int(y[i]) in prediction_set
        record = {'index': int(ids[i]), 'tick': tick, 'label': int(y[i]), 'prediction': int(p[i].argmax()),
                  'group': str(d['group'][ids[i]]), 'regime': d['context'][ids[i]].tolist(),
                  'set': prediction_set, 'accepted': bool(accepted), 'action': action, 'covered': covered,
                  'neff': float(neff), 'local_weight': float(blend), 'alpha': alpha, 'buffer': len(buffer),
                  'priority': len(prediction_set) + max(0, args.safe - neff) + float(np.sqrt(distance.min()))}
        queue.append((tick + args.delay, i, covered))
        while queue and queue[0][0] <= tick:
            _, old, old_covered = queue.pop(0)
            if args.variant not in ('aps', 'raps', 'regime', 'no_online'):
                alpha = float(np.clip(alpha + .005 * (args.alpha - (not old_covered)), .01, .30))
                buffer.append({'score': candidate[old, y[old]], 'z': context[old], 'mask': masks[old], 'time': tick})
                buffer = buffer[-args.buffer:]
        record['latency_ms'] = (time.perf_counter() - begin) * 1000
        records.append(record)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    covered = np.array([v['covered'] for v in records])
    truth = np.array([v['label'] for v in records])
    actions = np.array([v['action'] for v in records])
    accepted = actions >= 0
    miss = accepted & (actions == 0) & (truth != 0)
    other = accepted & (actions != truth) & ~miss
    grouped = {}
    for field in ['label', 'regime']:
        for key in sorted(set(str(v[field]) for v in records)):
            selected = np.array([str(v[field]) == key for v in records])
            n, hit = int(selected.sum()), int(covered[selected].sum())
            grouped[field + ':' + key] = {'coverage': hit / n, 'n': n, 'wilson95': wilson(hit, n)}
    for key in sorted(set((v['label'], tuple(v['regime'])) for v in records)):
        selected = np.array([(v['label'], tuple(v['regime'])) == key for v in records])
        n, hit = int(selected.sum()), int(covered[selected].sum())
        grouped['joint:' + str(key)] = {'coverage': hit / n, 'n': n, 'wilson95': wilson(hit, n)}
    units = sorted(set(v['group'] for v in records))
    bootstrap = []
    for _ in range(args.bootstrap):
        sampled = rng.choice(units, len(units), replace=True)
        b = np.concatenate([np.flatnonzero([v['group'] == unit for v in records]) for unit in sampled])
        bootstrap.append(float(covered[b].mean()))
    rolling = np.convolve(covered, np.ones(args.rolling) / args.rolling, 'valid') if len(covered) >= args.rolling else np.array([])
    stable = np.abs(rolling - (1 - args.alpha)) <= .02
    recovery = next((i + args.rolling for i in range(max(0, len(stable) - 19)) if stable[i:i + 20].all()), None)
    result = {'balanced_accuracy': balanced_accuracy_score(truth, [v['prediction'] for v in records]),
              'macro_f1': f1_score(truth, [v['prediction'] for v in records], average='macro'),
              'coverage': float(covered.mean()), 'coverage_bootstrap95': np.quantile(bootstrap, [.025, .975]).tolist() if bootstrap else None,
              'worst_group_shortfall': max(0., 1 - args.alpha - min(v['coverage'] for v in grouped.values())),
              'mean_set_size': float(np.mean([len(v['set']) for v in records])),
              'selective_risk': float((actions[accepted] != truth[accepted]).mean()) if accepted.any() else None,
              'rejection_rate': float((~accepted).mean()), 'fault_miss_rate': float(miss.sum() / max(1, (truth != 0).sum())),
              'decision_cost': float((10 * miss.sum() + 3 * other.sum() + (~accepted).sum()) / len(records)),
              'recovery_windows_from_stream_start': recovery, 'update_frequency': float(sum(v['alpha'] != records[max(0, i - 1)]['alpha'] for i, v in enumerate(records)) / len(records)),
              'latency_median_ms': float(np.median([v['latency_ms'] for v in records])),
              'peak_python_memory_bytes': peak, 'maximum_buffer': max(v['buffer'] for v in records), 'groups': grouped}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
    (out / 'decisions.json').write_text(json.dumps(records, indent=2))
    plt.plot(np.arange(len(rolling)) + args.rolling, rolling)
    plt.axhline(1 - args.alpha, color='gray', linestyle='--')
    plt.xlabel('Stream window')
    plt.ylabel('Rolling coverage')
    plt.tight_layout()
    plt.savefig(out / 'coverage.png', dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--window', type=int, default=2048)
    p.add_argument('--hop', type=int, default=1024)
    p = sub.add_parser('train')
    for key in ['data', 'output']:
        p.add_argument('--' + key, required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p = sub.add_parser('evaluate')
    for key in ['data', 'model', 'output']:
        p.add_argument('--' + key, required=True)
    p.add_argument('--variant', choices=['full', 'aps', 'raps', 'aci', 'regime', 'no_regime', 'no_mask', 'no_online', 'no_gate', 'no_blending'], default='full')
    p.add_argument('--gap', choices=['random', 'block', 'channel'], default='block')
    for key, default in [('missing', .3), ('spikes', .01), ('noise', 0.), ('alpha', .1), ('distance', 10.), ('confidence', .5)]:
        p.add_argument('--' + key, type=float, default=default)
    for key, default in [('delay', 25), ('buffer', 1200), ('minimum', 80), ('safe', 40), ('seed', 0), ('rolling', 50), ('bootstrap', 2000)]:
        p.add_argument('--' + key, type=int, default=default)
    args = parser.parse_args()
    if args.command == 'evaluate' and (not 0 <= args.missing <= 1 or not 0 <= args.spikes <= 1 or args.delay < 0):
        parser.error('missing/spikes must be fractions and delay must be nonnegative')
    {'prepare': prepare, 'train': train, 'evaluate': evaluate}[args.command](args)
