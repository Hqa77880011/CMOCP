"""Synthetic checks for online conformal streams."""
import csv
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cmocp import cli as c


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    # Equation (RAPS): sum of earlier probabilities plus u times this probability.
    np.testing.assert_allclose(c.scores(np.array([.7, .2, .1]), .2), [.14, .74, .92])
    low, high = c.wilson(5, 10)
    np.testing.assert_allclose([low, high], [.236593090512564, .763406909487436], atol=1e-10)
    model = c.Classifier(1, 2)
    x = torch.randn(4, 1, 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    old = model.head.weight.detach().clone()
    loss = torch.nn.functional.cross_entropy(model(x, torch.ones_like(x))[0], torch.tensor([0, 1, 0, 1]))
    loss.backward()
    assert torch.isfinite(loss) and model.head.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(old, model.head.weight)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rows = []
        rng = np.random.default_rng(0)
        for split in ['train', 'val', 'cal', 'test']:
            for label in range(2):
                path = f'{split}{label}.npy'
                np.save(root / path, rng.normal(label * 2, 1, 128).astype(np.float32))
                rows.append(dict(path=path, key='', label=label, split=split,
                                 group=path, rpm=1700, load=label))
        with (root / 'manifest.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        c.prepare(SimpleNamespace(manifest=str(root / 'manifest.csv'), output=str(root / 'data.npz'), window=32, hop=32))
        overlapping = rows + [{**rows[0], 'split': 'test'}]
        with (root / 'overlap.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(overlapping)
        try:
            c.prepare(SimpleNamespace(manifest=str(root / 'overlap.csv'), output=str(root / 'invalid.npz'),
                                      window=32, hop=32, guarded_seen=True))
        except ValueError as error:
            assert 'guard band' in str(error)
        else:
            raise AssertionError('Overlapping train/test recording segments must be rejected')
        guarded = [{**row, 'start': 0, 'stop': 32 if i == 0 else 128} for i, row in enumerate(rows)]
        guarded.append({**rows[0], 'split': 'test', 'start': 96, 'stop': 128})
        with (root / 'guarded.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(guarded[0]))
            writer.writeheader()
            writer.writerows(guarded)
        c.prepare(SimpleNamespace(manifest=str(root / 'guarded.csv'), output=str(root / 'guarded.npz'),
                                  window=32, hop=32, guarded_seen=True))
        with np.load(root / 'guarded.npz') as guarded_data:
            assert set(guarded_data['start'][guarded_data['group'] == rows[0]['group']]) == {0, 96}
        c.train(SimpleNamespace(data=str(root / 'data.npz'), output=str(root / 'model'), epochs=1, batch=4, seed=0))
        state = torch.load(root / 'model/model.pt', weights_only=False)
        first, second = c.Classifier(1, 2), c.Classifier(1, 2)
        first.load_state_dict(state['state'])
        second.load_state_dict(state['state'])
        first.eval(); second.eval()
        torch.testing.assert_close(first(x, torch.ones_like(x))[0], second(x, torch.ones_like(x))[0], rtol=0, atol=0)
        for variant in ['full', 'aps', 'no_online', 'no_blending']:
            args = SimpleNamespace(data=str(root / 'data.npz'), model=str(root / 'model/model.pt'),
                                   output=str(root / variant), variant=variant, seed=0, missing=.25,
                                   spikes=0., noise=0., gap='block', buffer=30, alpha=.1, minimum=80,
                                   safe=1, distance=100., confidence=0., delay=2, corruption_after=2, bootstrap=4, rolling=2)
            c.evaluate(args)
            records = json.loads((root / variant / 'decisions.json').read_text())
            metric = json.loads((root / variant / 'metrics.json').read_text())
            assert metric['coverage'] == sum(r['covered'] for r in records) / len(records)
            assert all(r['observed_fraction'] == 1 for r in records[:2])
            assert all(r['observed_fraction'] == .75 for r in records[2:])
            assert all(abs(r['alpha'] - .1) < 1e-12 for r in records[:3])
            if variant == 'no_blending':
                assert all(r['local_weight'] == 1 for r in records)
            if variant == 'full':
                assert records[0]['local_weight'] < 1
            if variant == 'no_online':
                assert len(set(r['buffer'] for r in records)) == 1
            assert (root / variant / 'coverage.png').stat().st_size > 0
    print('PASS: RAPS/Wilson arithmetic, raw data preparation, gradient/update, checkpoint, delayed stream and baseline controls')


if __name__ == '__main__':
    main()
