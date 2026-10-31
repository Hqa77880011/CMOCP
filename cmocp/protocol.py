"""Official CWRU acquisition and registered load-rotation/drift protocols."""
import argparse
import csv
import html
import io
import itertools
import json
import os
import re
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.io import loadmat

from .cli import evaluate, prepare, train

FAULT_PAGE = 'https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data'
NORMAL_PAGE = 'https://engineering.case.edu/bearingdatacenter/normal-baseline-data'
LABELS = ['healthy'] + [f'{kind}{size}' for kind in ['inner', 'ball', 'outer'] for size in ['007', '014', '021']]


def download(args):
    """Fetch selected original MAT records from the university's own pages."""
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    links = {}
    for page in [NORMAL_PAGE, FAULT_PAGE]:
        with urllib.request.urlopen(page, timeout=60) as response:
            text = response.read().decode('utf-8')
        for url, title in re.findall(r'href="([^\"]+\.mat)"[^>]*>([^<]+)<', text):
            links[html.unescape(title.strip())] = (url, page)
    selected = args.classes or LABELS
    if 'healthy' not in selected:
        raise ValueError('Include healthy so the registered healthy label remains zero')
    if not set(selected) <= set(LABELS):
        raise ValueError('classes must be registered CWRU labels: ' + ', '.join(LABELS))
    selected = ['healthy'] + [name for name in LABELS[1:] if name in selected]
    rows = []
    for label, name in enumerate(selected):
        for load in range(4):
            title = f'Normal_{load}' if name == 'healthy' else f"{ {'inner': 'IR', 'ball': 'B', 'outer': 'OR'}[name[:-3]]}{name[-3:]}{'@6' if name.startswith('outer') else ''}_{load}"
            url, page = links[title]
            if not url.startswith('https://engineering.case.edu/'):
                raise ValueError('Refusing a MAT link outside the registered university host')
            path = root / Path(url).name
            if not path.exists():
                with urllib.request.urlopen(url, timeout=90) as response:
                    payload = response.read()
                loadmat(io.BytesIO(payload))
                path.write_bytes(payload)
            source = loadmat(path)
            key = next(k for k in source if k.endswith('_DE_time'))
            rpm_keys = [k for k in source if 'RPM' in k.upper()]
            rpm = float(np.asarray(source[rpm_keys[0]]).ravel()[0]) if rpm_keys else [1797, 1772, 1750, 1730][load]
            rows.append(dict(path=path.name, key=key, label=label, class_name=name,
                             group='cwru_' + path.stem, load=load, rpm=rpm,
                             source_url=url, source_page=page))
    (root / 'catalog.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(f'Downloaded/validated {len(rows)} original MAT records; {len(selected)} classes')


def manifest(rows, path):
    with path.open('w', newline='', encoding='utf-8') as handle:
        fields = ['path', 'key', 'label', 'split', 'group', 'rpm', 'load', 'start', 'stop']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, '') for key in fields} for row in rows)


def catalog_rows(path, output):
    root = Path(path).resolve().parent
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    return [{**row, 'path': os.path.relpath(root / row['path'], output)} for row in rows]


def rotations(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = catalog_rows(args.catalog, out)
    assignments = list(itertools.permutations(range(4))) if args.all_permutations else [tuple((r + k) % 4 for k in range(4)) for r in range(4)]
    for fold, assignment in enumerate(assignments):
        role = dict(zip(assignment, ['train', 'val', 'cal', 'test']))
        current = [{**row, 'split': role[row['load']]} for row in rows]
        path = out / f'fold{fold}.csv'
        manifest(current, path)
        prepare(SimpleNamespace(manifest=str(path), output=str(out / f'fold{fold}.npz'),
                                window=args.window, hop=args.hop, max_windows_per_record=args.max_windows_per_record))
    (out / 'roles.json').write_text(json.dumps([dict(zip(['train', 'val', 'cal', 'test'], a)) for a in assignments], indent=2))


def round_robin(ids, groups):
    streams = [list(ids[groups[ids] == key]) for key in dict.fromkeys(groups[ids])]
    return [int(stream[t]) for t in range(max(map(len, streams), default=0)) for stream in streams if t < len(stream)]


def drift(args):
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = catalog_rows(args.catalog, out.parent)
    segments = []
    for row in rows:
        if row['load'] == 0:
            signal = loadmat(out.parent / row['path'])[row['key']]
            middle = signal.size // 2
            segments.extend([{**row, 'split': 'train', 'start': 0, 'stop': middle},
                             {**row, 'split': 'test', 'start': middle + 2 * args.window, 'stop': signal.size}])
        else:
            segments.append({**row, 'split': {1: 'val', 2: 'cal', 3: 'test'}[row['load']]})
    path = out.with_suffix('.csv')
    manifest(segments, path)
    prepare(SimpleNamespace(manifest=str(path), output=str(out), window=args.window, hop=args.hop,
                            guarded_seen=True, max_windows_per_record=args.max_windows_per_record))
    with np.load(out) as data:
        arrays = {k: data[k] for k in data.files}
    test = arrays['split'] == 'test'
    seen = np.flatnonzero(test & (arrays['context'][:, 1] == 0))
    unseen = np.flatnonzero(test & (arrays['context'][:, 1] == 3))
    prefix = round_robin(seen, arrays['group'])[:args.prefix_windows]
    suffix = round_robin(unseen, arrays['group'])[:max(0, args.total_windows - len(prefix))]
    if len(prefix) != args.prefix_windows or len(prefix) + len(suffix) != args.total_windows:
        raise ValueError('Not enough distinct windows for this stream length; reduce requested length or use all ten classes')
    order = np.r_[np.flatnonzero(~test), prefix, suffix].astype(int)
    for key in ['x', 'y', 'split', 'context', 'group', 'start']:
        arrays[key] = arrays[key][order]
    arrays['protocol'] = np.array('guarded seen load0 -> atomic unseen load3; no repeated windows')
    np.savez_compressed(out, **arrays)


def run(args):
    torch.set_num_threads(1)
    sources = sorted(Path(args.data).glob('fold*.npz')) if Path(args.data).is_dir() else [Path(args.data)]
    results = []
    for path in sources:
        output = Path(args.output) / path.stem
        train(SimpleNamespace(data=str(path), output=str(output / 'model'), epochs=args.epochs, batch=args.batch, seed=args.seed))
        for variant, missing, delay in itertools.product(args.variants, args.missing, args.delays):
            target = output / f'{variant}_missing{missing:g}_delay{delay}'
            evaluate(SimpleNamespace(data=str(path), model=str(output / 'model/model.pt'), output=str(target),
                                     variant=variant, gap='block', missing=missing, spikes=args.spikes, noise=0.,
                                     corruption_after=args.corruption_after, delay=delay, alpha=.1, buffer=1200,
                                     minimum=80, safe=40, distance=10., confidence=.5, seed=args.seed,
                                     bootstrap=args.bootstrap, rolling=args.rolling))
            metrics = json.loads((target / 'metrics.json').read_text())
            results.append(dict(fold=path.stem, variant=variant, missing=missing, delay=delay, **metrics))
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / 'summary.json').write_text(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('download')
    p.add_argument('--output', required=True)
    p.add_argument('--classes', nargs='+')
    for name in ['rotations', 'drift']:
        p = sub.add_parser(name)
        p.add_argument('--catalog', required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--window', type=int, default=2048)
        p.add_argument('--hop', type=int, default=1024)
        p.add_argument('--max-windows-per-record', type=int)
        if name == 'rotations':
            p.add_argument('--all-permutations', action='store_true')
        else:
            p.add_argument('--prefix-windows', type=int, default=160)
            p.add_argument('--total-windows', type=int, default=1200)
    p = sub.add_parser('run')
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--variants', nargs='+', default=['full', 'aps', 'raps', 'aci', 'regime', 'no_blending'])
    p.add_argument('--missing', type=float, nargs='+', default=[.1, .3, .5])
    p.add_argument('--delays', type=int, nargs='+', default=[0, 10, 25, 50])
    p.add_argument('--spikes', type=float, default=.01)
    p.add_argument('--corruption-after', type=int, default=0)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--rolling', type=int, default=50)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
