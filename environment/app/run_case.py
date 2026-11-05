"""Train a fresh single-network PINN and predict at requested coordinates."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch
from baseline_models import BaselineModel, make_samples, train_model
from error_metrics import relative_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--points', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    samples = make_samples(cfg, args.seed)
    model = BaselineModel('StandardPINN', cfg, args.seed, args.device)
    history, elapsed, evaluations = train_model(model, samples, args.output, args.seed,
                                               {'purpose': 'fresh repair evaluation'})
    coords = np.loadtxt(args.points, delimiter=',', skiprows=1, ndmin=2)
    if coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError('Expected finite r_um,z_um coordinates')
    prediction = model.predict(coords[:, 0], coords[:, 1])
    if not np.isfinite(prediction).all():
        raise ValueError('Non-finite prediction')
    with (args.output/'prediction.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['r_um', 'z_um', 'temperature_K'])
        writer.writerows(zip(coords[:, 0], coords[:, 1], prediction))
    (args.output/'run.json').write_text(json.dumps({
        'seed': args.seed, 'training_seconds': elapsed, 'total_seconds': time.perf_counter()-started,
        'steps': len(history), 'loss_evaluations': evaluations,
        'final_loss': history[-1]['objective'], 'device': args.device,
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
