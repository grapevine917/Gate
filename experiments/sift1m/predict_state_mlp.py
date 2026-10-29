#!/usr/bin/env python3
"""Predict state-only MLP probabilities and skips from raw ordered state."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np

def predict_state(raw_state, bundle):
    raw_state = np.asarray(raw_state, dtype='float32')
    if raw_state.ndim != 2 or raw_state.shape[1] != 17 or not np.isfinite(raw_state).all():
        raise ValueError('Expected finite raw pre_io_state with shape [N,17]')
    p = np.empty(len(raw_state), dtype='float64')
    for i in range(0, len(raw_state), 8192):
        x = raw_state[i:i+8192].copy()
        x -= bundle['mean']
        x /= bundle['scale']
        p[i:i+len(x)] = bundle['model'].predict_proba(x)[:, 1]
    return p

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-dir', default=str(Path(__file__).resolve().parent/'state_mlp'))
    parser.add_argument('--operating-point', default='validation_fnr_0.1pct')
    args = parser.parse_args()
    folder = Path(args.model_dir)
    assert json.loads((folder/'COMPLETE.json').read_text())['complete']
    threshold = json.loads((folder/'thresholds.json').read_text())[args.operating_point]['threshold']
    p = predict_state(np.load(args.state_file, mmap_mode='r'), joblib.load(folder/'model.joblib'))
    np.savez(args.output, p_retain=p, skip=p < threshold, threshold=threshold)
