#!/usr/bin/env python3
"""Predict M2 from raw query and candidate vectors, constructing interactions."""
import argparse
from pathlib import Path
import joblib
import numpy as np

def interaction_features(q, v):
    q, v = np.asarray(q, dtype='float32'), np.asarray(v, dtype='float32')
    if q.ndim != 2 or q.shape != v.shape or q.shape[1] != 128:
        raise ValueError('q and candidate v must both have shape [N,128]')
    if not (np.isfinite(q).all() and np.isfinite(v).all()):
        raise ValueError('Input vectors must be finite')
    return np.concatenate((q, v, np.abs(q-v), q*v), axis=1)

def predict_pairs(q, v, bundle):
    if np.shape(q) != np.shape(v) or len(np.shape(q)) != 2 or np.shape(q)[1] != 128:
        raise ValueError('q and candidate v must both have shape [N,128]')
    p = np.empty(len(q), dtype='float64')
    for i in range(0, len(q), 8192):
        x = interaction_features(q[i:i+8192], v[i:i+8192])
        x -= bundle['mean']
        x /= bundle['scale']
        p[i:i+len(x)] = bundle['model'].predict_proba(x)[:, 1]
    return p

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--q', required=True)
    parser.add_argument('--v', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default=str(Path(__file__).resolve().parent/'model2/model.joblib'))
    args = parser.parse_args()
    np.save(args.output, predict_pairs(np.load(args.q, mmap_mode='r'),
                                     np.load(args.v, mmap_mode='r'), joblib.load(args.model)))
