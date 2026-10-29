#!/usr/bin/env python3
"""Predict p(retain) from two [N,128] NPY arrays, using saved model1."""
import argparse
from pathlib import Path
import joblib
import numpy as np

def predict_pairs(q, v, bundle):
    q, v = np.asarray(q, dtype='float32'), np.asarray(v, dtype='float32')
    if q.ndim != 2 or q.shape != v.shape or q.shape[1] != 128:
        raise ValueError('q and candidate v must both have shape [N,128]')
    if not (np.isfinite(q).all() and np.isfinite(v).all()):
        raise ValueError('Input vectors must be finite')
    output = np.empty(len(q), dtype='float64')
    for i in range(0, len(q), 8192):
        x = np.concatenate((q[i:i+8192], v[i:i+8192]), axis=1)
        x -= bundle['mean']
        x /= bundle['scale']
        output[i:i+len(x)] = bundle['model'].predict_proba(x)[:, 1]
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--q', required=True)
    parser.add_argument('--v', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default=str(Path(__file__).resolve().parent/'model1/model.joblib'))
    args = parser.parse_args()
    np.save(args.output, predict_pairs(np.load(args.q), np.load(args.v), joblib.load(args.model)))
