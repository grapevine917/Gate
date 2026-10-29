#!/usr/bin/env python3
"""Predict M3 retain probabilities and skips from raw q, candidate v and state."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from predict_model2 import interaction_features

def joint_features(q, v, state):
    pair = interaction_features(q, v)
    state = np.asarray(state, dtype='float32')
    if state.shape != (len(pair),17) or not np.isfinite(state).all():
        raise ValueError('Expected finite raw pre_io_state with shape [N,17]')
    return np.concatenate((pair,state),axis=1)

def predict_joint(q, v, state, bundle):
    if np.shape(q) != np.shape(v) or len(np.shape(q)) != 2 or np.shape(q)[1] != 128:
        raise ValueError('q and candidate v must both have shape [N,128]')
    if np.shape(state) != (len(q),17):
        raise ValueError('state must have shape [N,17]')
    p=np.empty(len(q),dtype='float64')
    for i in range(0,len(q),8192):
        x=joint_features(q[i:i+8192],v[i:i+8192],state[i:i+8192])
        x-=bundle['mean']
        x/=bundle['scale']
        p[i:i+len(x)]=bundle['model'].predict_proba(x)[:,1]
    return p

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--q',required=True)
    parser.add_argument('--v',required=True)
    parser.add_argument('--state',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--model-dir',default=str(Path(__file__).resolve().parent/'model3'))
    parser.add_argument('--operating-point',default='validation_fnr_0.1pct')
    args=parser.parse_args()
    folder=Path(args.model_dir)
    assert json.loads((folder/'COMPLETE.json').read_text())['complete']
    p=predict_joint(np.load(args.q,mmap_mode='r'),np.load(args.v,mmap_mode='r'),
                    np.load(args.state,mmap_mode='r'),joblib.load(folder/'model.joblib'))
    threshold=json.loads((folder/'thresholds.json').read_text())[args.operating_point]['threshold']
    np.savez(args.output,p_retain=p,skip=p<threshold,threshold=threshold)
