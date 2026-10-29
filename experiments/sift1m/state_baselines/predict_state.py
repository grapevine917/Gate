#!/usr/bin/env python3
"""Predict retain probabilities and skip decisions from raw pre_io_state only."""
import argparse,json,os
os.environ.setdefault('OMP_NUM_THREADS','4'); os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import numpy as np
import joblib
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parent
class StateGate:
 def __init__(self,model='xgboost',operating_point='validation_fnr_0.1pct'):
  if model not in ['logistic_regression','xgboost']: raise ValueError('Unknown model')
  assert json.loads((ROOT/'COMPLETE.json').read_text())['complete']
  folder=ROOT/model; self.kind=model
  provenance=json.loads((ROOT/'provenance.json').read_text()); self.feature_names=provenance['features']
  self.threshold=json.loads((folder/'thresholds.json').read_text())[operating_point]['threshold']
  if model=='logistic_regression':
   b=joblib.load(folder/'model.joblib'); self.model=b['model']; self.mean=b['mean']; self.scale=b['scale']
  else:
   self.model=XGBClassifier(n_jobs=4); self.model.load_model(folder/'model.ubj')
 def predict(self,raw_state):
  x=np.atleast_2d(np.asarray(raw_state,dtype=np.float32))
  if x.shape[1]!=len(self.feature_names) or not np.isfinite(x).all(): raise ValueError('Expected finite raw state with 17 ordered columns')
  if self.kind=='logistic_regression': x=np.asarray((x-self.mean)/self.scale,dtype=np.float64)
  p=self.model.predict_proba(x)[:,1]; return {'p_retain':p,'skip':p<self.threshold,'threshold':self.threshold}
if __name__=='__main__':
 a=argparse.ArgumentParser(); a.add_argument('--model',choices=['logistic_regression','xgboost'],default='xgboost'); a.add_argument('--state-file',required=True); a.add_argument('--operating-point',default='validation_fnr_0.1pct'); a.add_argument('--output',required=True); args=a.parse_args()
 g=StateGate(args.model,args.operating_point); result=g.predict(np.load(args.state_file,mmap_mode='r')); np.savez(args.output,**result)
 print(f"Predicted {len(result['p_retain'])} samples, skip={int(result['skip'].sum())}, threshold={g.threshold}")
