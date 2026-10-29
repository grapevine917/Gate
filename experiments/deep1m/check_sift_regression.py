#!/usr/bin/env python3
"""Confirm dimension-general state gate entry preserves original SIFT controls."""
import sys,json
from pathlib import Path
import numpy as np
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E.parent/'sift1m'))
from run_online_m3 import E as SIFT,execute,mat
from calibrate_et_pruning import env_for
folder=E/'sift_regression';folder.mkdir(exist_ok=True)
configs=json.loads((SIFT/'et_pruning/online/configs.json').read_text())
keys=['baseline','et_unsafe_query_0.25pct','prune_fnr_0.5pct','joint_fnr_0.5pct_et_query_0.25pct']
checks=[]
for key in keys:
 c=next(r for r in configs if r['key']==key)
 execute(key,SIFT/'online_m3/test_query.fbin',SIFT/'online_m3/test_gt.bin',env_for(c['prune_tau'],c['et_tau']),output_dir=folder)
 for suffix,dtype in [('idx_uint32','<u4'),('dists_float','<f4')]:np.testing.assert_array_equal(mat(folder/f'{key}_100_{suffix}.bin',dtype),mat(SIFT/f'et_pruning/online/r0_{key}_100_{suffix}.bin',dtype))
 old=np.genfromtxt(SIFT/f'et_pruning/online/r0_{key}.csv',delimiter=',',names=True);new=np.genfromtxt(folder/f'{key}.csv',delimiter=',',names=True)
 for col in ('ssd_reads','expansions','gate_evaluated','gate_skipped','et_evaluated','et_stopped'):np.testing.assert_array_equal(new[col],old[col])
 checks.append(dict(key=key,ids_distances_and_all_gate_counts_identical=True));print('SIFT REGRESSION PASS',key,flush=True)
(folder/'COMPLETE.json').write_text(json.dumps(dict(complete=True,queries=1000,checks=checks),indent=2)+'\n')
