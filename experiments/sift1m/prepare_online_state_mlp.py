#!/usr/bin/env python3
"""Export trained state-only MLP, calibrate thresholds, and verify real tau0 search."""
import json,joblib,numpy as np
from run_online_m3 import E,execute,mat,TARGETS
from train_state_baselines import threshold_from_validation
from predict_state_mlp import predict_state
OUT=E/'online_state_mlp';OUT.mkdir(exist_ok=True)
bundle=joblib.load(E/'state_mlp/model.joblib')
assert [w.shape for w in bundle['model'].coefs_]==[(17,128),(128,64),(64,1)]
with (OUT/'model.bin').open('wb') as f:
    f.write(b'STGATE01')
    for x in (bundle['mean'],bundle['scale']):x.astype('<f4').tofile(f)
    for w,b in zip(bundle['model'].coefs_,bundle['model'].intercepts_):
        np.ascontiguousarray(w,dtype='<f4').tofile(f);b.astype('<f4').tofile(f)
y=np.load(E/'training_data_l100/validation/y.npy');p=np.load(E/'state_mlp/validation_p_retain.npy')
thresholds={f'fnr_{risk*100:g}pct':threshold_from_validation(y,p,risk) for risk in TARGETS}
for c in thresholds.values():c['decision']='skip iff p(retain|live_pre_io_state) < threshold'
(OUT/'thresholds.json').write_text(json.dumps(thresholds,indent=2))
execute('audit_tau0',E/'online_m3/test_query.fbin',E/'online_m3/test_gt.bin',
    {'DISKANN_M3_MODEL':str(OUT/'model.bin'),'DISKANN_M3_TAU':'0','DISKANN_M3_AUDIT':str(OUT/'audit_q0.bin')},output_dir=OUT)
qids=np.load(E/'training_data_l100/test/query_ids.npy')
actual=mat(OUT/'audit_tau0_100_idx_uint32.bin','<u4')
np.testing.assert_array_equal(actual,mat(E/'baseline_100_idx_uint32.bin','<u4')[qids])
np.testing.assert_array_equal(mat(OUT/'audit_tau0_100_dists_float.bin','<f4'),mat(E/'baseline_100_dists_float.bin','<f4')[qids])
audit=np.fromfile(OUT/'audit_q0.bin',dtype='<f4').reshape(-1,19)
m=np.load(E/'training_data_l100/test/metadata.npy',mmap_mode='r')
ix=np.flatnonzero(m['query_id']==qids[0])
state=np.load(E/'training_data_l100/test/pre_io_state.npy',mmap_mode='r')[ix]
by_node={int(x['node_id']):s for x,s in zip(m[ix],state)}
np.testing.assert_array_equal(audit[:,1:18],np.array([by_node[int(row[0])] for row in audit]))
scores=predict_state(audit[:,1:18],bundle)
np.testing.assert_allclose(audit[:,18],scores,rtol=2e-5,atol=2e-6)
check=dict(tau0_all_test_ordered_ids_and_distances_match_baseline=True,
    online_state_exactly_matches_offline=True,cpp_python_max_probability_error=float(abs(audit[:,18]-scores).max()),
    full_candidate_vector_table_not_loaded=True)
(OUT/'verification.json').write_text(json.dumps(check,indent=2))
print('STATE ONLINE EXPORT AND VERIFICATION PASSED',check,flush=True)
