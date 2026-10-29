from pathlib import Path
import json,hashlib
import numpy as np
R=Path(__file__).resolve().parent;A=json.loads((R/'aggregate.json').read_text());RUNS=json.loads((R/'all_test_runs.json').read_text());checks=0
for r in A:
 base=np.genfromtxt(r['prefix']+'.csv',delimiter=',',names=True)
 for q in [q for q in RUNS if q['key']==r['key'] and q['repeat']!=r['repeat']]:
  other=np.genfromtxt(q['prefix']+'.csv',delimiter=',',names=True)
  for field in ['query','hits','reads','expansions','skips','gate_evaluated','et_evaluated','et_stopped','model_calls']:np.testing.assert_array_equal(base[field],other[field])
  checks+=1
for split in ['validation','test']:
 rows=json.loads((R/'validation.json' if split=='validation' else R/'all_test_runs.json').read_text())
 for r in rows:
  m=json.loads(Path(r['prefix']+'.meta.json').read_text());assert m['reader_bytes']==m['linux_read_bytes']==m['reader_requests']*4096
for p,h in json.loads((R.parent/'main_comparison/INPUT_LOCK.json').read_text()).items():assert hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()==h
v=json.loads((R/'verification.json').read_text());v['independent_repeated_per_query_pairs_checked']=checks;v['all_per_query_hits_reads_expansions_skips_gate_et_model_calls_identical']=True;v['original_main_comparison_inputs_still_unchanged']=True
(R/'verification.json').write_text(json.dumps(v,indent=2)+'\n');print('INDEPENDENT VERIFICATION PASSED:',checks,'repeat pairs')

# The replacement must leave all non-Ours trajectories unchanged.
old={r['key']:r for r in json.loads((R.parent/'p0_b/aggregate.json').read_text())}
new={r['key']:r for r in A};unchanged=0
for key,r in new.items():
 if r['family'] in ['Tuned_DiskANN','Heuristic','ET']:
  for field in ['ids_sha256','total_hits','reads','expansions','skips','et_stop_fraction']:assert r[field]==old[key][field],(key,field)
  unchanged+=1
assert unchanged==192
a=R.parent/'p0_a_lr/runs/Ours_r0.ids';assert hashlib.file_digest(a.open('rb'),'sha256').hexdigest()==new['Ours_L100_p4']['ids_sha256']
assert new['Ours_L100_p4']['tau']==json.loads((R.parent/'p0_a_lr/PROTOCOL.json').read_text())['gate_threshold']
# Preserve the historical MLP experiment artifacts.
manifest=json.loads((R.parent/'p0_b/artifact_manifest.json').read_text())
for name in ['PROTOCOL.json','configs.json','aggregate.json','REPORT.md','all_test_runs.json']:
 assert hashlib.file_digest((R.parent/'p0_b'/name).open('rb'),'sha256').hexdigest()==manifest[name]['sha256']
v=json.loads((R/'verification.json').read_text());v['all192_non_Ours_rollouts_identical_previous_MLP_version']=True;v['LR_default_exact_P0A_LR_IDs']=True;v['historical_MLP_artifacts_preserved']=True
(R/'verification.json').write_text(json.dumps(v,indent=2)+'\n');print('LR REPLACEMENT CROSS-CHECKS PASSED')
