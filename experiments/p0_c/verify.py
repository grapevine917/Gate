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
for r in RUNS:
 m=json.loads(Path(r['prefix']+'.meta.json').read_text());assert m['reader_bytes']==m['linux_read_bytes']==m['reader_requests']*4096
for p,h in json.loads((R.parent/'main_comparison/INPUT_LOCK.json').read_text()).items():assert hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()==h
v=json.loads((R/'verification.json').read_text());v['independent_repeated_per_query_pairs_checked']=checks;v['all_per_query_hits_reads_expansions_skips_gate_et_model_calls_identical']=True;v['original_main_comparison_inputs_still_unchanged']=True
(R/'verification.json').write_text(json.dumps(v,indent=2)+'\n');print('INDEPENDENT VERIFICATION PASSED:',checks,'repeat pairs')
