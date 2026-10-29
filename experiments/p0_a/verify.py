from pathlib import Path
import json,hashlib,subprocess,os
import numpy as np
R=Path(__file__).resolve().parent;P=json.loads((R/'PROTOCOL.json').read_text());A=R/'audit';A.mkdir(exist_ok=True);C=A/'configs.txt';C.write_text('GateDisabled 4 100 0 1.1 0 -1\n')
if not (A/'GateDisabled_r0.meta.json').exists():
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 with (A/'run.log').open('w') as f:subprocess.run([str(R/'search'),P['index'],P['query_file'],P['gt_file'],P['model'],'unused',str(C),str(A),'0'],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
assert sha(A/'GateDisabled_r0.ids')==sha(R/'runs/Vanilla_r0.ids')
a=np.genfromtxt(A/'GateDisabled_r0.csv',delimiter=',',names=True);b=np.genfromtxt(R/'runs/Vanilla_r0.csv',delimiter=',',names=True)
for k in ['hits','reads','expansions']:np.testing.assert_array_equal(a[k],b[k])
assert np.all(a['skips']==0);assert a['model_calls'].sum()>0
v=json.loads((R/'verification.json').read_text());v['original_gate_disabled_exact_vanilla_ids_hits_reads_expansions']=True
(R/'verification.json').write_text(json.dumps(v,indent=2)+'\n');print('GATE DISABLED PARITY PASSED')
