from pathlib import Path
import json,hashlib,subprocess
import numpy as np
R=Path(__file__).resolve().parent;A=R.parent/'p0_a';M=R.parent/'main_comparison';C=R.parent/'p0_c';S=R.parent/'sift1m'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
source=(A/'search.cpp').read_text();(R/'search.cpp').write_text(source)
c=json.loads((A/'build.command.json').read_text());c=[str(R/'search.cpp') if x==str(A/'search.cpp') else str(R/'search') if x==str(A/'search') else x for x in c];dump(R/'build.command.json',c)
with (R/'build.log').open('w') as f:subprocess.run(c,stdout=f,stderr=subprocess.STDOUT,check=True)
# Same validation retain-FNR convention as P0-A; cache hits always Keep at runtime.
states=np.load(S/'training_data_l100/validation/pre_io_state.npy',mmap_mode='r');y=np.load(S/'training_data_l100/validation/y.npy');scores=np.ones(len(y),'<f4');native=np.fromfile(C/'prediction/validation_lr.scores','<f4');assert len(native)==int((states[:,10]==0).sum());scores[states[:,10]==0]=native
qids=np.load(S/'training_data_l100/validation/query_ids.npy');assert np.array_equal(qids,np.load(M/'data/validation_ids.npy'))
risks=[0,.001,.0025,.005,.01,.02,.05,.1];points=[];positive=np.sort(scores[y==1])
for risk in risks:
 allowed=int(np.floor(risk*len(positive)));tau=0. if risk==0 else float(positive[allowed]);false=int(((y==1)&(scores<tau)).sum());assert false<=allowed
 points.append(dict(risk=risk,tau=tau,validation_retain_count=len(positive),allowed_false_skips=allowed,false_skips=false,skip_count=int((scores<tau).sum())))
dump(R/'LR_CALIBRATION.json',dict(model=str(M/'models/lr.bin'),model_sha256=sha(M/'models/lr.bin'),source='Native frozen P0-C LR scores on baseline validation L100, same query split; never test scores. Cached scores forced Keep=1; retain-FNR denominator all retain candidates as original P0-A convention.',points=points))
p=json.loads((A/'PROTOCOL.json').read_text());p.update(model=str(M/'models/lr.bin'),gate_threshold=points[4]['tau'],default_definition='Frozen P0-C matched mixed-L17D LR, validation retain-FNR1% threshold. Weights differ from original P0-A MLP training; this is a replacement rerun, not a pure matched-training model ablation.',experiment='P0-A LR replacement; original MLP artifacts preserved',calibration='LR_CALIBRATION.json',repeats=9)
p['limits']+=['LR is mixed-L trained from P0-C while original P0-A/B MLP was L100-trained; cross-version deltas include different weights/training.','Offline validation retain-FNR budget is not an online Recall bound.']
dump(R/'PROTOCOL.json',p)
inputs=[R/'search',R/'search.cpp',R/'PROTOCOL.json',R/'LR_CALIBRATION.json',M/'models/lr.bin',M/'native/libdiskann.a',C/'prediction/validation_lr.scores',S/'training_data_l100/validation/pre_io_state.npy',S/'training_data_l100/validation/y.npy',M/'data/test.fbin',M/'data/test.gt',S/'index_disk.index',S/'index_pq_pivots.bin',S/'index_pq_compressed.bin']
dump(R/'INPUT_LOCK.json',{str(x):sha(x) for x in inputs});print('P0-A LR READY',p['gate_threshold'],flush=True)
