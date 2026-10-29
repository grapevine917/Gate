"""Validation-only baseline pilot; locks eight search L before model training."""
from pathlib import Path
import json,hashlib,subprocess,math
import numpy as np
R=Path(__file__).resolve().parent
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
if (R/'INPUT_LOCK.json').exists():raise SystemExit(0)
for name in ['INDEX_COMPLETE','DATA_COMPLETE','NATIVE_COMPLETE','COMPRESSION_COMPLETE','EXACT_AUDIT_COMPLETE']:assert json.loads((R/f'{name}.json').read_text()).get('passed',True)
P=json.loads((R/'PROTOCOL_BASE.json').read_text());folder=R/'pilot';folder.mkdir(exist_ok=True)
with (R/'models/dummy.bin').open('wb') as f:
 f.write(b'STGATE01');np.zeros(17,'<f4').tofile(f);np.ones(17,'<f4').tofile(f)
 for a,b in [(17,128),(128,64),(64,1)]:np.zeros(a*b+b,'<f4').tofile(f)
stats=[]
for L in P['L_pilot_candidates']:
 cp=folder/f'L{L}.configs';cp.write_text(f'pilot_L{L} 0 {L} 0 1.1 0 -1\n')
 prefix=folder/f'pilot_L{L}_r0'
 if not prefix.with_suffix('.meta.json').exists():
  cmd=[str(R/'native/search'),str(R/'index/ann'),str(R/'data/validation.u8bin'),str(R/'data/validation.gt'),str(R/'models'),str(R/'data/vectors.cvec'),str(cp),str(folder),'0']
  with (folder/f'L{L}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 a=np.genfromtxt(prefix.with_suffix('.csv'),delimiter=',',names=True);meta=json.loads(prefix.with_suffix('.meta.json').read_text());assert meta['reader_bytes']==int(a['reads'].sum())*4096
 physical_ok=meta['linux_read_bytes']==meta['reader_bytes']
 if not physical_ok:
  assert meta.get('linux_read_bytes_source')=='getrusage_ru_inblock_x512' and meta['linux_read_bytes']==0, 'Unexpected physical IO mismatch'
  print('PILOT RECALL ONLY: kernel physical IO accounting unavailable; timing not accepted for formal results',flush=True)
 recall=float(a['hits'].sum()/100);stats.append(dict(L=L,recall=recall,physical_io_verified=physical_ok,timing_valid_for_formal_results=physical_ok,reads=float(a['reads'].mean()),p95_us=float(np.percentile(a['latency_us'],95))))
 print('VALIDATION PILOT',stats[-1],flush=True)
 if L>=300 and recall>=99.9:break
maximum=stats[-1]['L'];LS={15,100,maximum};candidates=[v for v in P['L_pilot_candidates'] if v<=maximum]
while len(LS)<8:LS.add(max((v for v in candidates if v not in LS),key=lambda v:(min(abs(math.log(v/b)) for b in LS),-v)))
P['L_grid']=sorted(LS);dump(R/'PROTOCOL.json',P);dump(R/'L_PILOT_LOCK.json',dict(validation_results=stats,L_grid=sorted(LS),rule=P['L_selection_rule'],test_observed=False))
(R/'L.txt').write_text(' '.join(map(str,sorted(LS)))+'\n');(R/'L100.txt').write_text('100\n')
inputs=[*[R/'index'/f'ann{s}' for s in ['_disk.index','_pq_pivots.bin','_pq_compressed.bin']],*[R/'data'/f'{s}.{ext}' for s in ['train','validation','test'] for ext in ['u8bin','fbin','gt']],R/'data/vectors.cvec',*[R/'native'/n for n in ['libdiskann.a','collect','search','score','verify','online_gate.h']],R/'splits.json',R/'PROTOCOL.json',R/'PROTOCOL_BASE.json',R/'EXACT_AUDIT_COMPLETE.json']
lock=json.loads((R/'DATA_COMPLETE.json').read_text())['sha256']
lock.update({str(p):sha(p) for p in inputs});dump(R/'INPUT_LOCK.json',lock);print('INPUT AND L LOCK COMPLETE',P['L_grid'],flush=True)
