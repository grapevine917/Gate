from pathlib import Path
import json,hashlib,subprocess,struct,platform
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parent;LG=ROOT.parents[1];summary=json.loads((ROOT/'results.json').read_text());verification={}
for ds in ['sift','deep']:
 checks={};sets=[]
 for split in ['train','validation','test']:
  p=ROOT/ds/split;a=np.loadtxt(p/'labels.csv',delimiter=',',skiprows=1);x=a[:,7:];ids=np.load(p/'query_ids.npy');sets.append(set(ids.tolist()))
  assert len(np.unique(a[:,0]))==len(ids)==300 and np.isfinite(x).all()
  assert np.all(x[:,18]<=x[:,17]) and np.all((x[:,22]>0)==(x[:,18]>0)) and np.all(x[:,20]<=x[:,18]) and np.all(x[:,21]<=x[:,17])
  for q in np.unique(a[:,0]):
   v=x[a[:,0]==q];assert (np.diff(v[:,17])>0).all() and (np.diff(v[:,18])>=0).all()
  for q in np.unique(a[:,0]):
   rows=a[a[:,0]==q];past=[]
   for row in rows:
    state=row[7:];assert state[18]==len(past)
    assert np.isclose(state[19],len(past)/state[17] if state[17] else 0,rtol=1e-6,atol=1e-7)
    if past:
     assert np.isclose(state[23],np.mean(past),rtol=1e-6)
     assert np.isclose(state[25],min(past),rtol=1e-6) and np.isclose(state[26],max(past),rtol=1e-6) and np.isclose(state[27],past[-1],rtol=1e-6)
    if row[2]:past.append(state[0])
  checks[split]={'events':len(a),'history_invariants':True}
 assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
 data=np.loadtxt(ROOT/ds/'test/labels.csv',delimiter=',',skiprows=1)[:,7:].astype('<f4')
 inp=ROOT/ds/'native_test_states.fbin'
 with open(inp,'wb') as f:np.array(data.shape,'<u4').tofile(f);data.tofile(f)
 for name in ['retrained_17_lr','full_44_lr']:
  model=ROOT/ds/(name+'.bin');out=ROOT/ds/(name+'.native_scores.f32')
  r=subprocess.run([str(ROOT/'native_check'),str(model),str(inp),str(out)],check=True,capture_output=True,text=True)
  with open(model,'rb') as f:
   magic=f.read(8);width=struct.unpack('<I',f.read(4))[0];d=17 if magic==b'LRGATE01' else width;m=np.fromfile(f,'<f4',d);scale=np.fromfile(f,'<f4',d);w=np.fromfile(f,'<f4',d);b=np.fromfile(f,'<f4',1)[0]
  expected=expit(-(((data[:,:d].astype(float)-m)/scale)@w+b)).astype(np.float32);actual=np.fromfile(out,'<f4');error=float(np.max(np.abs(expected-actual)));assert error<1e-5
  checks[name]={'native_score_max_error':error,'native_history_unit_checks':True,'stdout':r.stdout.strip()}
 verification[ds]=checks
online=json.loads((ROOT/'online_results.json').read_text());assert set(online)=={'sift','deep'} and all(set(v)=={'validation','test'} for v in online.values())
(ROOT/'verification.json').write_text(json.dumps(verification,indent=2))
files=list(ROOT.glob('*.sh'))+list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+[ROOT/'native_check.cpp',ROOT/'native_check',ROOT/'results.json',ROOT/'online_results.json',ROOT/'query_splits.json',ROOT/'verification.json']+list(ROOT.glob('*/*.bin'))+list(ROOT.glob('*/*/labels.csv'))
files += [LG/'DiskANN/include/online_gate.h',LG/'DiskANN/src/pq_flash_index.cpp',LG/'DiskANN/apps/collect_history_labels.cpp',LG/'DiskANN/build/apps/collect_history_labels',LG/'DiskANN/build/apps/search_disk_index']
files += list((ROOT.parent/'io_predictability').glob('*/state_lr/model.bin'))+list((ROOT.parent/'io_predictability').glob('*/state_lr/thresholds.json'))
manifest={'complete':True,'python':platform.python_version(),'selection':'validation only','scope':'conditional frozen-policy predictability and actual replacement-policy online query results','sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(ROOT/'COMPLETE.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(verification,indent=2))
