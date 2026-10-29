from pathlib import Path
import json,subprocess,os,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent;LG=ROOT.parents[1];OLD=LG/'experiments/io_predictability'
def read(p,dt):
 with open(p,'rb') as f:
  n,d=np.fromfile(f,np.uint32,2);return np.fromfile(f,dt).reshape(int(n),int(d))
def write(p,a):
 with open(p,'wb') as f:np.array(a.shape,np.uint32).tofile(f);a.tofile(f)
manifest={}
for ds in ['sift','deep']:
 manifest[ds]={}
 for split in ['train','validation','test']:
  src=OLD/ds/split;dst=ROOT/ds/split;dst.mkdir(parents=True,exist_ok=True)
  ids=np.load(src/'query_ids.npy');ix=np.sort(np.random.default_rng(20261017).choice(len(ids),300,replace=False))
  write(dst/'query.fbin',read(src/'query.fbin',np.float32)[ix]);write(dst/'gt.bin',read(src/'gt.bin',np.uint32)[ix]);np.save(dst/'query_ids.npy',ids[ix]);manifest[ds][split]=ids[ix].tolist()
  tau=json.loads((OLD/ds/'state_lr/thresholds.json').read_text())['unsafe_fpr_0.005']['retain_threshold']
  args=[str(LG/'DiskANN/build/apps/collect_history_labels'),str(LG/'experiments'/('sift1m' if ds=='sift' else 'deep1m')/'index'),str(dst/'query.fbin'),str(dst/'gt.bin'),str(OLD/ds/'state_lr/model.bin'),str(tau),str(dst/'labels.csv')]
  (dst/'command.json').write_text(json.dumps({'args':args,'threshold':tau},indent=2))
  if (dst/'COMPLETE.json').exists():continue
  print(ds,split,'start',flush=True)
  with open(dst/'collect.log','w') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
  (dst/'COMPLETE.json').write_text(json.dumps({'complete':True,'sha256':hashlib.sha256((dst/'labels.csv').read_bytes()).hexdigest()}))
  print(ds,split,'complete',flush=True)
(ROOT/'query_splits.json').write_text(json.dumps(manifest,indent=2))
