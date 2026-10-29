from pathlib import Path
import json,hashlib,subprocess,time
import numpy as np
R=Path(__file__).resolve().parent
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
if (R/'COLLECT_COMPLETE.json').exists():raise SystemExit(0)
info={}
for split,prefix,L,cap in [('train','mixed','L.txt',32),('train','l100','L100.txt',10000),('validation','validation','L100.txt',10000),('test','test','L100.txt',10000)]:
 out=R/'data'/prefix
 with (R/f'collect_{prefix}.log').open('w') as f:subprocess.run([str(R/'native/collect'),str(R/'index/ann'),str(R/'data'/f'{split}.u8bin'),str(R/'data'/f'{split}.gt'),str(R/'models/dummy.bin'),str(R/L),str(out),str(cap)],stdout=f,stderr=subprocess.STDOUT,check=True)
 prune=np.fromfile(str(out)+'.prune.bin','<f4').reshape(-1,20);nodes=np.fromfile(str(out)+'.prune_nodes.u32','<u4');et=np.fromfile(str(out)+'.et.bin','<f4').reshape(-1,19)
 assert len(nodes)==len(prune) and np.all(nodes<100000000) and np.all(prune[:,1]==0) and np.isfinite(prune).all() and np.isfinite(et).all()
 assert np.any(nodes>2**24) and np.any(nodes%2), 'High node IDs must retain integer low bits'
 info[prefix]=dict(prune_rows=len(prune),cache_miss_rows=int((prune[:,13]==0).sum()),et_rows=len(et),queries=len(np.unique(prune[:,0])),node_min=int(nodes.min()),node_max=int(nodes.max()),node_id_dtype='uint32')
 print(time.strftime('%H:%M:%S'),'COLLECTED',prefix,info[prefix],flush=True)
lock=json.loads((R/'INPUT_LOCK.json').read_text())
for p in (R/'data').iterdir():
 if p.is_file() and p.suffix in ['.bin','.u32']:lock[str(p)]=sha(p)
dump(R/'INPUT_LOCK.json',lock);dump(R/'COLLECT_COMPLETE.json',info)
