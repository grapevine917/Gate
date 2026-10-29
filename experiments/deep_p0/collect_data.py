from pathlib import Path
import json,hashlib,subprocess,time
import numpy as np
R=Path(__file__).resolve().parent;D=R.parent/'deep1m'
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
if (R/'COLLECT_COMPLETE.json').exists():raise SystemExit(0)
# Rebuild the collector if the all-candidate diagnostic export changed.
commands=json.loads((R/'build.commands.json').read_text());cmd=next(c for c in commands if str(R/'collect.cpp') in c)
with (R/'build_collect.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
info={}
for split,prefix,L,cap in [('train','mixed','L.txt',32),('train','l100','L100.txt',10000),('validation','validation','L100.txt',10000),('test','test','L100.txt',10000)]:
 out=R/'data'/prefix
 with (R/f'collect_{prefix}.log').open('w') as f:subprocess.run([str(R/'native/collect'),str(D/'index'),str(R/'data'/f'{split}.fbin'),str(R/'data'/f'{split}.gt'),str(R/'models/dummy.bin'),str(R/L),str(out),str(cap)],stdout=f,stderr=subprocess.STDOUT,check=True)
 prune=np.fromfile(str(out)+'.prune.bin','<f4').reshape(-1,20);et=np.fromfile(str(out)+'.et.bin','<f4').reshape(-1,19);assert np.isfinite(prune).all() and np.isfinite(et).all()
 info[prefix]=dict(prune_rows=len(prune),cache_miss_rows=int((prune[:,13]==0).sum()),et_rows=len(et),queries=len(np.unique(prune[:,0])))
 print(time.strftime('%H:%M:%S'),'COLLECTED',prefix,info[prefix],flush=True)
lock=json.loads((R/'INPUT_LOCK.json').read_text());lock[str(R/'native/collect')]=sha(R/'native/collect')
for p in (R/'data').glob('*.bin'):lock[str(p)]=sha(p)
dump(R/'INPUT_LOCK.json',lock);dump(R/'COLLECT_COMPLETE.json',info)
