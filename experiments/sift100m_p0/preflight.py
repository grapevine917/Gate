"""Exercise the changed uint8 collector and disabled-policy parity on a small graph."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,subprocess,struct,shutil
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parents[1];S=R/'preflight';S.mkdir(exist_ok=True)
if (R/'PREFLIGHT_COMPLETE.json').exists():raise SystemExit(0)
base=np.memmap(ROOT/'data/sift100m/base.100M.u8bin','u1','r',offset=8,shape=(100000000,128));v=np.array(base[:20000]);q=np.fromfile(R/'data/train.u8bin','u1',offset=8).reshape(-1,128)[:32];dist=(v.astype('f8')**2).sum(1)[:,None]+(q.astype('f8')**2).sum(1)[None,:]-2*v.astype('f8')@q.astype('f8').T;gt=np.argsort(dist,axis=0)[:100].T.astype('<u4')
for a,name in [(v,'base.u8bin'),(q,'query.u8bin'),(gt,'query.gt')]:
 with (S/name).open('wb') as f:np.array(a.shape,'<u4').tofile(f);a.tofile(f)
cmd=[str(ROOT/'DiskANN/build/apps/build_disk_index'),'--data_type','uint8','--dist_fn','l2','--data_path',str(S/'base.u8bin'),'--index_path_prefix',str(S/'ann'),'-R','64','-L','100','-B','.05','-M','.2','-T','2','--QD','34']
if not (S/'ann_disk.index').exists():
 with (S/'build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
model=S/'models';model.mkdir(exist_ok=True)
with (model/'dummy.bin').open('wb') as f:
 f.write(b'STGATE01');np.zeros(17,'<f4').tofile(f);np.ones(17,'<f4').tofile(f)
 for a,b in [(17,128),(128,64),(64,1)]:np.zeros(a*b+b,'<f4').tofile(f)
shutil.copy(model/'dummy.bin',model/'l100.bin')
for key,kind in [('baseline',0),('disabled',4)]:
 cp=S/f'{key}.configs';cp.write_text(f'{key} {kind} 100 0 1.1 0 -1\n')
 with (S/f'{key}.log').open('w') as f:subprocess.run([str(R/'native/search'),str(S/'ann'),str(S/'query.u8bin'),str(S/'query.gt'),str(model),'unused',str(cp),str(S),'0'],stdout=f,stderr=subprocess.STDOUT,check=True)
a=np.genfromtxt(S/'baseline_r0.csv',delimiter=',',names=True);b=np.genfromtxt(S/'disabled_r0.csv',delimiter=',',names=True)
assert (S/'baseline_r0.ids').read_bytes()==(S/'disabled_r0.ids').read_bytes()
for k in ['hits','reads','expansions']:np.testing.assert_array_equal(a[k],b[k])
assert b['skips'].sum()==0
for key in ['baseline','disabled']:
 meta=json.loads((S/f'{key}_r0.meta.json').read_text());assert meta['reader_bytes']==meta['linux_read_bytes']==int(a['reads'].sum())*4096;assert meta['swap_before_kib']==meta['swap_after_kib']==0
L=S/'L.txt';L.write_text('15 100\n')
with (S/'collect.log').open('w') as f:subprocess.run([str(R/'native/collect'),str(S/'ann'),str(S/'query.u8bin'),str(S/'query.gt'),str(model/'dummy.bin'),str(L),str(S/'trajectory'),'32'],stdout=f,stderr=subprocess.STDOUT,check=True)
p=np.fromfile(S/'trajectory.prune.bin','<f4').reshape(-1,20);nodes=np.fromfile(S/'trajectory.prune_nodes.u32','<u4');assert len(nodes)==len(p)>0 and np.all(nodes<20000) and np.all(p[:,1]==0) and np.all(p[:,13]==0)
assert set(p[:,18])=={15,100}
info=dict(passed=True,base_vectors=20000,queries=32,prune_rows=len(p),disabled_gate_identical_IDs_reads_expansions=True,reader_physical_bytes_equal=True,node_sidecar_aligned=True,query_dtype='uint8')
(R/'PREFLIGHT_COMPLETE.json').write_text(json.dumps(info,indent=2)+'\n');print(info,flush=True)
