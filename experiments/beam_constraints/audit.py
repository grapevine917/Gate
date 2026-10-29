from pathlib import Path
import subprocess,json,struct
import numpy as np
R=Path(__file__).resolve().parent;LG=R.parents[1];out={}
for policy in range(4):
 prefix=R/'runs'/f'audit_p{policy}'
 args=[str(R/'native/collect'),str(LG/'experiments/sift1m/index'),str(R/'data/test.fbin'),str(R/'data/test.gt'),str(LG/'experiments/sift1m/online_state_mlp/model.bin'),'1.01',str(policy),str(prefix),'trace']
 with prefix.with_suffix('.log').open('w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
 stats=np.genfromtxt(prefix.with_suffix('.csv'),delimiter=',',names=True)
 total_beams=0;total_skips=0
 with prefix.with_suffix('.trace').open('rb') as f:
  for qi in range(300):
   row,n=struct.unpack('<II',f.read(8));assert row==qi
   a=np.frombuffer(f.read(n*19*4),'<f4').reshape(n,19);skips=0
   for hop in np.unique(a[:,9]):
    group=a[a[:,9]==hop];assert len(group)<=4
    scores=group[:,18];cached=group[:,11]!=0
    raw=(scores<1.01)&~cached
    if policy in [1,2]:
     eff=raw.copy();eff[np.flatnonzero(raw)[policy:]]=False
     assert eff.sum()<=policy
    elif policy==3:
     eff=raw.copy()
     if not cached.any():eff[np.argmax(scores)]=False
     assert (~eff).sum()>=1
    else:eff=raw
    skips+=int(eff.sum());total_beams+=1
   assert skips==stats[qi]['skips']
   assert n==stats[qi]['expansions']+stats[qi]['skips']
   assert (~(a[:,11]!=0)).sum()==stats[qi]['model_calls']
   total_skips+=skips
  assert not f.read(1)
 timed=R/'runs'/f'test_p{policy}_t1p01_r0.ids'
 assert prefix.with_suffix('.ids').read_bytes()==timed.read_bytes()
 out[str(policy)]=dict(queries=300,beams=total_beams,skips=total_skips,effective_skip_counts_match=True,constraints_verified=True,one_score_per_cache_miss=True,trace_time_ids_identical=True)
(R/'constraint_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print('constraint audit passed')
