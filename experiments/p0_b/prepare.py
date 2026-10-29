from pathlib import Path
import json,hashlib,subprocess,math
R=Path(__file__).resolve().parent;M=R.parent/'main_comparison';A=R.parent/'p0_a';S=R.parent/'sift1m'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
s=(A/'search.cpp').read_text().replace('diskann::OnlineGate state(argv[4],"",0);','diskann::OnlineGate state(argv[4],"",0),et(argv[5],"",1.1);').replace('nullptr,&state,nullptr,nullptr,nullptr','nullptr,&state,nullptr,nullptr,&state').replace('if(gate)gate->threshold=tau;','if(gate)gate->threshold=tau;et.threshold=et_tau;').replace('diskann::online_termination=nullptr;diskann::online_miss_only=true;','diskann::online_termination=(kind==6||kind==7)?&et:nullptr;diskann::online_miss_only=true;')
(R/'search.cpp').write_text(s)
c=json.loads((A/'build.command.json').read_text());c=[str(R/'search.cpp') if x==str(A/'search.cpp') else str(R/'search') if x==str(A/'search') else x for x in c];dump(R/'build.command.json',c)
with (R/'build.log').open('w') as f:subprocess.run(c,stdout=f,stderr=subprocess.STDOUT,check=True)
LS=[15,20,30,50,75,100,160,300];baseLs=set(LS)
while len(baseLs)<64:
 v=max((v for v in range(10,301) if v not in baseLs),key=lambda v:min(abs(math.log(v/b)) for b in baseLs));baseLs.add(v)
t=json.loads((S/'online_state_mlp/thresholds.json').read_text());taus=[0]+[t['fnr_'+k+'pct']['threshold'] for k in ['0.1','0.25','0.5','1','2']]+[.15,.3]
ets=[1.1,.999,.995,.99,.98,.95,.9,.8]
heur=[(1,-1),(.98,1),(.92,1.04),(.72,1),(.64,.92),(.5,.9),(.25,.75),(.1,.5)]
pairs=[(0,0),(4,0),(0,3),(2,1),(3,2),(4,3),(5,4),(6,5)]
configs=[]
def cfg(f,kind,L,j=0,tau=0,et=1.1,rank=0,exp=-1):return dict(key=f'{f}_L{L}_p{j}',family=f,kind=kind,L=L,preset=j,tau=tau,et_tau=et,rank=rank,exp=exp)
for L in sorted(baseLs):configs.append(cfg('Tuned_DiskANN',0,L))
for fam,kind in [('Heuristic',1),('Ours',4),('ET',6),('ET_Ours',7)]:
 for L in LS:
  for j in range(8):
   kw={}
   if kind==1:kw=dict(rank=int(L*heur[j][0]),exp=-1 if heur[j][1]<0 else int(L*heur[j][1]))
   if kind==4:kw=dict(tau=taus[j])
   if kind==6:kw=dict(et=ets[j])
   if kind==7:pi,ei=pairs[j];kw=dict(tau=taus[pi],et=ets[ei])
   configs.append(cfg(fam,kind,L,j,**kw))
assert len(configs)==320;dump(R/'configs.json',configs)
p=dict(dataset='SIFT1M',index=str(S/'index'),K=10,beam=4,threads=4,cache_nodes=10000,MKL_threads=1,query_counts=dict(validation=1000,test=1000),query_dir=str(M/'data'),gate_model=str(S/'online_state_mlp/model.bin'),gate='original frozen17D MLP from P0-A; L100-trained; no retraining',et_model=str(M/'models/et.bin'),ET='Existing mixed-L17D MLP predicting prefix Recall; native DiskANN beam-boundary stop after K exact candidates. Not faithful DARTH/LAET reproduction; trained on unpruned trajectories.',L_grid=LS,baseline_L=sorted(baseLs),configs_per_family=64,gate_thresholds=taus,et_thresholds=ets,heuristic_scaled_pairs=heur,joint_threshold_index_pairs=pairs,seed=20261101,targets=[90,95,98,99,99.5,99.7],max_L=300,selection='Freeze all grids before evaluation; choose per target and objective using validation only and lock before test; test full fixed grid descriptive empirical frontiers; report actual test Recall explicitly, not equate same validation bound with same test Recall.',frontier='Separate empirical nondominated P95/QPS/physical_read_bytes frontiers on actual test Recall; only measured markers, lines visual guides; no latency or throughput interpolation.',exact_recall='Also tabulate cost at identical integer total_hits across families; disabled-action configs allowed and flagged; may be dominated by higher-Recall configs.',timing='All validation configs once; all test configs once; repeat union of test per-objective frontiers, validation-selected configs and exact-shared-Recall metric winners to3 runs; iterate until final selections all3. All inference/feature/IO/search included; load, warmup, GT and output excluded.',warmup_queries=128,io='Meter actual direct read requests/lengths and /proc/self/io read_bytes delta for search window; WSL2 guest block traffic, not NAND telemetry. Primary physical byte metric is measured Linux read_bytes, never inferred from requests.',cache_hits_protected=True,limits=['Previously observed evaluation queries; exploratory','Original Gate trained only L100; fixed weights across L, extrapolation measured live','ET+Ours faces trajectory shift','Equal finite config count is not global optimum; joint uses8 predeclared threshold pairs, not full Cartesian product','Test empirical frontier is descriptive, not unbiased deployment tuning','Timing frontier selection on finite repeated observations has selection noise'])
dump(R/'PROTOCOL.json',p)
inputs=[R/'configs.json',R/'PROTOCOL.json',R/'search.cpp',R/'search',M/'native/libdiskann.a',S/'online_state_mlp/model.bin',M/'models/et.bin',S/'index_disk.index',S/'index_pq_pivots.bin',S/'index_pq_compressed.bin']+[M/f'data/{split}.{ext}' for split in ['validation','test'] for ext in ['fbin','gt']]
dump(R/'INPUT_LOCK.json',{str(x):sha(x) for x in inputs});print('P0-B BUILD/GRID LOCKED: 5 families x64',flush=True)
