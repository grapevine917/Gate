"""Prepare isolated DEEP1M P0-A/B/C models, queries and native code."""
from pathlib import Path
import json,hashlib,struct,shutil,subprocess
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parents[1];M=R.parent/'main_comparison';D=R.parent/'deep1m';N=R/'native';SEED=20261030;LS=[15,20,30,50,75,100,160,300]
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
P=dict(dataset='DEEP1M',base_shape=[1000000,96],index=str(D/'index'),index_R=64,index_Lbuild=100,PQ_bytes=34,K=10,beam=4,threads=4,MKL_threads=1,cache_nodes=10000,train_queries=1200,internal_holdout_queries=200,validation_queries=1000,test_queries=1000,L_grid=LS,targets=[90,95,98,99,99.5,99.7],seed=SEED,training='DEEP-only weights; same1200 training queries for all; matched mixed-L LR/XGB/stateMLP/qvMLP use same cache-miss sample rows and contribution proxy labels. Separate L100 MLP for original P0-A/B, using all L100 cache-miss rows. ET predicts actual prefix GT recall on mixed-L unpruned trajectories.',neural_training='160 inner epochs, fixed LR .001 through40/.0003 through80/.0001 through120/.00003 through160; choose best internal query loss then refit all1200. Single fixed seed; same17-128-64-1 hidden architecture; qv input401. CPU denormal flushing enabled for numerical underflow performance.',P0_A='L100 beam4; Vanilla, fixed rank>72 and completed>100 heuristic, original L100 MLP, mixed-L LR; each9 repeats; MLP/LR validation retain-FNR1% calibration, not final Recall guarantee.',P0_B='Original L100 MLP and mixed-L LR versions; baseline64 L, heuristic64, ET64, each Gate64 and joint64; joint8 predeclared pairs, not full Cartesian. Shared identical controls measured once, same frozen weights per version.',P0_C='Baseline + matched mixed-L LR/XGB/17DMLP/401DqvMLP,64 configs each; classification and native score diagnostics on same L100 held-out candidate rows.',evaluation='Freeze grid, run full validation; lock per quality/objective before test. Full fixed test grid is descriptive; repeat union of per-objective frontiers, validation-locked choices, exact-common-recall winners and feasible-quality winners to3 until stable. No cost interpolation.',physical_io='Native O_DIRECT request count/bytes and /proc/self/io search-window read_bytes measured. WSL2 guest block bytes, not NAND telemetry.',limits=['Previously observed DEEP benchmark queries; exploratory','Finite grid and8 joint pairs; no exhaustive policy-optimality claim','DEEP neural training schedule extended relative to original SIFT P0; not a pure dataset-only change','L100 MLP and mixed-L LR differ in training distribution; pure matched ablation is P0-C','Contribution proxy labels are not counterfactual final-Recall harm labels','ET uses unpruned trajectories; joint can face trajectory shift','Single training seed; repeated search timings do not include training variability'])
if (R/'PROTOCOL.json').exists():assert json.loads((R/'PROTOCOL.json').read_text())==P
else:dump(R/'PROTOCOL.json',P)
data_check=json.loads((D/'data_validation.json').read_text());assert data_check['passed']
for name,h in data_check['sha256'].items():assert sha(ROOT/'data'/name)==h
ids={s:np.load(D/'queries'/s/'query_ids.npy') for s in ['train','validation','test']};ids['train']=np.sort(np.random.default_rng(SEED).choice(ids['train'],1200,replace=False));hold=np.random.default_rng(SEED+1).choice(ids['train'],200,replace=False)
assert all(not set(ids[a])&set(ids[b]) for a,b in [('train','validation'),('train','test'),('validation','test')])
np.save(R/'data/internal_holdout_ids.npy',hold)
with (ROOT/'data/deep1M_query.fbin').open('rb') as f:n,d=np.fromfile(f,'<u4',2);q=np.fromfile(f,'<f4').reshape(n,d)
gt=np.fromfile(ROOT/'data/deep1M_groundtruth.ivecs','<i4').reshape(10000,101);assert np.all(gt[:,0]==100);gt=gt[:,1:].astype('<u4')
for split,rows in ids.items():
 np.save(R/'data'/f'{split}_ids.npy',rows)
 for a,ext in [(q,'fbin'),(gt,'gt')]:
  path=R/'data'/f'{split}.{ext}'
  with path.open('wb') as f:np.array([len(rows),a.shape[1]],'<u4').tofile(f);a[rows].tofile(f)
dump(R/'splits.json',{k:v.tolist() for k,v in ids.items()}|{'internal_holdout':hold.tolist()});dump(R/'data_verification.json',dict(passed=True,headers=[1000000,96],original_exact_GT_audit=data_check,query_disjoint=True))
(R/'L.txt').write_text(' '.join(map(str,LS)));(R/'L100.txt').write_text('100\n')
# A zero-threshold, constant-score dummy exposes all baseline states; it never prunes.
with (R/'models/dummy.bin').open('wb') as f:
 f.write(b'STGATE01');np.zeros(17,'<f4').tofile(f);np.ones(17,'<f4').tofile(f)
 for a,b in [(17,128),(128,64),(64,1)]:np.zeros(a*b+b,'<f4').tofile(f)
header=(M/'native/online_gate.h').read_text().replace('input_width=529','input_width=401').replace('state_only?17:529','state_only?17:401').replace('M3GATE01','D3GATE01').replace('d!=128','d!=96').replace('128d vectors','96d vectors').replace('std::array<float,529> x','std::array<float,401> x').replace('vectors.data()+size_t(node)*128','vectors.data()+size_t(node)*96').replace('for(int i=0;i<128;++i){x[i]=q[i];x[128+i]=v[i];x[256+i]=std::abs(q[i]-v[i]);x[384+i]=q[i]*v[i];}','for(int i=0;i<96;++i){x[i]=q[i];x[96+i]=v[i];x[192+i]=std::abs(q[i]-v[i]);x[288+i]=q[i]*v[i];}').replace('x[512+i]=state[i]','x[384+i]=state[i]')
(N/'online_gate.h').write_text(header);shutil.copy(M/'native/online_termination.h',N/'online_termination.h');shutil.copy(M/'native/pq_flash_index.cpp',N/'pq_flash_index.cpp')
(R/'collect.cpp').write_text((M/'collect.cpp').read_text().replace('if(audit[j*19+11]==0)ix.push_back(j);','if(cap>32 || audit[j*19+11]==0)ix.push_back(j);'))
source=(R.parent/'p0_a/search.cpp').read_text();a=source.index(' diskann::OnlineGate state(');b=source.index(' std::ifstream configs',a)
source=source[:a]+''' std::ifstream probe(argv[6]);std::string first_key;int first_kind;probe>>first_key>>first_kind;
 const char*names[]={"","","lr","xgb","l100","state","qv","","l100","lr"};
 std::unique_ptr<diskann::OnlineGate> owned,et;
 if(first_kind>=2 && first_kind!=7)owned=std::make_unique<diskann::OnlineGate>(std::string(argv[4])+"/"+names[first_kind]+".bin",first_kind==6?argv[5]:"",0);
 if(first_kind>=7)et=std::make_unique<diskann::OnlineGate>(std::string(argv[4])+"/et.bin","",1.1);
'''+source[b:]
source=source.replace('auto*gate=models[kind];','if(kind!=first_kind)throw std::runtime_error("one family per process");auto*gate=owned.get();').replace('if(gate)gate->threshold=tau;','if(gate)gate->threshold=tau;if(et)et->threshold=et_tau;').replace('diskann::online_termination=nullptr;diskann::online_miss_only=true;','diskann::online_termination=et.get();diskann::online_miss_only=true;')
(R/'search.cpp').write_text(source)
verify=(M/'verify.cpp').read_text().replace('float row[150]','float row[118]').replace('row[128]','row[96]').replace('row+129','row+97').replace('row[146+k]','row[114+k]').replace('q[128]','q[96]')
(R/'verify.cpp').write_text(verify)
score=(R.parent/'p0_c/score.cpp').read_text().replace('147','115').replace('row[128]','row[96]').replace('r[128]','r[96]').replace('r+129','r+97')
(R/'score.cpp').write_text(score)
template=json.loads((R.parent/'p0_b_lr/build.command.json').read_text());source_i=template.index(str(R.parent/'p0_b_lr/search.cpp'));prefix=[x.replace(str(M/'native'),str(N)) for x in template[:source_i]];libs=[x.replace(str(M/'native'),str(N)) for x in template[template.index('-L/opt/intel/oneapi/mkl/latest/lib/intel64'):]];cmds=[]
def build(cmd,name):
 cmds.append(cmd);dump(R/'build.commands.json',cmds)
 with (R/name).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
if not (N/'libdiskann.a').exists():
 build([*prefix,'-c',str(N/'pq_flash_index.cpp'),'-o',str(N/'pq_flash_index.cpp.o')],'build_index.log');shutil.copy(M/'native/libdiskann.a',N/'libdiskann.a');build(['ar','r',str(N/'libdiskann.a'),str(N/'pq_flash_index.cpp.o')],'build_archive.log')
for name in ['collect','search','verify','score']:
 if not (N/name).exists() or (N/name).stat().st_mtime<(R/f'{name}.cpp').stat().st_mtime:build([*prefix,str(R/f'{name}.cpp'),'-o',str(N/name),*libs],f'build_{name}.log')
inputs=[ROOT/'data/deep1M_base.fbin',ROOT/'data/deep1M_query.fbin',ROOT/'data/deep1M_groundtruth.ivecs',*[D/f'index{s}' for s in ['_disk.index','_pq_pivots.bin','_pq_compressed.bin']],*[R/'data'/f'{s}.{ext}' for s in ids for ext in ['fbin','gt']],N/'libdiskann.a',N/'search',N/'collect',N/'online_gate.h',R/'PROTOCOL.json']
dump(R/'INPUT_LOCK.json',{str(p):sha(p) for p in inputs});print('DEEP P0 PREPARED',flush=True)
