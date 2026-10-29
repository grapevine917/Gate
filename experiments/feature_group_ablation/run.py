from common import *
import subprocess,sys,time,struct,joblib
from concurrent.futures import ThreadPoolExecutor
if not all((R/'models'/f'{g}_s{s}/COMPLETE.json').exists() for g in GROUPS for s in SEEDS):subprocess.run([sys.executable,str(R/'train.py')],check=True)
for split in ['validation','test']:assert np.array_equal(np.load(E1/'data'/f'{split}_query_ids.npy'),np.load(DATA/split/'query_ids.npy'))
proto=json.loads((R/'PROTOCOL.json').read_text());binary=E1/'native/collect';manifest=json.loads((DATA/'manifest.json').read_text())
# All learned-input groups use the same executable and graph, without history or secondary pruning.
base=dict(key='DiskANN',group='DiskANN',seed=-1,model='-',tau=0.,target_FNR=None);configs=[base]
for group in GROUPS:
 for seed in SEEDS:
  f=R/'models'/f'{group}_s{seed}'
  for t in json.loads((f/'thresholds.json').read_text()):configs.append(dict(key=f'{group}_s{seed}_{t["key"]}',group=group,seed=seed,model=str(f/'model.bin'),tau=t['tau'],target_FNR=t['target_FNR']))
dump(R/'configs.json',configs)
inputs={str(p):sha(p) for p in [binary,R/'PROTOCOL.json',DATA/'manifest.json',E1/'data/validation.fbin',E1/'data/validation.gt',E1/'data/test.fbin',E1/'data/test.gt',*[(R/'models'/f'{g}_s{s}/model.bin') for g in GROUPS for s in SEEDS]]}
if (R/'INPUT_LOCK.json').exists():assert json.loads((R/'INPUT_LOCK.json').read_text())==inputs,'Changed inputs require new directory'
else:dump(R/'INPUT_LOCK.json',inputs)
env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_PROC_BIND='true',OMP_PLACES='cores')
def execute(stage,cfg,split='test',mode='time'):
 directory=R/stage;directory.mkdir(exist_ok=True);p=directory/cfg['key'];done=Path(str(p)+'.done.json');cmd=[str(binary),str(S/'index'),str(E1/'data'/f'{split}.fbin'),str(E1/'data'/f'{split}.gt'),cfg['model'],str(cfg['tau']),'-',str(p),mode]
 lock=dict(argv=cmd,CPU_affinity=CPUS,env={k:env[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_PROC_BIND','OMP_PLACES']})
 if done.exists():assert json.loads(done.read_text())['command']==lock
 else:
  dump(Path(str(p)+'.command.json'),dict(**lock,launcher_argv=['taskset','-c',','.join(map(str,CPUS)),*cmd],utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));log('RUN '+stage+'/'+cfg['key'])
  with Path(str(p)+'.log').open('w') as f:subprocess.run(['taskset','-c',','.join(map(str,CPUS)),*cmd],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  dump(done,dict(command=lock,csv_sha256=sha(Path(str(p)+'.csv')),ids_sha256=sha(Path(str(p)+'.ids'))))
 a=np.genfromtxt(Path(str(p)+'.csv'),delimiter=',',names=True);assert len(a)==1000
 return p,a
# Zero-threshold audit checks native mappings and probabilities for all models, on live states.
bp,ba=execute('audit',base,mode='trace');audits={}
for cfg in configs:
 if cfg['group']=='DiskANN' or cfg['tau']!=0:continue
 p,a=execute('audit',cfg,mode='trace');assert sha(Path(str(p)+'.ids'))==sha(Path(str(bp)+'.ids'))
 for k in ['hits','reads','expansions','pq_comparisons']:assert np.array_equal(a[k],ba[k]),(cfg['key'],k)
 assert np.all(a['skips']==0);bundle=joblib.load(Path(cfg['model']).parent/'model.joblib');pieces=[]
 with Path(str(p)+'.trace').open('rb') as f:
  while header:=f.read(8):
   row,n=struct.unpack('<II',header);v=np.frombuffer(f.read(n*76),'<f4').reshape(n,19);take=(v[:,11]==0);pieces.append(v[take])
 trace=np.concatenate(pieces);idx=np.random.default_rng(cfg['seed']).choice(len(trace),min(4000,len(trace)),replace=False);z=np.array(trace[idx,1:18][:,bundle['feature_indices']],np.float32,order='C');z-=bundle['mean'];z/=bundle['scale'];error=float(np.max(np.abs(predict(bundle['model'],z)-trace[idx,18])));assert error<3e-6,error
 audits[cfg['key']]=dict(tau_zero_ids_counts_equal_DiskANN=True,live_score_sampled=len(idx),max_native_Python_error=error,mapped_feature_indices=bundle['feature_indices'])
dump(R/'native_audit.json',audits);log('ALL MODEL NATIVE AUDITS VERIFIED')
# Validation searches select an actual point under the same Recall constraint, not a label-FNR proxy.
val=[]
with ThreadPoolExecutor(max_workers=4) as pool:validation_outputs=list(pool.map(lambda c:execute('validation',c,'validation'),configs))
for cfg,(p,a) in zip(configs,validation_outputs):
 val.append(dict(**cfg,recall_pct=float(a['hits'].mean()*10),reads_per_query=float(a['reads'].mean()),expansions_per_query=float(a['expansions'].mean()),latency_us_single_pass=float(a['total_us'].mean())))
csvsave(R/'validation_summary.csv',val);selection=[]
for rmin in RMINS:
 for group in GROUPS:
  for seed in SEEDS:
   eligible=[v for v in val if v['group']==group and v['seed']==seed and v['recall_pct']>=rmin-1e-9];winner=min(eligible,key=lambda v:(v['reads_per_query'],v['tau'])) if eligible else None;selection.append(dict(R_min_pct=rmin,group=group,seed=seed,selected=winner))
lock=dict(primary_R_min_pct=PRIMARY,selection=selection,source='validation graph Recall/IO only; frozen before test and timing',configs_sha256=sha(R/'configs.json'))
if (R/'SELECTION_LOCK.json').exists():assert json.loads((R/'SELECTION_LOCK.json').read_text())==lock
else:dump(R/'SELECTION_LOCK.json',lock)
log('VALIDATION SELECTED POINTS FROZEN')
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda c:execute('test_curve',c),configs))
chosen=[base]+[x['selected'] for x in selection if x['R_min_pct']==PRIMARY and x['selected']]
# Preserve the exact source configs, stripping single-pass validation metrics.
bykey={c['key']:c for c in configs};chosen=[bykey[c['key']] for c in chosen];dump(R/'timing_configs.json',chosen)
# Profile is separate: inference/normalization region timers affect absolute search latency.
for cfg in chosen:execute('profile',cfg,mode='profile')
log('ALL TRAINING / AUDIT / CURVES / PROFILE FINISHED; START DEDICATED TIMING')
order=np.random.default_rng(20261024);timing_order=[]
for repeat in range(5):
 keys=[]
 for i in order.permutation(len(chosen)):
  cfg=chosen[i];keys.append(cfg['key']);execute(f'timing_r{repeat}',cfg)
 timing_order.append(dict(repeat=repeat,keys=keys));dump(R/'timing_order.json',timing_order)
log('DEDICATED TIMING COMPLETE')
subprocess.run([sys.executable,str(R/'collect.py')],check=True)
