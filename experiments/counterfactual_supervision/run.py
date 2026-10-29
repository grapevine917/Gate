import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import numpy as np
import json,subprocess,random,hashlib,csv
R=Path(__file__).resolve().parent;LG=R.parents[1];S=R.parent/'sift1m';E1=R.parent/'e1_similarity';D=R/'data';RUN=R/'runs';RUN.mkdir(exist_ok=True)
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvsave(p,rows):
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def loadcsv(prefix):return np.atleast_1d(np.genfromtxt(str(prefix)+'.csv',delimiter=',',names=True))
original_taus=[v['threshold'] for v in json.loads((S/'online_state_mlp/thresholds.json').read_text()).values()]
taus=sorted(set([0.]+list(np.geomspace(.0001,.1,13))+list(np.arange(.11,.501,.01))+[.6,.7,.85,1.01]+original_taus))
configs=[dict(key=f'MLP_t{i:03}',family='MLP',tau=float(t),rho=0.,gmin=0.) for i,t in enumerate(taus)]
for i,rho in enumerate([0.,.0001,.0003,.001,.003,.01,.03,.1,.3,1.01]):
 for j,g in enumerate([-1e9,0.,.5,1.,1.5]):configs.append(dict(key=f'CF_r{i:02}_g{j}',family='CF',tau=.5,rho=rho,gmin=g))
dump(R/'configs.json',configs)
locked=json.loads((R/'INPUT_LOCK.json').read_text());assert all(sha(Path(p))==h for p,h in locked.items())
current={'model':sha(R/'model.bin'),'binary':sha(R/'native/collect'),'configs':sha(R/'configs.json'),'protocol':sha(R/'PROTOCOL.json')}
if (R/'RUN_LOCK.json').exists():assert json.loads((R/'RUN_LOCK.json').read_text())==current
else:dump(R/'RUN_LOCK.json',current)
verification={}
p=subprocess.run([str(R/'native/verify'),str(R/'model.bin'),str(R/'native_check.bin')],text=True,capture_output=True,check=True);verification['native_vs_pytorch']=json.loads(p.stdout)
def run(split,c,rep=0,reference=False):
 prefix=RUN/f'{split}_{c["key"]}_r{rep}{"_ref" if reference else ""}'
 done=Path(str(prefix)+'.done.json')
 if not done.exists():
  args=[str(R/'native/collect'),str(S/'index'),str(D/f'{split}.fbin'),str(D/f'{split}.gt'),str(S/'online_state_mlp/model.bin'),str(c['tau']),str(R/'model.bin') if c['family']=='CF' else '-',str(prefix),'time',str(c['rho']),str(c['gmin'])]
  if reference:args=[str(E1/'native/collect'),*args[1:9]]
  dump(Path(str(prefix)+'.command.json'),dict(argv=args,thread_env={k:os.environ[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}))
  with open(str(prefix)+'.log','w') as out:subprocess.run(args,stdout=out,stderr=subprocess.STDOUT,check=True)
  a=loadcsv(prefix);assert len(a)==250 and np.all(np.isfinite(a['total_us']))
  assert np.all(a['reads']<=a['expansions'])
  ids=np.fromfile(str(prefix)+'.ids','<u8').reshape(250,10)
  with (D/f'{split}.gt').open('rb') as f:n,d=np.fromfile(f,'<u4',2);gt=np.fromfile(f,'<u4').reshape(n,d)[:,:10]
  # Count unique retrieved IDs, independently recompute Recall from saved IDs and ground truth.
  hits=np.array([len(set(row)&set(truth)) for row,truth in zip(ids,gt)]);np.testing.assert_array_equal(hits,a['hits'])
  row=dict(**c,split=split,repeat=rep,recall=float(hits.mean()*10),reads=float(a['reads'].mean()),expansions=float(a['expansions'].mean()),skips=float(a['skips'].mean()),model_calls=float(a['model_calls'].mean()),latency_us=float(a['total_us'].mean()),p95_us=float(np.percentile(a['total_us'],95)),prefix=str(prefix),csv_sha256=sha(Path(str(prefix)+'.csv')),ids_sha256=sha(Path(str(prefix)+'.ids')))
  dump(done,row)
 return json.loads(done.read_text())
# Parity across aggressive and mild thresholds, including all-keep.
for tau in [0.,original_taus[1],.27]:
 c=dict(key='parity_'+str(tau).replace('.','p'),family='MLP',tau=tau,rho=0.,gmin=0.)
 a=run('validation',c);b=run('validation',c,reference=True)
 assert a['ids_sha256']==b['ids_sha256']
 for key in ['recall','reads','expansions','skips','model_calls']:assert a[key]==b[key]
 verification['MLP_'+str(tau)]='IDs and counts exactly match original E1 native program'
 if tau==0:zero=a
c=dict(key='CF_no_prune',family='CF',tau=.5,rho=0.,gmin=-1e9);a=run('validation',c);assert a['ids_sha256']==zero['ids_sha256']
for key in ['recall','reads','expansions','skips','model_calls']:assert a[key]==zero[key]
verification['CF_no_prune']='identical IDs and counts to unchanged DiskANN/MLP tau0'
dump(R/'verification.json',verification)
# Validation only. Identical queries and graph parameters for both families.
order=list(configs);random.Random(20261024).shuffle(order);validation=[]
for i,c in enumerate(order):
 a=run('validation',c);validation.append(a);print('VAL',i+1,len(order),c['key'],a['recall'],a['reads'],flush=True)
dump(R/'validation.json',validation);csvsave(R/'validation_summary.csv',[{k:v for k,v in a.items() if k not in ['csv_sha256','ids_sha256']} for a in validation])
selected=[]
for target in json.loads((R/'PROTOCOL.json').read_text())['recall_targets']:
 for family in ['MLP','CF']:
  feasible=[a for a in validation if a['family']==family and a['recall']+1e-10>=target]
  if not feasible:selected.append(dict(target=target,family=family,key=None));continue
  best=min(feasible,key=lambda a:(a['reads'],a['expansions'],a['key']))
  selected.append(dict(target=target,family=family,key=best['key'],validation_recall=best['recall'],validation_reads=best['reads']))
selection=dict(test_used=False,selected=selected,configs_sha256=sha(R/'configs.json'),model_sha256=sha(R/'model.bin'))
if (R/'SELECTION_LOCK.json').exists():assert json.loads((R/'SELECTION_LOCK.json').read_text())==selection
else:dump(R/'SELECTION_LOCK.json',selection)
# Freeze before first test call; no test-driven reselection or threshold interpolation.
chosen=[c for c in configs if c['key'] in {a['key'] for a in selected}];test=[]
for rep in range(3):
 order=list(chosen);random.Random(20261024+rep).shuffle(order)
 for c in order:
  a=run('test',c,rep);test.append(a);print('TEST',rep,c['key'],a['recall'],a['reads'],a['latency_us'],flush=True)
dump(R/'test.json',test)
summary=[]
for s in selected:
 if s['key'] is None:summary.append(dict(**s,test_recall=None,reads=None,expansions=None,latency_us=None,latency_min=None,latency_max=None,test_meets_target=False));continue
 rs=[a for a in test if a['key']==s['key']];assert len(rs)==3
 for k in ['ids_sha256','recall','reads','expansions','skips','model_calls']:assert len({a[k] for a in rs})==1
 summary.append(dict(**s,test_recall=rs[0]['recall'],reads=rs[0]['reads'],expansions=rs[0]['expansions'],latency_us=float(np.median([a['latency_us'] for a in rs])),latency_min=min(a['latency_us'] for a in rs),latency_max=max(a['latency_us'] for a in rs),test_meets_target=rs[0]['recall']+1e-10>=s['target']))
dump(R/'selected_results.json',summary);csvsave(R/'selected_results.csv',summary)
# Paired query bootstrap: selected policies on identical held-out queries.
bootstrap=np.random.default_rng(20261024).integers(0,250,(2000,250));comparisons=[]
for target in json.loads((R/'PROTOCOL.json').read_text())['recall_targets']:
 pair={r['family']:r for r in summary if r['target']==target}
 if any(pair[f]['key'] is None for f in ['MLP','CF']):continue
 a=loadcsv(next(r['prefix'] for r in test if r['key']==pair['MLP']['key']));b=loadcsv(next(r['prefix'] for r in test if r['key']==pair['CF']['key']))
 row=dict(target=target,baseline=pair['MLP']['key'],counterfactual=pair['CF']['key'],both_test_feasible=bool(pair['MLP']['test_meets_target'] and pair['CF']['test_meets_target']))
 for label,delta in [('recall_pp',(b['hits']-a['hits'])*10),('reads',b['reads']-a['reads']),('expansions',b['expansions']-a['expansions'])]:
  lo,hi=np.percentile(delta[bootstrap].mean(1),[2.5,97.5]);row[label+'_delta']=float(delta.mean());row[label+'_ci95']=[float(lo),float(hi)]
 comparisons.append(row)
dump(R/'paired_comparisons.json',comparisons)
assert all(sha(Path(p))==h for p,h in locked.items());dump(R/'ONLINE_COMPLETE.json',dict(validation_configs=len(validation),test_configs=len(chosen),test_runs=len(test),original_inputs_unchanged=True))
print('ONLINE COMPLETE',flush=True)
