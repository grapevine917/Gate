from common import *
import subprocess,sys
for script,marker in [('prepare.py','PREPARED.json'),('build.py','native/search')]:
 if not (R/marker).exists():subprocess.run([sys.executable,str(R/script)],check=True)
proto=json.loads((R/'PROTOCOL.json').read_text());configs=json.loads((R/'configs.json').read_text())
paths=[R/'native/search',R/'native/libdiskann.a',R/'native/policy.h',R/'policy.h',R/'search.cpp',R/'risk_models.bin',R/'risk_models.json',R/'history.bin',R/'partitions.txt',R/'configs.txt',R/'PROTOCOL.json',E1/'data/test.fbin',E1/'data/test.gt',E1/'PROVENANCE.json',R.parent/'sift1m/online_state_mlp/model.bin'];inputs={str(p):sha(p) for p in paths}
if (R/'INPUT_LOCK.json').exists():assert json.loads((R/'INPUT_LOCK.json').read_text())==inputs,'Changed locked inputs: new output directory required'
else:dump(R/'INPUT_LOCK.json',inputs)
def run(tag,order,cf=None,cold=False):
 (R/'graph'/tag).mkdir(parents=True,exist_ok=True);cf=cf or R/'configs.txt'
 cmd=[str(R/'native/search'),str(R.parent/'sift1m/index'),str(E1/'data/test.fbin'),str(E1/'data/test.gt'),str(R.parent/'sift1m/online_state_mlp/model.bin'),str(proto['original_tau']),str(R/'history.bin'),str(R/'risk_models.bin'),str(R/'partitions.txt'),str(cf),str(R/'orders'/order),tag,str(int(cold)),str(json.loads((E2/'PROTOCOL.json').read_text())['state_high_if_normalized_l2_le'])]
 dump(R/(tag+'.command.json'),dict(argv=cmd,cwd=str(R),env={k:os.environ[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),order_sha256=sha(R/'orders'/order),configs_sha256=sha(cf)))
 log('START '+tag)
 with (R/(tag+'.log')).open('a') as f:subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
run('audit','audit.txt')
a=dict(np.load(E42/'features.npz'));p=dict(np.load(E41/'keep_positions.npz'));mods=json.loads((R/'risk_models.json').read_text());dt=np.dtype([('row','<u4'),('node','<u4'),('ordinal','<u4'),('pad','<u4'),('x','<f8',(28,)),('logits','<f8',(2,))]);z=np.fromfile(R/'graph/audit/MLP.audit.bin',dt);assert len(z)==len(a['U'])
key={(int(r),int(n),int(s)):i for i,(r,n,s) in enumerate(zip(a['query_row'],p['node_id'],p['search_step']))};ix=np.array([key[(int(v['row']),int(v['node']),int(v['ordinal']))] for v in z]);assert len(np.unique(ix))==len(ix)
x=a['x'].copy();x[:,[23,24,26,27]]=0;errors=np.max(np.abs(z['x']-x[ix]),axis=0);assert errors.max()<1e-7,errors
pred=np.column_stack([x@np.array(m['weights'])+m['intercept'] for m in mods]);score_errors=np.max(np.abs(z['logits']-pred[ix]),axis=0);assert score_errors.max()<2e-6,score_errors
assert sha(R/'graph/audit/MLP.ids')==sha(E41/'intervention.baseline.ids');assert sha(R/'graph/audit/MLP.cache.bin')==sha(E1/'runs_main/test_MLP_trace.cache.bin')
b=np.genfromtxt(R/'graph/audit/MLP.csv',delimiter=',',names=True);e1=np.genfromtxt(E1/'runs_main/test_MLP_r0_time.csv',delimiter=',',names=True)
for col,orig in [('hits','hits'),('reads','reads'),('expanded','expansions'),('neighbor_cmps','pq_comparisons'),('model_calls','model_calls'),('original_skips','skips')]:assert np.array_equal(b[col],e1[orig]),col
assert np.all(b['secondary_skips']==0)
dump(R/'native_parity.json',dict(all_93410_keep_features_and_risk_logits_verified=True,feature_errors=dict(zip(FEATURES,errors.tolist())),risk_logit_errors=score_errors.tolist(),baseline_ids_reads_expansion_comparisons_model_calls_skips_equal_e1=True,cache_equal_e1=True))
log('NATIVE AUDIT VERIFIED')
run('validation','validation.txt')
val=[]
for c in configs:
 g=np.genfromtxt(R/'graph/validation'/f'{c["key"]}.csv',delimiter=',',names=True);val.append(dict(key=c['key'],family=c['family'],kind=c['kind'],threshold=c['threshold'],recall_pct=float(g['hits'].mean()*10),reads_per_query=float(g['reads'].mean()),secondary_skips_per_query=float(g['secondary_skips'].mean()),risk_calls_per_query=float(g['risk_LR_calls'].mean())))
selection=[];families=list(dict.fromkeys(c['family'] for c in configs));base=next(v for v in val if v['key']=='MLP')
for rmin in proto['R_min_sensitivity_pct']:
 for family in families:
  choices=[v for v in val if (v['family']==family or v['key']=='MLP') and v['recall_pct']>=rmin-1e-9]
  winner=min(choices,key=lambda v:(v['reads_per_query'],-v['recall_pct'],v['threshold'])) if choices else None
  selection.append(dict(R_min_pct=rmin,family=family,feasible_on_validation=winner is not None,selected=winner))
locked=dict(primary_R_min_pct=proto['primary_R_min_pct'],selection=selection,source='validation sequential dynamic graph runs only; absolute Recall constraints',configs_sha256=sha(R/'configs.json'),risk_models_sha256=sha(R/'risk_models.json'))
if (R/'SELECTION_LOCK.json').exists():assert json.loads((R/'SELECTION_LOCK.json').read_text())==locked
else:dump(R/'SELECTION_LOCK.json',locked)
csvsave(R/'validation_summary.csv',val);log('VALIDATION ABSOLUTE-RECALL SELECTION LOCKED')
run('test','test.txt')
# Frozen selected points plus unconditional mark and baseline, without test-driven choice.
keys={'MLP','History_unconditional'}
for item in selection:
 if item['R_min_pct']==proto['primary_R_min_pct'] and item['selected']:keys.add(item['selected']['key'])
chosen=[c for c in configs if c['key'] in keys]
with (R/'robustness_configs.txt').open('w') as f:
 f.write(str(len(chosen))+'\n')
 for c in chosen:f.write(f'{c["key"]} {c["kind"]} {c["threshold"]:.17g}\n')
dump(R/'robustness_configs.json',chosen)
for order in range(3):
 for cold in [False,True]:run(f'test_shuffle{order}_{"cold" if cold else "warm"}',f'test_shuffle{order}.txt',R/'robustness_configs.txt',cold)
# Exact deterministic replay of selected policies and their history evolution.
run('test_repeat','test.txt',R/'robustness_configs.txt')
subprocess.run([sys.executable,str(R/'collect.py')],check=True)
