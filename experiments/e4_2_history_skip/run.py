from common import *
import subprocess,sys
if not (R/'PREPARED.json').exists():subprocess.run([sys.executable,str(R/'prepare.py')],check=True)
if not (R/'ablations.json').exists():subprocess.run([sys.executable,str(R/'ablations.py')],check=True)
if not (R/'native/search').exists():subprocess.run([sys.executable,str(R/'build.py')],check=True)
(R/'graph').mkdir(exist_ok=True)
paths=[R/'native/search',R/'native/libdiskann.a',R/'policy.h',R/'search.cpp',R/'models.bin',R/'history.bin',R/'query_distances.bin',R/'partitions.txt',R/'configs.txt',R/'PROTOCOL.json',E1/'data/test.fbin',E1/'data/test.gt',R.parent/'sift1m/online_state_mlp/model.bin'];inputs={str(p):sha(p) for p in paths}
if (R/'INPUT_LOCK.json').exists():assert json.loads((R/'INPUT_LOCK.json').read_text())==inputs,'Changed native inputs; use new directory'
else:dump(R/'INPUT_LOCK.json',inputs)
protocol=json.loads((R/'PROTOCOL.json').read_text());tau=json.loads((E1/'SELECTION_LOCK.json').read_text())['selected']['MLP']
def run(phase):
 cmd=[str(R/'native/search'),str(R.parent/'sift1m/index'),str(E1/'data/test.fbin'),str(E1/'data/test.gt'),str(R.parent/'sift1m/online_state_mlp/model.bin'),str(tau),str(R/'history.bin'),str(R/'models.bin'),str(R/'query_distances.bin'),str(R/'partitions.txt'),str(R/'configs.txt'),phase,str(protocol['state_distance_cutoff']),str(protocol['query_distance_cutoff'])]
 dump(R/(phase+'.command.json'),dict(argv=cmd,cwd=str(R),env={k:os.environ[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));log('START GRAPH '+phase)
 with (R/(phase+'.log')).open('a') as f:subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
run('baseline')
# Full native/Python feature and score parity on unmodified paths, before any graph-policy experiment.
p=dict(np.load(E4/'keep_positions.npz'));a=load();scores=np.load(R/'scores.npy');dt=np.dtype([('row','<u4'),('node','<u4'),('ordinal','<u4'),('pad','<u4'),('x','<f8',(28,)),('scores','<f8',(7,))]);audit=np.fromfile(R/'graph/baseline.baseline.audit.bin',dt);assert len(audit)==len(a['U'])
keys={(int(q),int(node),int(step)):i for i,(q,node,step) in enumerate(zip(a['query_row'],p['node_id'],p['search_step']))};ix=np.array([keys[(int(v['row']),int(v['node']),int(v['ordinal']))] for v in audit]);assert len(np.unique(ix))==len(ix)
max_features=np.max(np.abs(audit['x']-a['x'][ix]),axis=0);max_scores=np.max(np.abs(audit['scores']-scores[ix]),axis=0);assert np.max(max_features)<1e-7,(max_features,FEATURES);assert np.max(max_scores)<2e-6,max_scores
assert sha(R/'graph/baseline.baseline.ids')==sha(E4/'intervention.baseline.ids');assert sha(R/'graph/baseline.baseline.cache.bin')==sha(E1/'runs_main/test_MLP_trace.cache.bin')
base=np.genfromtxt(R/'graph/baseline.baseline.csv',delimiter=',',names=True);orig=np.genfromtxt(E1/'runs_main/test_MLP_r0_time.csv',delimiter=',',names=True)
for x,y in [('hits','hits'),('reads','reads'),('expanded','expansions'),('neighbor_cmps','pq_comparisons')]:assert np.array_equal(base[x],orig[y]),x
for row in range(1000):assert base['live_keeps'][row]==np.sum(a['query_row']==row)
dump(R/'native_parity.json',dict(all_93410_positions_audited=True,max_feature_errors=dict(zip(FEATURES,max_features.tolist())),max_score_errors=dict(zip(METHODS,max_scores.tolist())),baseline_ids_reads_expanded_neighbor_cmps_equal_e1=True,cache_equal_e1=True))
log('NATIVE BASELINE / ALL FEATURES / ALL SCORES VERIFIED')
run('validation')
configs=json.loads((R/'configs.json').read_text());selection={};valrows=np.flatnonzero(a['query_partition']==1);b=base[valrows];val_summary=[]
for c in configs:
 g=np.genfromtxt(R/'graph'/f'{c["key"]}.validation.csv',delimiter=',',names=True);assert np.array_equal(g['query_row'],valrows)
 val_summary.append(dict(key=c['key'],method=c['method'],target_coverage=c['target_coverage'],recall_pct=float(g['hits'].mean()*10),delta_recall_pp=float((g['hits']-b['hits']).mean()*10),reads_per_query=float(g['reads'].mean()),saved_reads_per_query=float((b['reads']-g['reads']).mean()),live_pruned_per_query=float(g['pruned'].mean()),actual_live_coverage=float(g['pruned'].sum()/g['live_keeps'].sum())))
for m in range(7):
 rows=[v for v in val_summary if v['method'] in [-1,m] and v['delta_recall_pp']>=-.2-1e-9];winner=max(rows,key=lambda v:(v['saved_reads_per_query'],v['delta_recall_pp'],-v['target_coverage']));selection[METHODS[m]]=winner
lock=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected=selection,quality_budget_pp=.2,selection_source='validation graph only; not test labels or graph outcomes',configs_sha256=sha(R/'configs.json'),models_sha256=sha(R/'models.json'))
if (R/'SELECTION_LOCK.json').exists():assert json.loads((R/'SELECTION_LOCK.json').read_text())['selected']==selection
else:dump(R/'SELECTION_LOCK.json',lock)
csvsave(R/'validation_graph_summary.csv',val_summary);log('VALIDATION SELECTION FROZEN')
run('test')
subprocess.run([sys.executable,str(R/'analyze.py')],check=True)
