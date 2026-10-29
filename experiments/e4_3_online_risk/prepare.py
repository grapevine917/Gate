from common import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import struct,shutil
log('PREPARE risk-of-Recall-loss labels, frozen fit queries and stream protocols')
a=dict(np.load(E42/'features.npz'));o=dict(np.load(E41/'outcomes.npz'));fit=a['partition']==0;x=a['x'].copy();x[:,[23,24,26,27]]=0;y=(o['delta_hits']<0).astype(int)
thresholds=[0.,1e-5,5e-5,.0001,.0005,.001,.002,.005,.01,.02,.05,.1,.2,.5,1.]
Rmins=[99.60,99.44,99.20,99.00]
protocol=dict(stage='E4-3',objective='minimize validation mean SSD requests subject to validation Recall@10 >= absolute R_min; no inference-saving objective',primary_R_min_pct=99.44,R_min_sensitivity_pct=Rmins,R_min_source='carry previous 0.2pp allowance from existing overall 99.64% reference; explicit absolute threshold frozen before new experiment',dataset='SIFT1M; original E1 graph',L=100,K=10,beam=4,original_MLP=str(R.parent/'sift1m/online_state_mlp/model.bin'),original_tau=json.loads((E1/'SELECTION_LOCK.json').read_text())['selected']['MLP'],DRAM_cache_nodes=10000,cache_policy='preserve E1-E4 miss-only MLP: DRAM hits protected and not secondary candidates',query_pool='Existing E4-1 1000 queries, previously observed; exploratory within-pool split, not a new confirmatory test set',query_partition_source=str(E42/'PROTOCOL.json'),fit_query_rows=np.flatnonzero(a['query_partition']==0).tolist(),validation_query_rows=np.flatnonzero(a['query_partition']==1).tolist(),test_query_rows=np.flatnonzero(a['query_partition']==2).tolist(),history_seed='same frozen E1 1000 training history queries; warm runs start from this seed, cold runs from empty history',history_updates='buffer original fresh-MLP outcomes along actual live search; after current query finishes, increment observations for every original model call and append state records ONLY for original MLP Skip; secondary Skip stays an original Keep observation and never becomes a historical Skip',causality='single active Query at a time; every query reads only seed or previously completed Query records; commit at query boundary; no GT or U used for updates',history_capacity='retain all original Skip states over each finite 250-query stream; no eviction, duplicate vector observations count as events',state_normalization='frozen E2 training mean/std; distance min standardized L2 over same-node history Skip states',risk_target='P(single-node forced Skip lowers Recall | original MLP Keep), trained on E4-1 delta_hits<0 (C/D), NOT on not-U (B/C/D)',risk_models={'state_LR':'current native retain logit + original normalized 17D state','history_LR':'state_LR + H + log1p(total history observations) + log1p(history Skip observations) + Skip fraction + log1p(nearest Skip state distance) + State-close indicator'},risk_fit='500 E4-2 fit queries only; StandardScaler fit-only, LogisticRegression C=1, no class weighting, max_iter=3000; original MLP unchanged',risk_probability_thresholds=thresholds,confidence_thresholds=[.03,.04,.06,.09,.12,.2,.4],controls=['MLP passive dynamic history (no secondary pruning)','unconditional dynamic historical Skip mark','risk history LR with frozen seed history','risk state LR with dynamic history','risk history LR with dynamic history (primary)','native confidence within dynamic history marks','global native confidence threshold without H gate'],selection='each family and R_min: minimum validation mean requests among feasible frozen grid points plus baseline fallback; no feasible config explicitly reported; selected configs locked before test',graph='real full graph search; fresh MLP is executed first on each miss; second scorer only on current Keep with past Skip (except explicit global confidence control); state features and history counters recomputed on changed trajectories',stream_order='validation and main test in ascending original E1 row order; robustness uses 3 fixed independently shuffled orders of test Queries; benchmark order is not production traffic',robustness='main-R_min selected configurations: 3 shuffled orders with warm and cold initialization; reset independent policy bank per config/order; frozen-history control on empty seed cannot create marks',latency='per-query search time includes actual lookup/scoring/buffering, plus separately measured commit time; graph load/cache setup excluded; no precomputed secondary decision tapes or Query distance tables',confidence='1000 paired Query bootstrap samples conditional on each realized sequential history trajectory; not a bootstrap of new causal stream histories',seed=SEED,scope='test GT only for offline scoring after selection; no guarantee a validation-feasible config remains test-feasible')
if (R/'PROTOCOL.json').exists():assert json.loads((R/'PROTOCOL.json').read_text())==protocol
else:dump(R/'PROTOCOL.json',protocol)
models=[]
for name,cols in [('state_LR',[0]+list(range(1,18))),('history_LR',[0]+list(range(1,18))+list(range(18,23))+[25])]:
 z=x[fit][:,cols];sc=StandardScaler().fit(z);lr=LogisticRegression(C=1,max_iter=3000,random_state=SEED).fit(sc.transform(z),y[fit]);w=lr.coef_[0]/sc.scale_;weights=np.zeros(28);weights[cols]=w;models.append(dict(name=name,weights=weights.tolist(),intercept=float(lr.intercept_[0]-w@sc.mean_),features=[FEATURES[j] for j in cols],fit_n=int(fit.sum()),fit_harmful=int(y[fit].sum()),label='delta_hits < 0',normalization='fit-only StandardScaler on original E2-standardized states and fit history statistics'))
dump(R/'risk_models.json',models)
with (R/'risk_models.bin').open('wb') as f:
 f.write(struct.pack('<2I',2,28))
 for m in models:np.asarray([m['intercept']]+m['weights'],'<f8').tofile(f)
configs=[dict(key='MLP',kind=0,threshold=0.,family='MLP'),dict(key='History_unconditional',kind=1,threshold=1.,family='History_unconditional')]
for kind,family in [(2,'Risk_dynamic_state'),(3,'Risk_static_history'),(4,'Risk_dynamic_history')]:
 for i,t in enumerate(thresholds):configs.append(dict(key=f'{family}_t{i:02d}',kind=kind,threshold=t,family=family))
for kind,family in [(5,'Confidence_dynamic_history'),(6,'Confidence_global')]:
 for i,t in enumerate(protocol['confidence_thresholds']):configs.append(dict(key=f'{family}_t{i:02d}',kind=kind,threshold=t,family=family))
dump(R/'configs.json',configs)
with (R/'configs.txt').open('w') as f:
 f.write(str(len(configs))+'\n')
 for c in configs:f.write(f'{c["key"]} {c["kind"]} {c["threshold"]:.17g}\n')
shutil.copy(E42/'history.bin',R/'history.bin')
splits=json.loads((E1/'query_splits.json').read_text())
with (R/'partitions.txt').open('w') as f:
 for i in range(1000):f.write(f'{i} {int(a["query_partition"][i])} {int(splits["test"][i])}\n')
(R/'orders').mkdir(exist_ok=True)
for name,rows in [('audit',np.arange(1000)),('validation',np.flatnonzero(a['query_partition']==1)),('test',np.flatnonzero(a['query_partition']==2))]:np.savetxt(R/'orders'/f'{name}.txt',rows,fmt='%d')
test=np.flatnonzero(a['query_partition']==2)
for i in range(3):np.savetxt(R/'orders'/f'test_shuffle{i}.txt',np.random.default_rng(SEED+i).permutation(test),fmt='%d')
log(f'PREPARED {len(configs)} policy configurations')
dump(R/'PREPARED.json',dict(complete=True,protocol_sha256=sha(R/'PROTOCOL.json'),configs=len(configs),fit_queries=500,validation_queries=250,test_queries=250,history_sha256=sha(R/'history.bin'),risk_training_positive=int(y[fit].sum()),risk_training_positions=int(fit.sum())))
