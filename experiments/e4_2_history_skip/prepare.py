from common import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
import struct
log('PREPARE fixed query split and historical feature bank')
p=dict(np.load(E4/'keep_positions.npz'));out=dict(np.load(E4/'outcomes.npz'));h=dict(np.load(E1/'MLP_history_trace.npz'));splits=json.loads((E1/'query_splits.json').read_text());q=np.load(E1/'data/query_vectors.npy').astype(float)
norm=np.load(E2/'normalization.npz');mean=norm['mean'];denom=norm['std']+1e-8;cut=json.loads((E2/'PROTOCOL.json').read_text());scut=cut['state_high_if_normalized_l2_le'];qcut=cut['query_high_if_l2_le']
qids=np.array(splits['test']);rng=np.random.default_rng(SEED);order=rng.permutation(1000);partition=np.empty(1000,np.uint8);partition[order[:500]]=0;partition[order[500:750]]=1;partition[order[750:]]=2;row=out['query_row'];part=partition[row]
lock=dict(seed=SEED,fit_query_rows=order[:500].tolist(),validation_query_rows=order[500:750].tolist(),test_query_rows=order[750:].tolist(),source='Existing E4-1 test query pool; disjoint query partitions within this analysis, not a previously unseen confirmatory dataset',history='1000 E1 training historical queries only, frozen',normalization='E2 training-only mean/std',state_distance_cutoff=scut,query_distance_cutoff=qcut,features=FEATURES,models={'M0':'negative native retain logit; lower Keep score first','M1':'H binary; higher first, deterministic hash for ties','M2':'historical Skip / actual model-invoked same-node observations; higher first; H gate','M3':'logistic(log1p min same-node historical Skip state distance); H gate','M4':'logistic(log1p min same-node historical Skip Query distance); H gate','M5':'logistic(M3 state distance, M4 query distance, query distance at nearest Skip state); H gate','M6':'logistic(current retain logit + 17 normalized state dimensions + H + log history total + log Skip count + Skip fraction + log Skip state distance + state_close); no Query distance, no hard H gate'},logistic='C=1, standardization on fit partition only; no class reweighting or hyperparameter selection',coverages=COVERAGES,history_methods_gate='M1-M5 only select H=1; requested coverage above capacity reported unavailable',tie_policy='deterministic query_id,node_id hash independent of U; fractional boundary-tie expected precision also reported',graph='fresh MLP first; only additional pruning of live Keep miss candidates; features recomputed on changed live state; no baseline intervention tape used',selection='validation graph max SSD requests saved under <=0.2pp aggregate additional Recall loss, otherwise baseline; frozen menu curves evaluated on test without selecting a test winner',positive_label='E4-1 A class from isolated single-node interventions; not cumulative-search safety',primary_contrasts=['P(A|H)-P(A|not H)','P(A|H,state<scut)-P(A|H)','M5-M3 prediction performance on identical held-out Conflict Keep positions'],confidence='1000 target-query cluster bootstrap draws; history fixed; exploratory post-hoc E4 dataset')
if (R/'PROTOCOL.json').exists():assert json.loads((R/'PROTOCOL.json').read_text())==lock
else:dump(R/'PROTOCOL.json',lock)
assert not (set(splits['history'])&set(splits['test']))
hrows={int(q):i for i,q in enumerate(splits['history'])};histq=q[splits['history']];testq=q[splits['test']];qd=np.sqrt(np.maximum((testq**2).sum(1)[:,None]+(histq**2).sum(1)[None,:]-2*testq@histq.T,0));qd.astype('<f8').tofile(R/'query_distances.bin')
take=np.flatnonzero(h['model_invoked']);by={}
for i in take:by.setdefault(int(h['node_id'][i]),[]).append(int(i))
hz=(h['state_features'].astype(float)-mean)/denom;pz=(p['state_features'].astype(float)-mean)/denom
n=len(row);total=np.zeros(n);skips=np.zeros(n);ds=np.full(n,np.inf);dq=np.full(n,np.inf);dqstate=np.full(n,np.inf);joint=np.zeros(n,bool)
for j,node in enumerate(p['node_id']):
 ii=by.get(int(node),[]);total[j]=len(ii)
 si=[i for i in ii if h['model_decision'][i]];skips[j]=len(si)
 if not si:continue
 dd=np.linalg.norm(hz[si]-pz[j],axis=1);qq=qd[row[j],[hrows[int(h['query_id'][i])] for i in si]];nn=int(np.argmin(dd));ds[j]=dd[nn];dq[j]=qq.min();dqstate[j]=qq[nn];joint[j]=np.any((dd<scut)&(qq<qcut))
H=skips>0
x=np.zeros((n,len(FEATURES)));score=np.clip(p['model_score'].astype(float),1e-7,1-1e-7);x[:,0]=np.log(score/(1-score));x[:,1:18]=pz;x[:,18]=H;x[:,19]=np.log1p(total);x[:,20]=np.log1p(skips);x[:,21]=skips/np.maximum(total,1);x[:,22]=np.log1p(np.where(H,ds,0));x[:,23]=np.log1p(np.where(H,dq,0));x[:,24]=np.log1p(np.where(H,dqstate,0));x[:,25]=H&(ds<scut);x[:,26]=H&(dq<qcut);x[:,27]=joint
np.savez_compressed(R/'features.npz',x=x,U=out['U'],query_row=row,query_id=p['query_id'],node_id=p['node_id'],H=H,state_distance=ds,query_distance=dq,query_at_state_nn=dqstate,joint_close=joint,history_total=total,history_skip=skips,partition=part,query_partition=partition,tie=tie_hash(p['query_id'],p['node_id']))
models=[dict(name=m,weights=[0.]*len(FEATURES),intercept=0.,features=[]) for m in METHODS];models[0]['weights'][0]=-1;models[1]['weights'][18]=1;models[2]['weights'][21]=1
cols={3:[22],4:[23],5:[22,23,24],6:[0]+list(range(1,18))+list(range(18,23))+[25]}
for k,c in cols.items():
 mask=(part==0)&(H if k<6 else True);z=x[mask][:,c];y=out['U'][mask].astype(int);scale=StandardScaler().fit(z);lr=LogisticRegression(C=1,max_iter=3000,random_state=SEED).fit(scale.transform(z),y);w=lr.coef_[0]/scale.scale_;b=float(lr.intercept_[0]-w@scale.mean_);weights=np.zeros(len(FEATURES));weights[c]=w;models[k].update(weights=weights.tolist(),intercept=b,features=[FEATURES[i] for i in c],fit_n=int(mask.sum()),fit_positive=int(y.sum()),coefficients=dict(zip([FEATURES[i] for i in c],w.tolist())))
dump(R/'models.json',models)
with (R/'models.bin').open('wb') as f:
 f.write(struct.pack('<2I',7,len(FEATURES)))
 for m in models:np.asarray([m['intercept']]+m['weights'],'<f8').tofile(f)
# Match E3 history ordering and representation; no changes to legacy artifacts.
assert (E3/'history.bin').exists()
with (R/'history.bin').open('wb') as f:
 np.asarray(mean,'<f8').tofile(f);np.asarray(denom,'<f8').tofile(f);ordered=take[np.argsort(h['node_id'][take],kind='stable')];f.write(struct.pack('<I',len(ordered)))
 for i in ordered:f.write(struct.pack('<4I17d',int(h['node_id'][i]),hrows[int(h['query_id'][i])],int(h['model_decision'][i]),0,*hz[i]))
assert sha(R/'history.bin')==sha(E3/'history.bin')
scores=evaluate_scores(x,models);np.save(R/'scores.npy',scores)
# Coverage thresholds use validation baseline Keeps only, never test outcomes or test score quantiles.
configs=[dict(key='baseline',method=-1,cutoff=1e30,tiecut=-1.,target_coverage=0.)]
for k,m in enumerate(METHODS):
 vm=part==1;eligible=vm&(H if 1<=k<=5 else True);ix=np.flatnonzero(eligible);capacity=len(ix)/vm.sum()
 targets=[(f'c{int(round(c*10000)):04d}',c) for c in COVERAGES if c<=capacity]
 if 1<=k<=5:targets.append(('allH',capacity))
 for label,cov in targets:
  count=min(len(ix),max(1,int(np.ceil(cov*vm.sum()))));rank=ix[np.lexsort((tie_hash(p['query_id'][ix],p['node_id'][ix]),-scores[ix,k]))];chosen=rank[:count];boundary=float(scores[rank[count-1],k]);tied=rank[scores[rank,k]==boundary];chosen_tie=np.intersect1d(chosen,tied);others=np.setdiff1d(tied,chosen_tie)
  if len(others):tiecut=float((tie_hash(p['query_id'][chosen_tie],p['node_id'][chosen_tie]).max()+tie_hash(p['query_id'][others],p['node_id'][others]).min())/2);cutoff=boundary
  elif count<len(rank):cutoff=float((boundary+scores[rank[count],k])/2);tiecut=1.
  else:cutoff=boundary-1e-9;tiecut=1.
  configs.append(dict(key=f'{m}_{label}',method=k,cutoff=cutoff,tiecut=tiecut,target_coverage=float(cov),validation_baseline_capacity=capacity))
dump(R/'configs.json',configs)
# Native driver reads all 1000 queries but runs only provided partition rows.
with (R/'partitions.txt').open('w') as f:
 for i in range(1000):f.write(f'{i} {int(partition[i])} {int(qids[i])}\n')
with (R/'configs.txt').open('w') as f:
 f.write(str(len(configs))+'\n')
 for c in configs:f.write(f'{c["key"]} {c["method"]} {c["cutoff"]:.17g} {c["tiecut"]:.17g}\n')
dump(R/'PREPARED.json',dict(complete=True,positions=n,conflict=int(H.sum()),partition_counts={str(k):int(np.sum(part==k)) for k in range(3)},models=7,graph_configs=len(configs),protocol_sha256=sha(R/'PROTOCOL.json')))
log(f'PREPARED n={n}, conflicts={H.sum()}, configs={len(configs)}')
