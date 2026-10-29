"""Query-disjoint prediction of singleton-safe request-saving cache misses."""
from pathlib import Path
import json,struct,time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,balanced_accuracy_score
ROOT=Path(__file__).resolve().parent
STATE_NAMES=['candidate_pq','best_pq','worst_pq','candidate_rank','queue_size','unexpanded_count','visited_count','completed_expansions','io_beam_round','beam_position','cached_node','best_exact_so_far','has_exact_so_far','candidate_over_best_pq','candidate_over_worst_pq','L','beam_width']
def matrix(path):
 with open(path,'rb') as f:n,d=np.fromfile(f,np.uint32,2);return np.fromfile(f,np.float32).reshape(int(n),int(d))
def dataset(ds,split):
 root=ROOT/ds/split
 features=np.loadtxt(root/'collected_features.csv',delimiter=',',skiprows=1,dtype=np.float64)
 source=ROOT.parent/'io_opportunity'/f'{ds}_L100_candidates.csv' if split=='test' else root/'collected_candidates.csv'
 labels=np.loadtxt(source,delimiter=',',skiprows=1,dtype=np.float64)
 assert len(np.unique(features[:,0]))==1000 and len(np.unique(labels[:,0]))==1000
 key=lambda a:a[:,0].astype(np.int64)*1000000+a[:,1].astype(np.int64)
 fk,lk=key(features),key(labels);order=np.argsort(fk);ix=np.searchsorted(fk[order],lk)
 assert len(np.unique(fk))==len(fk) and np.array_equal(fk[order[ix]],lk)
 states=features[order[ix],2:].astype(np.float32)
 assert np.array_equal(states[:,10]==0,labels[:,2]==1)
 miss=labels[:,2]==1;states=states[miss];labels=labels[miss]
 y=((labels[:,6]==1)&(labels[:,9]>0)).astype(np.int8)
 q=labels[:,0].astype(np.int64)
 info={'n_queries':1000,'n_cache_miss_expansions':len(y),'safe_saving_fraction':float(y.mean()),'quality_critical_count':int((labels[:,6]==0).sum())}
 np.savez_compressed(root/'dataset.npz',state=states,y=y,query=q,node=labels[:,1].astype(np.uint32),delta=labels[:,9].astype(np.int32),quality_safe=labels[:,6].astype(np.int8))
 return states,y,q,labels,matrix(root/'query.fbin'),info

def f32(a):return np.asarray(a,dtype='<f4').tobytes()
def export(path,model,scaler,qdim):
 with open(path,'wb') as f:
  if isinstance(model,LogisticRegression):
   f.write(b'LRGATE01');f.write(struct.pack('<I',qdim));f.write(f32(scaler.mean_));f.write(f32(scaler.scale_));f.write(f32(model.coef_[0]));f.write(f32(model.intercept_))
  else:
   t=model.tree_;f.write(b'TRGATE01');f.write(struct.pack('<II',qdim,t.node_count))
   for a in [t.feature,t.children_left,t.children_right]:f.write(np.asarray(a,dtype='<i4').tobytes())
   f.write(f32(t.threshold));f.write(f32(t.value[:,0,1]/t.value[:,0,:].sum(axis=1)))
def native_scores(path,x):
 # Validate exported float32 parameters against the model before evaluating predictions.
 with open(path,'rb') as f:
  magic=f.read(8);qdim=struct.unpack('<I',f.read(4))[0];d=17+qdim
  if magic==b'LRGATE01':
   mean=np.fromfile(f,'<f4',d);scale=np.fromfile(f,'<f4',d);w=np.fromfile(f,'<f4',d);b=np.fromfile(f,'<f4',1)[0]
   z=((x.astype(np.float64)-mean)/scale)@w+b
   from scipy.special import expit
   # Match native retained probability rounded to float32, then strict gate inequality.
   return expit(-z).astype(np.float32)
  n=struct.unpack('<I',f.read(4))[0];feat=np.fromfile(f,'<i4',n);left=np.fromfile(f,'<i4',n);right=np.fromfile(f,'<i4',n);threshold=np.fromfile(f,'<f4',n);safe=np.fromfile(f,'<f4',n)
  nodes=np.zeros(len(x),np.int32)
  for _ in range(n):
   active=feat[nodes]>=0
   if not active.any():break
   old=nodes[active];v=x[active,feat[old]];nodes[active]=np.where(v<=threshold[old],left[old],right[old])
  return (1-safe[nodes]).astype(np.float32)

summary={}
for ds in ['sift','deep']:
 data={s:dataset(ds,s) for s in ['train','validation','test']}
 ids={s:set(np.load(ROOT/ds/s/'query_ids.npy').tolist()) for s in data}
 assert not (ids['train']&ids['validation'] or ids['train']&ids['test'] or ids['validation']&ids['test'])
 results={};val_tables={};test_tables={}
 for name in ['state_lr','state_tree','state_query_lr']:
  inputs={s:np.concatenate([d[0],d[4][d[2]]],axis=1) if name=='state_query_lr' else d[0] for s,d in data.items()}
  scaler=None;model=DecisionTreeClassifier(max_depth=4,min_samples_leaf=200,class_weight='balanced',random_state=42) if name=='state_tree' else LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=42)
  x=inputs['train']
  if name!='state_tree':scaler=StandardScaler().fit(x);x=scaler.transform(x)
  start=time.perf_counter();model.fit(x,data['train'][1]);fit_sec=time.perf_counter()-start
  dest=ROOT/ds/name;dest.mkdir(exist_ok=True)
  export(dest/'model.bin',model,scaler,inputs['train'].shape[1]-17)
  probs={};retains={};metrics={}
  for split in ['validation','test']:
   xx=scaler.transform(inputs[split]) if scaler is not None else inputs[split]
   start=time.perf_counter();p=model.predict_proba(xx)[:,1];infer=time.perf_counter()-start
   retain=native_scores(dest/'model.bin',inputs[split]);assert np.max(np.abs((1-retain)-p))<0.0001
   retains[split]=retain;probs[split]=1-retain;y=data[split][1]
   metrics[split]={'roc_auc':float(roc_auc_score(y,probs[split])),'average_precision':float(average_precision_score(y,probs[split])),
                   'prevalence_baseline_ap':float(y.mean()),'accuracy_at_half':float(accuracy_score(y,p>=.5)),
                   'balanced_accuracy_at_half':float(balanced_accuracy_score(y,p>=.5)),
                   'batched_python_predict_us_per_node':infer/len(y)*1e6}
  thresholds={}
  for target in [0.001,0.005,0.01]:
   neg=np.sort(retains['validation'][data['validation'][1]==0]);budget=int(np.floor(len(neg)*target));tau=float(neg[min(budget,len(neg)-1)])
   key=f'unsafe_fpr_{target:g}';thresholds[key]={'retain_threshold':tau,'validation_unsafe_budget':budget,'selection_split':'validation','target_unsafe_fpr':target}
   for split in ['validation','test']:
    y=data[split][1];labs=data[split][3];pred=retains[split]<tau;tp=int((pred&(y==1)).sum());fp=int((pred&(y==0)).sum());critical=labs[:,6]==0
    metrics[split][key]={'predicted_skip_count':int(pred.sum()),'precision_safe_saving':tp/max(1,int(pred.sum())),
                        'recall_safe_saving':tp/int(y.sum()),'unsafe_fpr':fp/int((y==0).sum()),
                        'quality_critical_skip_rate':int((pred&critical).sum())/int(critical.sum()),
                        'quality_critical_skips':int((pred&critical).sum()),'replacement_read_skips':int((pred&(labs[:,6]==1)&(labs[:,9]<=0)).sum())}
  (dest/'thresholds.json').write_text(json.dumps(thresholds,indent=2)+'\n')
  results[name]={'fit_seconds':fit_sec,'metrics':metrics,'thresholds':thresholds,
                 'state_features':STATE_NAMES,'query_dimensions':inputs['train'].shape[1]-17}
  if name=='state_tree':results[name]['feature_importances']=dict(zip(STATE_NAMES,model.feature_importances_.tolist()))
  else:results[name]['standardized_coefficients']=dict(zip(STATE_NAMES+([f'q{i}' for i in range(data['train'][4].shape[1])] if name=='state_query_lr' else []),model.coef_[0].tolist()))
 summary[ds]={'splits':{s:d[5] for s,d in data.items()},'models':results}
(ROOT/'classification_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({d:{m:r['metrics'] for m,r in v['models'].items()} for d,v in summary.items()},indent=2))
