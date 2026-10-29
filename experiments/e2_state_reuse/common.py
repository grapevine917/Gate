import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import datetime,json,hashlib,csv
import numpy as np
R=Path(__file__).resolve().parent;LG=R.parents[1];E1=R.parent/'e1_similarity';S=R.parent/'sift1m';SEED=20261019
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def log(msg):
 line=datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+msg;print(line,flush=True)
 with (R/'experiment.log').open('a') as f:f.write(line+'\n')
def csvsave(p,rows):
 if not rows:return
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def trace(name,split):return dict(np.load(E1/f'{name}_{split}_trace.npz'))
def node_pairs(a,b=None):
 # All same-node pairs. Training uses distinct unordered query pairs, then both directions.
 ia=np.flatnonzero(a['model_invoked']);sa=ia[np.argsort(a['node_id'][ia],kind='stable')];ua,starts,counts=np.unique(a['node_id'][sa],return_index=True,return_counts=True)
 left=[];right=[]
 if b is None:
  for start,count in zip(starts,counts):
   group=sa[start:start+count]
   if count>1:
    i,j=np.triu_indices(count,1);left.extend(group[i]);right.extend(group[j])
  x=np.array(left,np.int64);y=np.array(right,np.int64);return np.r_[x,y],np.r_[y,x]
 ib=np.flatnonzero(b['model_invoked']);sb=ib[np.argsort(b['node_id'][ib],kind='stable')];ub,st,ct=np.unique(b['node_id'][sb],return_index=True,return_counts=True)
 _,ai,bi=np.intersect1d(ua,ub,return_indices=True)
 for i,j in zip(ai,bi):
  x=sa[starts[i]:starts[i]+counts[i]];y=sb[st[j]:st[j]+ct[j]]
  left.extend(np.repeat(x,len(y)));right.extend(np.tile(y,len(x)))
 return np.array(left,np.int64),np.array(right,np.int64)
def cross_control(a,b,left,right):
 rng=np.random.default_rng(SEED);byq={int(q):np.flatnonzero((a['query_id']==q)&a['model_invoked']) for q in np.unique(a['query_id'])}
 out=np.empty(len(left),np.int64)
 for k,(i,j) in enumerate(zip(left,right)):
  pool=byq[int(a['query_id'][i])];pool=pool[a['node_id'][pool]!=b['node_id'][j]];out[k]=rng.choice(pool)
 return out,right.copy()
def pair_data(a,b,left,right,mean,std,q,tau,qcut,scut):
 xs=a['state_features'][left].astype(float);xt=b['state_features'][right].astype(float)
 dq=np.linalg.norm(q[a['query_id'][left]].astype(float)-q[b['query_id'][right]].astype(float),axis=1)
 ds=np.linalg.norm((xs-xt)/(std+1e-8),axis=1)
 hs=a['model_decision'][left].astype(bool);ts=b['model_decision'][right].astype(bool);dp=abs(a['model_score'][left]-b['model_score'][right]).astype(float)
 group=np.where(dq<=qcut,np.where(ds<=scut,1,2),np.where(ds<=scut,3,4)).astype(np.uint8)
 return dict(source_record=left,target_record=right,source_query=a['query_id'][left],target_query=b['query_id'][right],source_node=a['node_id'][left],target_node=b['node_id'][right],query_distance=dq,state_distance=ds,delta_p=dp,source_score=a['model_score'][left],target_score=b['model_score'][right],source_skip=hs,target_skip=ts,agreement=hs==ts,false_skip=hs&~ts,group=group)
def rate_ci(values,denom,cluster):
 values=np.asarray(values,float);denom=np.asarray(denom,bool)
 if not denom.any():return None,None,None
 _,ix=np.unique(cluster,return_inverse=True);n=ix.max()+1;s=np.bincount(ix,weights=values*denom,minlength=n);d=np.bincount(ix,weights=denom,minlength=n)
 rng=np.random.default_rng(SEED);idx=rng.integers(0,n,(500,n));ds=d[idx].sum(1);v=s[idx].sum(1)/np.maximum(ds,1);v=v[ds>0]
 return float(s.sum()/d.sum()),*map(float,np.percentile(v,[2.5,97.5]))
def summarize(p,mask,label):
 take=np.flatnonzero(mask);n=len(take);c=p['target_query'][take];ss=p['source_skip'][take];ts=p['target_skip'][take]
 row={'group':label,'pairs':n,'target_queries':int(len(np.unique(c))),'target_positions':int(len(np.unique(p['target_record'][take]))),'source_skip_count':int(ss.sum()),'target_keep_count':int((~ts).sum()),'false_skip_count':int(p['false_skip'][take].sum())}
 for key,v,d in [('agreement',p['agreement'][take],np.ones(n,bool)),('delta_p',p['delta_p'][take],np.ones(n,bool)),('false_skip_rate',p['false_skip'][take],~ts),('false_skip_given_reuse',p['false_skip'][take],ss),('skip_precision',ts,ss)]:
  row[key],row[key+'_lo'],row[key+'_hi']=rate_ci(v,d,c)
 for key in ['query_distance','state_distance']:row['mean_'+key]=float(p[key][take].mean()) if n else None
 return row
