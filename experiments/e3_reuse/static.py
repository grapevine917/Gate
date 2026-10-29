from common import *
import struct
DT=np.dtype([('node','<u4'),('reused','<u4'),('fresh','<f4'),('state','<f4',(17,))])
def tapes(path):
 out=[]
 with path.open('rb') as f:
  nq=struct.unpack('<I',f.read(4))[0]
  for _ in range(nq):
   n=struct.unpack('<I',f.read(4))[0];out.append(np.frombuffer(f.read(n*80),DT).copy())
  assert not f.read(1)
 return out
def write_tapes(path,rows):
 with path.open('wb') as f:
  f.write(struct.pack('<I',len(rows)))
  for a in rows:f.write(struct.pack('<I',len(a)));f.write(a.tobytes())
def static_analysis(split,configs):
 tape=tapes(R/'plans'/f'{split}_R0_plan.tape');qrow=np.repeat(np.arange(len(tape)),[len(a) for a in tape]);a=np.concatenate(tape);n=len(a)
 h=dict(np.load(E1/'MLP_history_trace.npz'));take=np.flatnonzero(h['model_invoked']);ids=np.array(json.loads((E1/'query_splits.json').read_text())['history']);lookup={int(q):i for i,q in enumerate(ids)}
 norm=np.load(E2/'normalization.npz');z=(h['state_features'][take].astype(float)-norm['mean'])/(norm['std']+1e-8);hr=np.array([lookup[int(q)] for q in h['query_id'][take]]);hs=h['model_decision'][take]!=0
 bank={};ordered=np.argsort(h['node_id'][take],kind='stable');nodes,starts,counts=np.unique(h['node_id'][take][ordered],return_index=True,return_counts=True)
 for node,start,count in zip(nodes,starts,counts):
  ix=ordered[start:start+count];bank[int(node)]=(hr[ix],z[ix],hs[ix])
 neighbours=np.load(R/f'{split}_neighbours.npz');nn=neighbours['history_rows'];qd=neighbours['l2'];qcuts=json.loads((R/'PROTOCOL.json').read_text())['query_cutoffs_train_only']
 union=np.zeros(n,bool);qnn=union.copy();ns=np.zeros(n,bool);nd=np.full(n,np.inf);hybrid={qc:(np.zeros(n,bool),np.full(n,np.inf),np.zeros(n,int),np.zeros(n,int)) for qc in qcuts}
 for i,(r,qr) in enumerate(zip(a,qrow)):
  v=bank.get(int(r['node']))
  if v is None:continue
  histories,states,skips=v;union[i]=skips.any();match=histories==nn[qr,0]
  if match.any():qnn[i]=skips[match][0]
  target=(r['state'].astype(float)-norm['mean'])/(norm['std']+1e-8);ds=((states-target)**2).sum(1);best=int(ds.argmin());ns[i]=skips[best];nd[i]=np.sqrt(ds[best])
  for qc in qcuts:
   mask=np.isin(histories,nn[qr][qd[qr]<=qc]);ss,dd,available,votes=hybrid[qc];available[i]=mask.sum();votes[i]=skips[mask].sum()
   if mask.any():
    ix=np.flatnonzero(mask);b=ix[ds[mask].argmin()];ss[i]=skips[b];dd[i]=np.sqrt(ds[b])
 tau=json.loads((R/'PROTOCOL.json').read_text())['tau'];freshskip=a['fresh']<tau;rows=[];masks={}
 for c in configs:
  k=c['kind']
  if k==0:mask=np.zeros(n,bool)
  elif k==1:mask=union.copy()
  elif k==2:mask=qnn.copy()
  elif k==3:mask=ns&(nd<=c['scut'])
  elif k==4:ss,dd,_,_=hybrid[c['qcut']];mask=ss&(dd<=c['scut'])
  else:_,_,av,sk=hybrid[c['qcut']];mask=(av>=c['votes'])&(av==sk)
  if c['oracle']:mask&=freshskip
  fp=mask&~freshskip;rows.append({'key':c['key'],'eligible':n,'reused':int(mask.sum()),'reuse_rate':float(mask.mean()),'model_saving':float(mask.mean()),'false_skip':int(fp.sum()),'false_skip_rate':float(fp.sum()/(~freshskip).sum()),'false_skip_given_reuse':float(fp.sum()/mask.sum()) if mask.any() else None,'delta_recall_pp':None});masks[c['key']]=mask
 np.savez_compressed(R/f'{split}_static_masks.npz',query_row=qrow,node_id=a['node'],fresh_skip=freshskip,**masks);csvsave(R/f'{split}_static.csv',rows);dump(R/f'{split}_static.json',rows)
 return rows
