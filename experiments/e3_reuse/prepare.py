from common import *
import struct
def main():
 log('PREPARE: frozen training history and precomputed query neighbours')
 h=dict(np.load(E1/'MLP_history_trace.npz'));take=np.flatnonzero(h['model_invoked']);splits=json.loads((E1/'query_splits.json').read_text());hids=np.array(splits['history']);hrow={int(q):i for i,q in enumerate(hids)}
 norm=np.load(E2/'normalization.npz');mean=norm['mean'];denom=norm['std']+1e-8
 order=take[np.argsort(h['node_id'][take],kind='stable')]
 with (R/'history.bin').open('wb') as f:
  np.asarray(mean,'<f8').tofile(f);np.asarray(denom,'<f8').tofile(f);f.write(struct.pack('<I',len(order)))
  for i in order:f.write(struct.pack('<4I17d',int(h['node_id'][i]),hrow[int(h['query_id'][i])],int(h['model_decision'][i]),0,*((h['state_features'][i].astype(float)-mean)/denom)))
 q=np.load(E1/'data/query_vectors.npy').astype(float);hist=q[hids]
 for split in ['validation','test']:
  ids=np.array(splits[split]);d=np.sqrt(np.maximum((q[ids]**2).sum(1)[:,None]+(hist**2).sum(1)[None,:]-2*q[ids]@hist.T,0));nn=np.argsort(d,axis=1,kind='stable')[:,:3];dist=np.take_along_axis(d,nn,axis=1)
  np.savez(R/f'{split}_neighbours.npz',history_rows=nn,l2=dist,query_ids=ids)
  with (R/f'{split}_neighbours.txt').open('w') as f:
   for n,ds in zip(nn,dist):f.write(' '.join(map(str,n))+' '+' '.join(format(v,'.17g') for v in ds)+'\n')
 d=np.sqrt(np.maximum((hist**2).sum(1)[:,None]+(hist**2).sum(1)[None,:]-2*hist@hist.T,0));np.fill_diagonal(d,np.inf)
 qcuts=sorted(set([float(json.loads((E2/'PROTOCOL.json').read_text())['query_high_if_l2_le']),*map(float,np.quantile(d.min(1),[.9,.99]))]))
 scut=json.loads((E2/'PROTOCOL.json').read_text())['state_high_if_normalized_l2_le'];scuts=sorted(set([.5,1.,2.,float(scut),6.]))
 configs=[]
 def add(name,kind,qcut=1e30,scut=1e30,votes=2,oracle=0):
  c={'key':name,'kind':kind,'qcut':qcut,'scut':scut,'votes':votes,'oracle':oracle};configs.append(c);return c
 add('R0',0);add('R1',1);add('R2',2)
 for j,s in enumerate(scuts+[1e30]):add(f'R3_s{j}',3,scut=s)
 for i,qc in enumerate(qcuts):
  for j,s in enumerate(scuts):add(f'R4_q{i}_s{j}',4,qc,s)
  for votes in [2,3]:add(f'R5_q{i}_v{votes}',5,qc,votes=votes)
 # Free model-decision oracle over ANY history Skip for the same node.
 add('OracleA',1,oracle=1)
 dump(R/'configs.json',configs)
 protocol={'stage':'E3','dataset':'SIFT1M','L':100,'K':10,'beam':4,'threads':4,'cache_nodes':10000,'MLP':str(S/'online_state_mlp/model.bin'),'tau':json.loads((E1/'SELECTION_LOCK.json').read_text())['selected']['MLP'],'history_queries':1000,'neighbour_queries':3,'history_source':'E1 training queries only, immutable; both Keep and Skip records kept','normalization':str(E2/'normalization.npz'),'query_cutoffs_train_only':qcuts,'state_cutoffs_train_only':scuts,'max_additional_recall_loss_pp':.2,'selection':'validation real graph replay: maximize (baseline model calls - actual logical calls)/baseline calls subject to <=0.2pp recall loss; ties lower false skips then stricter thresholds; test never chooses deployable settings','R1':'union of historical learned Skip marks by node; any historical Skip suffices; different from legacy contribution global mark','R2':'single nearest historical Query; reuse its node Skip marks, no distance cutoff','R3':'closest normalized state among ALL same-node historical Keep/Skip records; reuse only if closest label Skip and distance<=scut','R4':'among same-node records from top3 historical Queries passing qcut, choose closest state; reuse if Skip and ds<=scut; both finite thresholds','R5':'top3 Queries passing qcut; at least votes distinct available histories and ALL available same-node histories vote Skip; absent records do not vote','OracleA':'free knowledge of fresh current Skip; only reuse union-history Skip when current model also Skip; unavailable online; fresh-score shadow is outside latency pass','OracleB':'exact multiple-choice quality-budget optimization over already replayed frozen strategy trajectories per query; budget0.2pp, baseline option allowed; exact optimum within evaluated trajectory menu, NOT unrestricted combinatorial upper bound','static':'baseline trajectories only, no final-recall claims; validation used for sweep; held-out test diagnostics after lock','graph':'planning pass physically scores fresh model for diagnostics; tape pass really bypasses inference for reused decisions and verifies node/state/path parity','latency':'warm graph replay of precomputed decision tapes, excludes query-neighbour and state matching/maintenance; includes O(1) tape access and parity checks; three randomized repeats; NOT deployed cache latency','eligible':'all live cache-miss model decision events; ReuseRate=reused/live eligible; ModelSaving=(fixed R0 calls - actual tape calls)/fixed R0 calls; path changes can separate these','false_skip_rate':'reused Skip with shadow fresh Keep / all shadow fresh Keep on corresponding live path; also report erroneous fraction among reused','quality':'Recall@10 relative to exact SIFT1M GT; aggregate recall budget, not per-query guarantee','legacy_baseline':'retain E1 contribution-based global_once separately; not rename its rule to R1','timing_not_in_selection':True,'no_online_cache_implementation_claim':True}
 dump(R/'PROTOCOL.json',protocol);dump(R/'PREPARED.json',{'complete':True,'history_events':len(order),'union_skip_nodes':int(len(np.unique(h['node_id'][take][h['model_decision'][take]!=0]))),'configs':len(configs),'protocol_sha256':sha(R/'PROTOCOL.json')})
 log(f'PREPARED {len(configs)} configurations, query cutoffs {qcuts}')
if __name__=='__main__':main()
