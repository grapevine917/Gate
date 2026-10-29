#!/usr/bin/env python3
from common import *
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,log_loss,brier_score_loss
def main():
 log('PREPARE START: training-only normalization and group thresholds')
 x=np.load(S/'training_data_l100/train/pre_io_state.npy',mmap_mode='r')
 mean=x.mean(0,dtype=np.float64);ss=np.zeros(17)
 for start in range(0,len(x),16384):ss+=((x[start:start+16384].astype(float)-mean)**2).sum(0)
 std=np.sqrt(ss/len(x));np.savez(R/'normalization.npz',mean=mean,std=std,epsilon=1e-8,train_samples=len(x))
 q=np.load(E1/'data/query_vectors.npy');h=trace('MLP','history');left,right=node_pairs(h)
 ds=np.linalg.norm((h['state_features'][left].astype(float)-h['state_features'][right])/(std+1e-8),axis=1)
 dq=np.linalg.norm(q[h['query_id'][left]].astype(float)-q[h['query_id'][right]],axis=1)
 qcut=float(np.median(dq));scut=float(np.median(ds));lock=json.loads((E1/'SELECTION_LOCK.json').read_text())
 protocol={'stage':'E2','base_protocol':str(E1/'PROTOCOL.json'),'main_model':'frozen E1 state-only MLP','thresholds':lock['selected'],'L':100,'k':10,'beam':4,'threads':4,'cache_nodes':10000,'normalization':'float64 mean and population std over existing 894077 training baseline states only; epsilon1e-8; no test fit','query_high_if_l2_le':qcut,'state_high_if_normalized_l2_le':scut,'threshold_source':'medians of same-node history-history event pairs, distinct training queries, both directions; no model decision labels used','train_calibration_pairs':len(left),'constant_feature_indices':np.flatnonzero(std==0).tolist(),'groups':{'G1':'query high / state high','G2':'query high / state low','G3':'query low / state high','G4':'query low / state low'},'primary_population':'all same-node directed history->test model-invoked event pairs','cross_node_control':'one random different-node donor from SAME historical query for each same-node pair; target position and query distance distribution held fixed','false_skip_rate':'P(source Skip | target Keep): model-reference FPR, not ground-truth safety','false_skip_given_reuse':'P(target Keep | source Skip): 1-SkipPrecision','score_difference':'absolute native retain-score difference','predictive_comparison':'logistic models of agreement using log1p(query distance), log1p(state distance), or both; all fitted on training history-history pairs; evaluated on history-test pairs','intervention':'one donor decision substituted at exact target query candidate ordinal+node; identical prefix state checked; fresh MLP thereafter; all unique (target query, target step, donor decision) evaluated once and mapped to donor events','intervention_selection':'all same-node pairs and matched cross-node controls; no outcome-based selection; test pairs used for evaluation only','confidence':'500 target-query cluster bootstrap replicates; history fixed; repeated positions/donors not independent','scope_limit':'state-only model agreement is partly architectural; retrieval safety assessed separately by actual single-replacement searches; not cumulative reuse or deployed lookup timing'}
 dump(R/'PROTOCOL.json',protocol);dump(R/'SELECTION_LOCK.json',{'test_used_to_fit_normalization_or_cutoffs':False,'protocol_sha256':sha(R/'PROTOCOL.json'),'source_e1_lock_sha256':sha(E1/'SELECTION_LOCK.json')})
 log(f'LOCKED train pairs={len(left)} query_cut={qcut:.6f} state_cut={scut:.6f}')
 allout={};interventions=[];keys={};idrows=[]
 testrow={int(q):i for i,q in enumerate(json.loads((E1/'query_splits.json').read_text())['test'])}
 for name in ['MLP','LR','XGBoost']:
  log('ANALYZE '+name)
  h=trace(name,'history');t=trace(name,'test');tau=lock['selected'][name];li,ri=node_pairs(h);cal=pair_data(h,h,li,ri,mean,std,q,tau,qcut,scut)
  li,ri=node_pairs(h,t);same=pair_data(h,t,li,ri,mean,std,q,tau,qcut,scut);ci,cj=cross_control(h,t,li,ri);cross=pair_data(h,t,ci,cj,mean,std,q,tau,qcut,scut)
  out={};models={}
  for kind,cols in [('query_only',['query_distance']),('state_only',['state_distance']),('joint',['query_distance','state_distance'])]:
   model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=SEED));xx=np.column_stack([np.log1p(cal[k]) for k in cols]);model.fit(xx,cal['agreement']);models[kind]=(model,cols)
  for kind,p in [('same_node',same),('cross_node',cross)]:
   np.savez_compressed(R/f'{name}_{kind}_pairs.npz',**p)
   groups=[summarize(p,p['group']==g,f'G{g}') for g in range(1,5)];csvsave(R/f'{name}_{kind}_groups.csv',groups)
   bins=[]
   for variable,calvalues in [('state_distance',cal['state_distance']),('query_distance',cal['query_distance'])]:
    edges=np.quantile(calvalues,[0,.1,.25,.5,.75,.9,1]);edges[0]=-np.inf;edges[-1]=np.inf
    for j in range(6):
     mask=(p[variable]>edges[j])&(p[variable]<=edges[j+1]);r=summarize(p,mask,f'{variable}_{j}');r.update(variable=variable,bin=j,lower=None if not np.isfinite(edges[j]) else float(edges[j]),upper=None if not np.isfinite(edges[j+1]) else float(edges[j+1]));bins.append(r)
   csvsave(R/f'{name}_{kind}_distance_bins.csv',bins)
   predictions={};predmetrics=[]
   for modelname,(model,cols) in models.items():
    pred=model.predict_proba(np.column_stack([np.log1p(p[k]) for k in cols]))[:,1];predictions[modelname]=pred
    predmetrics.append({'predictor':modelname,'auc':float(roc_auc_score(p['agreement'],pred)),'log_loss':float(log_loss(p['agreement'],pred)),'brier':float(brier_score_loss(p['agreement'],pred))})
   np.savez_compressed(R/f'{name}_{kind}_agreement_predictions.npz',target_query=p['target_query'],agreement=p['agreement'],**predictions)
   # Paired query-cluster bootstrap of held-out log-loss improvement.
   y=p['agreement'];loss={k:-(y*np.log(np.clip(v,1e-12,1))+(~y)*np.log(np.clip(1-v,1e-12,1))) for k,v in predictions.items()}
   advantage=loss['query_only']-loss['state_only'];mu,lo,hi=rate_ci(advantage,np.ones(len(y),bool),p['target_query'])
   out[kind]={'groups':groups,'distance_bins':bins,'predictive_metrics':predmetrics,'state_vs_query_logloss_improvement':{'mean':mu,'ci95':[lo,hi]},'spearman_delta_p':{v:float(stats.spearmanr(p[v],p['delta_p']).statistic) for v in ['query_distance','state_distance']},'event_pairs':len(y)}
   if name=='MLP':
    for event,(i,j) in enumerate(zip(p['source_record'],p['target_record'])):
     qid=int(t['query_id'][j]);step=int(t['search_step'][j]);decision=int(h['model_decision'][i]);key=(qid,step,decision)
     if key not in keys:
      keys[key]=len(interventions);interventions.append({'request':len(interventions),'query_id':qid,'query_row':testrow[qid],'step':step,'node':int(t['node_id'][j]),'replacement_skip':decision,'expected_score':float(t['model_score'][j]),'state':t['state_features'][j].tolist()})
     idrows.append({'population':kind,'event':event,'request':keys[key],'group':int(p['group'][event]),'target_query':qid,'source_query':int(h['query_id'][i]),'target_record':int(j),'source_record':int(i),'source_skip':decision,'target_skip':int(t['model_decision'][j])})
  # common target-position support is explicitly reported; no forced complete-case conclusion.
  sets=[set(same['target_record'][same['group']==g].tolist()) for g in range(1,5)]
  out['same_position_support']={'all_four_groups':len(set.intersection(*sets)),'G1_G2':len(sets[0]&sets[1]),'G1_G3':len(sets[0]&sets[2]),'G2_G4':len(sets[1]&sets[3]),'G3_G4':len(sets[2]&sets[3])}
  allout[name]=out
 csvsave(R/'intervention_event_map.csv',idrows)
 # Binary: request,qrow,step,node,skip uint32; expected score,state[17] float32.
 with (R/'interventions.bin').open('wb') as f:
  import struct
  f.write(struct.pack('<I',len(interventions)))
  for r in interventions:f.write(struct.pack('<5I18f',r['request'],r['query_row'],r['step'],r['node'],r['replacement_skip'],r['expected_score'],*r['state']))
 dump(R/'intervention_requests.json',interventions);dump(R/'analysis.json',allout)
 dump(R/'PREPARED.json',{'complete':True,'requests':len(interventions),'event_rows':len(idrows),'same_node_events':allout['MLP']['same_node']['event_pairs'],'cross_node_events':allout['MLP']['cross_node']['event_pairs'],'source_e1_artifact_manifest_sha256':sha(E1/'artifact_manifest.json')})
 log(f'PREPARE COMPLETE: {len(interventions)} unique replacements, {len(idrows)} donor events')
if __name__=='__main__':main()
