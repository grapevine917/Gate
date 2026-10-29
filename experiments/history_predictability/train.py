"""Conditional prediction on causal frozen-policy trajectories; validation-only thresholds."""
from pathlib import Path
import json,hashlib,struct
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'io_predictability'
BASE=['candidate_pq','best_pq','worst_pq','rank','queue_size','unexpanded','visited','expanded','beam_round','beam_position','cached','best_exact','has_exact','pq_over_best','pq_over_worst','L','beam']
EXTRA=['seen','skipped','skip_fraction','skip_streak','since_skip','has_skip','skip_pq_mean','skip_pq_std','skip_pq_min','skip_pq_max','skip_pq_last','pq_minus_skip_mean','pq_over_skip_mean']+['delta_'+BASE[j] for j in [0,1,2,3,4,5,6,7,8,11,13,14]]+['has_previous','new_visited_per_expansion']
assert len(BASE+EXTRA)==44
GROUPS={'retrained_17_lr':list(range(17)),'pruning_history_lr':list(range(23)),'distance_history_lr':list(range(17))+list(range(22,30)),'search_changes_lr':list(range(17))+list(range(30,44)),'full_44_lr':list(range(44)),'full_44_hgb':list(range(44))}
def load(ds,split):
 r=np.loadtxt(ROOT/ds/split/'labels.csv',delimiter=',',skiprows=1)
 q=r[:,0].astype(int);x=r[:,7:];crit=r[:,4]<r[:,3];y=(~crit)&(r[:,6]<r[:,5])
 assert x.shape[1]==44 and np.isfinite(x).all() and len(np.unique(q))==300 and np.all(x[:,10]==0)
 assert len(set(zip(q,r[:,1])))==len(r)
 return {'x':x,'q':q,'critical':crit,'y':y,'rows':r}
def original(ds,x):
 with open(OLD/ds/'state_lr/model.bin','rb') as f:
  assert f.read(8)==b'LRGATE01' and struct.unpack('<I',f.read(4))[0]==0
  mean=np.fromfile(f,'<f4',17);scale=np.fromfile(f,'<f4',17);w=np.fromfile(f,'<f4',17);b=np.fromfile(f,'<f4',1)[0]
 return 1-expit(-(((x[:,:17]-mean)/scale)@w+b)).astype(np.float32)
def metric(d,p,t):
 skip=p>t;y=d['y'];c=d['critical'];n=int(skip.sum());tp=int((skip&y).sum());fp=int((skip&c).sum())
 return {'threshold':float(t),'skip_count':n,'skip_fraction':float(skip.mean()),'safe_saving_coverage':tp/max(1,int(y.sum())),'critical_skips':fp,'critical_count':int(c.sum()),'critical_skip_rate':fp/max(1,int(c.sum())),'safe_saving_precision':tp/max(1,n),'non_saving_safe_skips':int((skip&~c&~y).sum()),'original_skipped_critical_count':int((c&(d['rows'][:,2]!=0)).sum()),'original_skipped_critical_still_skipped':int((skip&c&(d['rows'][:,2]!=0)).sum()),'original_skips_retained_fraction':float(skip[d['rows'][:,2]!=0].mean())}
def thresholds(d,p):
 result={}
 for coverage in [.1,.25,.5]:
  a=np.sort(p[d['y']]);k=max(1,int(np.floor(len(a)*coverage)));result[f'coverage_{coverage:g}']=float(np.nextafter(a[-k],-np.inf))
 for budget in [.001,.005,.01]:
  a=np.sort(p[d['critical']])[::-1];k=int(np.floor(len(a)*budget));result[f'critical_budget_{budget:g}']=float(a[min(k,len(a)-1)]) if len(a) else 1.0
 return result
summary={};predictions={}
for ds in ['sift','deep']:
 data={s:load(ds,s) for s in ['train','validation','test']}
 ids={s:set(np.load(ROOT/ds/s/'query_ids.npy').tolist()) for s in data};assert not any(ids[a]&ids[b] for a,b in [('train','test'),('validation','test'),('train','validation')])
 models={};allpred={};allth={}
 for name in ['original_17_lr']+list(GROUPS):
  if name=='original_17_lr':scores={s:original(ds,d['x']) for s,d in data.items()};cols=list(range(17));coef=None
  else:
   cols=GROUPS[name];scaler=StandardScaler().fit(data['train']['x'][:,cols]);xx=scaler.transform(data['train']['x'][:,cols]);y=data['train']['y'];model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=1,min_samples_leaf=100,random_state=42) if name.endswith('hgb') else LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=42)
   model.fit(xx,y);scores={s:model.predict_proba(scaler.transform(d['x'][:,cols]))[:,1] for s,d in data.items()};coef=None if name.endswith('hgb') else dict(zip([ (BASE+EXTRA)[i] for i in cols],model.coef_[0].tolist()))
   if name in ['retrained_17_lr','full_44_lr']:
    dest=ROOT/ds/(name+'.bin')
    with open(dest,'wb') as f:
     f.write(b'LRGATE01' if name=='retrained_17_lr' else b'HLGATE01');f.write(struct.pack('<I',0 if name=='retrained_17_lr' else 44))
     for a in [scaler.mean_,scaler.scale_,model.coef_[0],model.intercept_]:f.write(np.asarray(a,dtype='<f4').tobytes())
   import joblib
   joblib.dump({'model':model,'scaler':scaler,'columns':cols},ROOT/ds/f'{name}.joblib')
  th=thresholds(data['validation'],scores['validation']);allpred[name]=scores;allth[name]=th
  models[name]={'features':[(BASE+EXTRA)[i] for i in cols],'coefficients':coef,'thresholds':th,'metrics':{s:{'roc_auc':float(roc_auc_score(d['y'],scores[s])),'ap':float(average_precision_score(d['y'],scores[s])),'operating_points':{k:metric(d,scores[s],t) for k,t in th.items()}} for s,d in data.items() if s!='train'}}
  print(ds,name,'AP',models[name]['metrics']['test']['ap'],flush=True)
 # Score parity with actual frozen LR native decisions, including round-to-float before strict inequality.
 tau=json.loads((OLD/ds/'state_lr/thresholds.json').read_text())['unsafe_fpr_0.005']['retain_threshold']
 parity={}
 for s,d in data.items():
  native=1-original(ds,d['x']);actual=d['rows'][:,2]!=0;pred=native<tau
  # Computing 1-p introduces rounding; use direct native score for exact parity.
  with open(OLD/ds/'state_lr/model.bin','rb') as f:
   f.read(12);m=np.fromfile(f,'<f4',17);sc=np.fromfile(f,'<f4',17);w=np.fromfile(f,'<f4',17);b=np.fromfile(f,'<f4',1)[0]
  pred=expit(-(((d['x'][:,:17]-m)/sc)@w+b)).astype(np.float32)<tau
  parity[s]={'decision_mismatches':int((actual!=pred).sum()),'events':len(pred)};assert np.array_equal(actual,pred)
 # Cluster bootstrap comparison, using frozen thresholds; resample complete queries.
 test=data['test'];key='coverage_0.25';c=test['critical'];q=test['q'];den=np.bincount(q,weights=c,minlength=300)
 num=[]
 for n in ['retrained_17_lr','full_44_lr']:
  skip=allpred[n]['test']>allth[n][key];num.append(np.bincount(q,weights=(skip&c),minlength=300))
 rng=np.random.default_rng(42);deltas=[]
 for _ in range(1000):
  ix=rng.integers(0,300,300);denom=den[ix].sum()
  if denom:deltas.append(float((num[1][ix].sum()-num[0][ix].sum())/denom))
 comparison={'definition':'full44 minus retrained17 critical skip rate at validation 25% safe-saving coverage','difference':float((num[1].sum()-num[0].sum())/den.sum()),'query_bootstrap_95_ci':np.quantile(deltas,[.025,.975]).tolist(),'bootstrap_queries':300,'replicates':1000}
 summary[ds]={'splits':{s:{'queries':300,'events':len(d['y']),'critical':int(d['critical'].sum()),'safe_saving':int(d['y'].sum()),'previously_skipped':int(d['rows'][:,2].sum())} for s,d in data.items()},'native_score_parity':parity,'models':models,'primary_comparison':comparison}
 (ROOT/'results.json').write_text(json.dumps(summary,indent=2))
 np.savez_compressed(ROOT/ds/'test_predictions.npz',**{n:v['test'] for n,v in allpred.items()},query=test['q'],critical=c,y=test['y'])
lines=['# 剪枝历史可预测性实验','','真实冻结 LR 剪枝路径；每数据集 train/validation/test 各300 queries。关键标签为当前候选强制跳过比强制保留损失 Recall@10，之后继续冻结策略。特征均在当前读取前产生。完整方法见 PROTOCOL.md。','','## 主比较：验证安全省请求覆盖率25%，冻结阈值后测试','','|数据集|模型|测试AP|安全省请求覆盖率|关键误剪/关键数|关键误剪率|','|---|---|---:|---:|---:|---:|']
for ds,r in summary.items():
 for name,m in r['models'].items():
  z=m['metrics']['test'];v=z['operating_points']['coverage_0.25'];lines.append(f"|{ds}|{name}|{z['ap']:.4f}|{v['safe_saving_coverage']:.2%}|{v['critical_skips']}/{v['critical_count']}|{v['critical_skip_rate']:.2%}|")
lines+=['','## 原LR已经误剪的关键候选（相同25%验证覆盖阈值）','','|数据集|模型|仍被判定跳过/原LR误剪关键候选数|原LR剪枝决定保留比例|','|---|---|---:|---:|']
for ds,r in summary.items():
 for name,m in r['models'].items():
  v=m['metrics']['test']['operating_points']['coverage_0.25'];lines.append(f"|{ds}|{name}|{v['original_skipped_critical_still_skipped']}/{v['original_skipped_critical_count']}|{v['original_skips_retained_fraction']:.2%}|")
lines+=['','这里的“保留比例”指接受原LR的skip决定比例。该子组关键事件少，不能仅凭几个节点的差异宣称稳定保护。','']
lines+=['','## 按 query 重采样的不确定性','']
for ds,r in summary.items():
 c=r['primary_comparison'];lo,hi=c['query_bootstrap_95_ci'];lines.append(f"- {ds}: 完整44维 LR 减去重训17维 LR 的关键误剪率差 {100*c['difference']:.3f} 个百分点，95% bootstrap区间 [{100*lo:.3f}, {100*hi:.3f}] 个百分点。")
lines+=['','## 关键误剪预算0.5%（验证集选择）','','|数据集|模型|测试安全省请求覆盖率|测试关键误剪率|关键误剪数|','|---|---|---:|---:|---:|']
for ds,r in summary.items():
 for name,m in r['models'].items():
  v=m['metrics']['test']['operating_points']['critical_budget_0.005'];lines.append(f"|{ds}|{name}|{v['safe_saving_coverage']:.2%}|{v['critical_skip_rate']:.2%}|{v['critical_skips']}|")
lines+=['','本实验是冻结策略分布上的条件可预测性检验；不是增强策略在线部署的净请求/Recall实测。不同测试覆盖率必须与误剪率一起解读。300条测试查询和稀少关键节点使严格预算下的不确定性较大。不能据零次误剪宣称保证安全，也不能把逐节点读取差累加成查询净收益。完整阈值、消融、标签规模和分数一致性校验在results.json。']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
paths=[ROOT/'PROTOCOL.md',Path(__file__),ROOT/'collect.py',ROOT/'results.json',ROOT/'query_splits.json']
paths+=list(ROOT.glob('*/*/labels.csv'))
(ROOT/'CLASSIFICATION_COMPLETE.json').write_text(json.dumps({'complete':True,'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2))
