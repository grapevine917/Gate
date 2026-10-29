from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,csv,time,hashlib,struct
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score,log_loss
import xgboost as xgb
D=Path(__file__).resolve().parent;ROOT=D.parents[1];M=ROOT/'experiments/main_comparison/models';DATA=ROOT/'experiments/sift1m/training_data_l100';NAMES=json.loads((DATA/'manifest.json').read_text())['pre_io_state_columns']
GROUPS={name:[j] for j,name in enumerate(NAMES)}|{'Global11':[1,2,4,5,6,7,8,11,12,15,16],'Candidate6':[0,3,9,10,13,14],'RankProgress':[3,6,7,8],'Distance':[0,1,2,11,12,13,14],'All17':list(range(17))}
MODELS=['LR','MLP','XGB'];REPEATS=10

def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def save(p,rows):
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def status(**kw):dump(D/'STATUS.json',dict(updated=time.strftime('%F %T'),**kw))

def load_models():
 with (M/'lr.bin').open('rb') as f:
  assert f.read(8)==b'LRGATE01' and struct.unpack('<I',f.read(4))[0]==0
  lr=np.fromfile(f,'<f4');assert len(lr)==52
 mu,sd,w,b=lr[:17],lr[17:34],lr[34:51],lr[51]
 def lr_predict(x):return expit(-(((x-mu)/sd)@w+b)).astype('f4')
 with (M/'state.bin').open('rb') as f:
  assert f.read(8)==b'STGATE01';raw=np.fromfile(f,'<f4')
 mm,ss=raw[:17],raw[17:34];at=34;layers=[]
 for a,z in [(17,128),(128,64),(64,1)]:
  ww=raw[at:at+a*z].reshape(a,z);at+=a*z;bb=raw[at:at+z];at+=z;layers.append((ww,bb))
 assert at==len(raw)
 def mlp_predict(x):
  out=[]
  for start in range(0,len(x),4096):
   a=(x[start:start+4096]-mm)/ss
   for j,(ww,bb) in enumerate(layers):
    a=a@ww+bb
    if j<2:np.maximum(a,0,out=a)
   out.append(expit(a[:,0]))
  return np.concatenate(out).astype('f4')
 booster=xgb.Booster();booster.load_model(M/'xgb.ubj');booster.set_param({'nthread':1})
 preds={'LR':lr_predict,'MLP':mlp_predict,'XGB':lambda a:booster.inplace_predict(a)}
 with (M/'native_check.bin').open('rb') as f:n=struct.unpack('<I',f.read(4))[0];check=np.fromfile(f,'<f4').reshape(n,150)
 for name,col in [('LR',0),('XGB',1),('MLP',2)]:np.testing.assert_allclose(preds[name](check[:,129:146].copy()),check[:,146+col],rtol=3e-5,atol=4e-6)
 return preds,(-w).copy()

def metrics(y,p,thresholds):
 p=np.clip(p.astype('f8'),1e-12,1-1e-12);out=dict(auc=float(roc_auc_score(y,p)),logloss=float(log_loss(y,p,labels=[0,1])))
 for risk,t in thresholds.items():
  skipped=p<t;out['skip_'+risk]=float(skipped.mean());out['fnr_'+risk]=float(skipped[y==1].mean())
 return out

def main():
 if (D/'COMPLETE.json').exists():return
 prepared=json.loads((D/'PREPARED_MODEL_METADATA.json').read_text())
 assert all((ROOT/p).stat().st_size==v['bytes'] and (ROOT/p).stat().st_mtime_ns==v['mtime_ns'] for p,v in prepared.items()), 'Prepared model sources changed'
 protocol=dict(dataset='SIFT1M',weights='current main comparison old MIXED-L LR/StateMLP/XGB, trained on1200 queries; same weights as P0-A/B',analysis_scope='existing baseline-L100 VALIDATION pre-I/O states,1000 validation queries, cache misses only; NOT full multi-L evaluation, not an online latency experiment',input='17 pre_io_state columns; no new query/index search; no retraining',comparison='identical rows/labels, raw-feature permutations shared across models',methods='10 repeats of each single feature and jointly shuffled Global11/Candidate6/RankProgress/Distance/All17 groups; frozen original validation thresholds at retain-FNR budgets0.1% and1%',thresholds='reuse frozen p0_old_models_rerun/sift1m thresholds; do not recalibrate after permutation',metrics='AUC drop, logloss increase, fixed-threshold skip/FNR changes; shuffle SD is not a confidence interval',uncertainty='paired query bootstrap of averaged per-query logloss increase,1000 resamples; permutation still may form off-manifold states; no causal or retrained-necessity claim',coefficients='LR standardized coefficients only, not directly comparable with MLP/XGB scores; correlated features can share coefficients',scheduling='start only after expanded P0-B root COMPLETE plus released run.lock; all compute single-thread; no concurrent formal graph timing')
 dump(D/'PROTOCOL.json',protocol)
 paths=[M/n for n in ['lr.bin','state.bin','xgb.ubj','native_check.bin']]+[DATA/'manifest.json',DATA/'validation/pre_io_state.npy',DATA/'validation/y.npy',DATA/'validation/metadata.npy',DATA/'validation/query_ids.npy',ROOT/'experiments/main_comparison/data/validation_ids.npy',ROOT/'experiments/p0_old_models_rerun/sift1m/thresholds.json']
 hashes={str(p.relative_to(ROOT)):sha(p) for p in paths};dump(D/'SOURCE_LOCK.json',hashes)
 ids=np.load(DATA/'validation/query_ids.npy');np.testing.assert_array_equal(ids,np.load(ROOT/'experiments/main_comparison/data/validation_ids.npy'))
 raw=np.load(DATA/'validation/pre_io_state.npy');labels=np.load(DATA/'validation/y.npy');meta=np.load(DATA/'validation/metadata.npy')
 np.testing.assert_array_equal(labels,meta['label']);assert len(ids)==1000 and set(meta['query_id'])==set(ids)
 keep=raw[:,10]==0;x=np.array(raw[keep],dtype='f4',order='C');y=labels[keep];qids=meta['query_id'][keep];uq,qi=np.unique(qids,return_inverse=True);counts=np.bincount(qi);assert len(uq)==1000 and np.isfinite(x).all()
 preds,coefs=load_models();thresholds=json.loads((ROOT/'experiments/p0_old_models_rerun/sift1m/thresholds.json').read_text());keys={'LR':'lr','MLP':'state','XGB':'xgb'}
 ts={m:{str(risk):next(z['tau'] for z in thresholds[keys[m]] if z['risk']==risk) for risk in [.001,.01]} for m in MODELS}
 baseline={m:preds[m](x) for m in MODELS};base_metrics={m:metrics(y,p,ts[m]) for m,p in baseline.items()}
 dump(D/'baseline.json',dict(rows=len(x),queries=len(uq),cached_rows_excluded=int((~keep).sum()),metrics=base_metrics,thresholds=ts,native_export_predictions_verified=True,analysis_split='validation',deployment_L=100))
 logterms=lambda p:-(y*np.log(np.clip(p,1e-12,1-1e-12))+(1-y)*np.log(np.clip(1-p,1e-12,1-1e-12)))
 base_loss={m:logterms(p.astype('f8')) for m,p in baseline.items()};rows=[];summaries=[]
 boot=np.random.default_rng(64011).integers(0,len(uq),size=(1000,len(uq)),dtype='i4')
 for gi,(name,cols) in enumerate(GROUPS.items()):
  losses={m:np.zeros(len(uq)) for m in MODELS};group_rows=[]
  constant=bool(all(np.ptp(x[:,c])==0 for c in cols))
  for rep in range(REPEATS):
   work=x.copy()
   if not constant:
    order=np.random.default_rng(65000+gi*100+rep).permutation(len(x));work[:,cols]=x[order][:,cols]
   for m in MODELS:
    status(phase='permutation',group=name,group_index=gi+1,groups=len(GROUPS),model=m,repeat=rep+1,repeats=REPEATS)
    p=preds[m](work) if not constant else baseline[m];met=metrics(y,p,ts[m]);r=dict(model=m,group=name,n_features=len(cols),repeat=rep,constant=constant,auc_drop=base_metrics[m]['auc']-met['auc'],logloss_increase=met['logloss']-base_metrics[m]['logloss'])
    for risk in ts[m]:r['fnr_change_'+risk]=met['fnr_'+risk]-base_metrics[m]['fnr_'+risk];r['skip_change_'+risk]=met['skip_'+risk]-base_metrics[m]['skip_'+risk]
    rows.append(r);group_rows.append(r);diff=logterms(p.astype('f8'))-base_loss[m];losses[m]+=np.bincount(qi,weights=diff)/counts/REPEATS
  for m in MODELS:
   rs=[r for r in group_rows if r['model']==m];out=dict(model=m,group=name,n_features=len(cols),constant=constant)
   for k in ['auc_drop','logloss_increase','fnr_change_0.001','skip_change_0.001','fnr_change_0.01','skip_change_0.01']:
    a=np.array([r[k] for r in rs]);out[k+'_mean']=float(a.mean());out[k+'_shuffle_sd']=float(a.std(ddof=1))
   ci=np.percentile(losses[m][boot].mean(axis=1),[2.5,97.5]);out.update(query_equal_logloss_increase=float(losses[m].mean()),query_bootstrap_logloss_lo=float(ci[0]),query_bootstrap_logloss_hi=float(ci[1]))
   summaries.append(out)
  save(D/'permutation_runs.csv',rows);save(D/'importance_summary.csv',summaries);print(time.strftime('%T'),'GROUP COMPLETE',name,flush=True)
 # Signed standardized LR weights supplement shared permutation analysis.
 save(D/'lr_coefficients.csv',[dict(feature=n,standardized_coefficient=float(coefs[j]),abs_coefficient=float(abs(coefs[j])),constant_on_analysis=bool(np.ptp(x[:,j])==0)) for j,n in enumerate(NAMES)])
 agreement=[]
 for i,m in enumerate(MODELS):
  for n in MODELS[i+1:]:
   aa={r['group']:r for r in summaries if r['model']==m and r['group'] in NAMES and not r['constant']};bb={r['group']:r for r in summaries if r['model']==n and r['group'] in aa};names=list(aa);agreement.append(dict(model_a=m,model_b=n,variable_features=len(names),permutation_auc_spearman=float(spearmanr([aa[k]['auc_drop_mean'] for k in names],[bb[k]['auc_drop_mean'] for k in names]).statistic)))
 save(D/'ranking_agreement.csv',agreement)
 os.environ['MPLCONFIGDIR']=str(D/'mpl_cache')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 order=sorted(NAMES,key=lambda name:-np.mean([r['auc_drop_mean'] for r in summaries if r['group']==name]))
 fig,axes=plt.subplots(1,3,figsize=(16,8),sharey=True)
 for ax,m in zip(axes,MODELS):
  byname={r['group']:r for r in summaries if r['model']==m};a=[byname[n]['auc_drop_mean'] for n in order];sd=[byname[n]['auc_drop_shuffle_sd'] for n in order];ax.barh(range(17),a,xerr=sd);ax.set(yticks=range(17),yticklabels=order,xlabel='AUC decrease after permutation',title=m);ax.axvline(0,color='gray',lw=.5)
 axes[0].invert_yaxis();fig.suptitle('Gate feature reliance: same mixed-L models, same SIFT1M validation L100 states');fig.tight_layout()
 for ext in ['png','svg','pdf']:fig.savefig(D/f'importance_comparison.{ext}',dpi=160)
 plt.close(fig)
 lines=['# 当前主模型的统一特征依赖分析','','SIFT1M旧混合L权重，验证集基线L100未缓存状态。固定模型，未重训、未进行新在线搜索。Permutation及阈值指标均为代理retain标签指标，不是最终Recall或物理IO收益。','','## 基准','', '|模型|AUC|Logloss|验证冻结阈值1%点实际FNR|','|---|---:|---:|---:|']
 for m in MODELS:lines.append(f'| {m} | {base_metrics[m]["auc"]:.6f} | {base_metrics[m]["logloss"]:.6f} | {100*base_metrics[m]["fnr_0.01"]:.4f}% |')
 lines+=['','## 单特征重要性','','同一列、同一轮的随机排列在三模型之间完全一致。误差条是10次排列的标准差，不是置信区间。','', '![统一特征分析](importance_comparison.png)','']
 for m in MODELS:
  rs=sorted([r for r in summaries if r['model']==m and r['group'] in NAMES],key=lambda r:-r['auc_drop_mean']);lines += [f'### {m}','','|特征|AUC下降均值|排列SD|Logloss增加|','|---|---:|---:|---:|']
  for r in rs:lines.append(f'|{r["group"]}|{r["auc_drop_mean"]:.6f}|{r["auc_drop_shuffle_sd"]:.6f}|{r["logloss_increase_mean"]:.6f}|')
  lines.append('')
 lines+=['## 解释边界','','原先的XGB分析使用纯L100旧模型且观察过测试集；本次使用当前主实验的混合L权重和验证状态，不能把两次变化单独归因于解释方法。特征相关时，单列打乱可能产生不自然组合；组内联合打乱保留组内部关联，但仍破坏其与其他状态和标签的关系。重要性反映固定模型依赖，不证明因果关系、最优子集或删除重训后的性能。L/Beam及cache在本分析子集为常量，零重要性只适用于该切片。基线轨迹也不等于在线剪枝后的状态分布。对LR系数亦需考虑标准化和共线性。','','完整组分析、冻结阈值变化与按query bootstrap区间见CSV。按query区间反映该组验证查询的采样变动，不包含训练种子不确定性。']
 (D/'REPORT.md').write_text('\n'.join(lines)+'\n')
 assert all(sha(ROOT/p)==h for p,h in hashes.items())
 dump(D/'COMPLETE.json',dict(complete=True,models=MODELS,rows=len(x),queries=1000,groups=len(GROUPS),repeats=10,report_sha256=sha(D/'REPORT.md'),finished=time.strftime('%F %T')))
 status(phase='complete',complete=True)
if __name__=='__main__':main()
