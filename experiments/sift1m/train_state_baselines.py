#!/usr/bin/env python3
"""Train two fixed state-only baselines; select risk thresholds on validation only."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='4'
import sys,json,csv,time,warnings,hashlib,importlib.metadata
from pathlib import Path
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
from xgboost import XGBClassifier
from threadpoolctl import threadpool_limits
E=Path(__file__).resolve().parent; DATA=E/'training_data_l100'; OUT=E/'state_baselines'
SEED=42; TARGETS=[('validation_zero_false_skip',0.0),('validation_fnr_0.1pct',0.001),('validation_fnr_1pct',0.01)]
def dump(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False))
def operating(y,p,tau,query_ids):
 skip=(p<tau); safe=(y==0); retain=~safe
 tn=int((skip&safe).sum()); fn=int((skip&retain).sum()); fp=int((~skip&safe).sum()); tp=int((~skip&retain).sum())
 uq,inv=np.unique(query_ids,return_inverse=True); size=np.bincount(inv)
 skipped=np.bincount(inv,weights=skip,minlength=len(uq)); bad=np.bincount(inv,weights=skip&retain,minlength=len(uq))
 pred=(~skip).astype('u1')
 return {'threshold_p_retain_strict_lt':float(tau),'n':len(y),'queries':len(uq),'safe_y0':int(safe.sum()),'retain_y1':int(retain.sum()),'skipped':tn+fn,'safe_skipped':tn,'retain_mislabeled_skipped':fn,'safe_retained':fp,'retain_retained':tp,'skip_rate':float(skip.mean()),'safe_skip_recall':tn/max(1,int(safe.sum())),'retain_false_skip_rate':fn/max(1,int(retain.sum())),'skip_precision_safe':tn/max(1,tn+fn),'skip_contamination_rate':fn/max(1,tn+fn),'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'f1_retain':float(f1_score(y,pred,zero_division=0)),'confusion_matrix_true_rows_pred_columns':[[tn,fp],[fn,tp]],'queries_with_retain_false_skip':int((bad>0).sum()),'queries_with_retain_false_skip_rate':float((bad>0).mean()),'skipped_per_query_mean':float(skipped.mean()),'skipped_per_query_p50':float(np.median(skipped)),'skipped_per_query_p90':float(np.percentile(skipped,90)),'query_mean_skip_rate':float((skipped/size).mean())}
def threshold_from_validation(y,p,risk):
 positive=np.sort(p[y==1]); allowed=int(np.floor(risk*len(positive)))
 tau=float(positive[allowed]) # strict < excludes this first disallowed retain score and all ties
 fn=int(((y==1)&(p<tau)).sum()); assert fn<=allowed
 return {'threshold':tau,'selection_split':'validation','target_retain_false_skip_rate':risk,'validation_retain_budget_count':allowed,'validation_retain_false_skip_count':fn,'decision':'skip iff p(y=1|state) < threshold'}
def ranking(y,p):
 return {'roc_auc_retain':float(roc_auc_score(y,p)),'average_precision_retain':float(average_precision_score(y,p)),'average_precision_safe_skip':float(average_precision_score(1-y,1-p)),'log_loss':float(log_loss(y,p,labels=[0,1])),'brier':float(np.mean((p-y)**2))}
def per_query_csv(path,meta,y,p,tau):
 q,inv=np.unique(meta['query_id'],return_inverse=True); skipped=p<tau; bad=skipped&(y==1)
 counts=np.bincount(inv); sk=np.bincount(inv,weights=skipped).astype(int); miss=np.bincount(inv,weights=bad).astype(int)
 with path.open('w',newline='') as f:
  w=csv.writer(f); w.writerow(['query_id','expansions','predicted_skip','correct_safe_skip','retain_false_skip','skip_rate'])
  w.writerows((int(x),int(n),int(s),int(s-b),int(b),float(s/n)) for x,n,s,b in zip(q,counts,sk,miss))
if __name__=='__main__':
 assert json.loads((DATA/'COMPLETE.json').read_text())['complete']
 manifest=json.loads((DATA/'manifest.json').read_text()); names=manifest['pre_io_state_columns']; assert len(names)==17
 forbidden=['query_id','node_id','oracle','neighbor_was','in_final','admitted','led_to_improvement','current_distance_exact']
 assert not any(any(x in n for x in forbidden) for n in names)
 OUT.mkdir(exist_ok=True); (OUT/'COMPLETE.json').unlink(missing_ok=True)
 splits={}
 for split in ['train','validation','test']:
  folder=DATA/split; x=np.load(folder/'pre_io_state.npy',mmap_mode='r'); y=np.load(folder/'y.npy',mmap_mode='r'); meta=np.load(folder/'metadata.npy',mmap_mode='r')
  assert x.shape==(len(y),17) and np.isfinite(x).all() and set(np.unique(y))=={0,1}
  assert np.array_equal(y,meta['label']); splits[split]={'x':x,'y':y,'meta':meta}
 qsets=[set(map(int,np.unique(splits[s]['meta']['query_id']))) for s in splits]
 assert not(qsets[0]&qsets[1] or qsets[0]&qsets[2] or qsets[1]&qsets[2])
 stats=np.load(DATA/'pre_io_state_train_normalization.npz'); mean=stats['mean']; scale=stats['scale']
 assert np.allclose(mean,splits['train']['x'].mean(axis=0,dtype=np.float64),rtol=1e-6,atol=1e-6)
 versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','xgboost-cpu','joblib','threadpoolctl']}
 requirements='\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n'; (OUT/'requirements.lock.txt').write_text(requirements)
 logistic=LogisticRegression(C=1.0,solver='lbfgs',tol=1e-5,max_iter=1000,class_weight=None,random_state=SEED)
 xgb=XGBClassifier(objective='binary:logistic',tree_method='hist',device='cpu',n_estimators=500,max_depth=6,learning_rate=0.05,min_child_weight=10,subsample=0.8,colsample_bytree=1.0,reg_lambda=1.0,max_bin=256,n_jobs=4,random_state=SEED,eval_metric='logloss',early_stopping_rounds=30)
 summaries={}; combined_rows=[]
 with threadpool_limits(limits=4):
  for model_name,model in [('logistic_regression',logistic),('xgboost',xgb)]:
   folder=OUT/model_name; folder.mkdir(exist_ok=True)
   normalize=model_name=='logistic_regression'
   def transform(a): return np.asarray((a-mean)/scale,dtype=np.float64) if normalize else np.asarray(a,dtype=np.float32)
   train_x=transform(splits['train']['x']); val_x=transform(splits['validation']['x'])
   start=time.perf_counter(); print('Training',model_name,train_x.shape,flush=True)
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    if normalize: model.fit(train_x,splits['train']['y'])
    else: model.fit(train_x,splits['train']['y'],eval_set=[(val_x,splits['validation']['y'])],verbose=50)
   elapsed=time.perf_counter()-start
   assert np.array_equal(model.classes_,[0,1])
   assert not any(issubclass(w.category,ConvergenceWarning) for w in caught), 'Logistic Regression failed to converge'
   # XGBoost's default missing=np.nan is a parameter sentinel, not a data value.
   config=json.loads(json.dumps(model.get_params()),parse_constant=lambda token: token)
   dump(folder/'config.json',config)
   fit={'training_seconds':elapsed,'warnings':[str(w.message) for w in caught]}
   if normalize:
    fit['n_iterations']=int(model.n_iter_[0]); joblib.dump({'model':model,'mean':mean,'scale':scale,'feature_names':names,'state_view':'pre_io'},folder/'model.joblib')
    coefficients=sorted(zip(names,map(float,model.coef_[0])),key=lambda a:abs(a[1]),reverse=True)
    with (folder/'coefficients.csv').open('w',newline='') as f:
     w=csv.writer(f); w.writerow(['feature','standardized_coefficient_positive_means_retain']); w.writerows(coefficients)
    dump(folder/'intercept.json',{'intercept':float(model.intercept_[0])})
   else:
    model.save_model(folder/'model.ubj'); fit['best_iteration']=int(model.best_iteration); fit['best_validation_logloss']=float(model.best_score); fit['trained_boosting_rounds']=len(model.evals_result()['validation_0']['logloss'])
    dump(folder/'training_history.json',model.evals_result())
    with (folder/'feature_importance.csv').open('w',newline='') as f:
     w=csv.writer(f); w.writerow(['feature','normalized_gain']); w.writerows(sorted(zip(names,map(float,model.feature_importances_)),key=lambda a:a[1],reverse=True))
   probabilities={}
   for split in splits:
    x=train_x if split=='train' else val_x if split=='validation' else transform(splits[split]['x'])
    probabilities[split]=model.predict_proba(x)[:,1].astype('<f8')
    assert np.all(np.isfinite(probabilities[split])) and np.all((probabilities[split]>=0)&(probabilities[split]<=1))
    np.save(folder/f'{split}_p_retain.npy',probabilities[split])
   # Lock all thresholds before considering test scores/labels for evaluation.
   chosen={'default_0.5':{'threshold':0.5,'selection_split':'fixed','decision':'skip iff p(y=1|state) < 0.5'}}
   for name,risk in TARGETS: chosen[name]=threshold_from_validation(splits['validation']['y'],probabilities['validation'],risk)
   dump(folder/'thresholds.json',chosen)
   metrics={'model':model_name,'fit':fit,'parameters':config,'inputs':names,'normalization':'train_only_standardization' if normalize else 'raw_state_hist_trees','operating_points':chosen,'splits':{}}
   for split in splits:
    y=splits[split]['y']; p=probabilities[split]; qs=splits[split]['meta']['query_id']
    metrics['splits'][split]={'ranking':ranking(y,p),'operating':{}}
    for name,op in chosen.items():
     val=operating(y,p,op['threshold'],qs); metrics['splits'][split]['operating'][name]=val
     combined_rows.append({'model':model_name,'split':split,'operating_point':name,'roc_auc':metrics['splits'][split]['ranking']['roc_auc_retain'],'average_precision_safe_skip':metrics['splits'][split]['ranking']['average_precision_safe_skip'],**{k:v for k,v in val.items() if k!='confusion_matrix_true_rows_pred_columns'}})
     if split in ['validation','test']: per_query_csv(folder/f'{split}_per_query_{name}.csv',splits[split]['meta'],y,p,op['threshold'])
   with (folder/'threshold_curve.csv').open('w',newline='') as f:
    rows=[]
    for split in ['validation','test']:
     for tau in np.unique(np.r_[np.linspace(0,1,101),[v['threshold'] for v in chosen.values()]]):
      op=operating(splits[split]['y'],probabilities[split],tau,splits[split]['meta']['query_id']); rows.append({'split':split,**{k:v for k,v in op.items() if k!='confusion_matrix_true_rows_pred_columns'}})
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
   # Reload exported model and check that saved probabilities are reproducible.
   verification_x=transform(splits['test']['x'][:256]); expected=probabilities['test'][:256]
   if normalize:
    bundle=joblib.load(folder/'model.joblib'); reloaded=bundle['model']; actual=reloaded.predict_proba(verification_x)[:,1]
   else:
    reloaded=XGBClassifier(); reloaded.load_model(folder/'model.ubj'); actual=reloaded.predict_proba(verification_x)[:,1]
   assert np.array_equal(actual,expected), 'Saved model does not reproduce predictions'
   b=splits['test']['x'][:4096]; transform(b); model.predict_proba(transform(b))
   latency=[]
   for _ in range(5):
    begin=time.perf_counter(); model.predict_proba(transform(b)); latency.append(time.perf_counter()-begin)
   metrics['batch_inference']={'rows':len(b),'repeats':5,'threads':4,'includes_state_transform':True,'mean_seconds':float(np.mean(latency)),'microseconds_per_row_batch':float(np.mean(latency)/len(b)*1e6),'note':'batched CPU inference; not single-node latency or DiskANN speedup'}
   metrics['saved_model_roundtrip_verified']=True
   dump(folder/'metrics.json',metrics); summaries[model_name]=metrics
   print('Completed',model_name,'test AUC',metrics['splits']['test']['ranking']['roc_auc_retain'],flush=True)
 with (OUT/'comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(combined_rows[0])); w.writeheader(); w.writerows(combined_rows)
 provenance={'seed':SEED,'state_view':'pre_io','features':names,'data_manifest':str(DATA/'manifest.json'),'data_manifest_sha256':hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest(),'split_counts':manifest['splits'],'versions':versions,'models':list(summaries),'test_used_for_training_or_selection':False,'label_definition':manifest['label_definition'],'constant_features':['search_L','beam_width'],'threshold_rule':'p_retain < tau; threshold selected using validation retain FNR only','model_selection':'One fixed baseline configuration per model; XGBoost early stopping uses validation logloss only.'}
 dump(OUT/'provenance.json',provenance); dump(OUT/'results.json',summaries)
 report=['# State-only baseline: Logistic Regression / XGBoost','', '使用 L=100 expansion 数据；train/validation/test 分别为 8000/1000/1000 个互斥 query，894077/111715/111841 个样本。仅使用 17 维 pre_io_state，不读取 q/v 向量，不使用 query/node ID、邻居观测或事后标签作为输入。L 和 beam_width 是常量特征。','', 'y=0 为已验证 oracle safe-to-skip，y=1 为保守 retain 标签。模型输出 p_retain；p_retain < tau 才跳过。LR 使用 train-only 标准化、C=1、L-BFGS；XGBoost 使用 CPU hist、depth=6、最多500棵树、learning_rate=.05，validation logloss early stopping=30。没有超参数搜索。','', '## 测试集：默认阈值 0.5','', '| 模型 | ROC-AUC | AP(safe) | Accuracy | Skip rate | Retain 误跳过率 | Skip precision |','|---|---:|---:|---:|---:|---:|---:|']
 for name,m in summaries.items():
  t=m['splits']['test']; op=t['operating']['default_0.5']; r=t['ranking']
  report.append(f"| {name} | {r['roc_auc_retain']:.6f} | {r['average_precision_safe_skip']:.6f} | {op['accuracy']*100:.3f}% | {op['skip_rate']*100:.3f}% | {op['retain_false_skip_rate']*100:.3f}% | {op['skip_precision_safe']*100:.3f}% |")
 report+=['','## 验证集选阈值，固定后评估测试集','', '| 模型 | 验证集 retain FNR 上限 | tau | Test skip rate | Test skipped/query | Test retain 误跳过数 | Test retain FNR | Test skip precision | 有误跳过的 test query 数 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,m in summaries.items():
  for op_name,risk in TARGETS:
   op=m['splits']['test']['operating'][op_name]; tau=m['operating_points'][op_name]['threshold']
   report.append(f"| {name} | {risk*100:.1f}% | {tau:.8g} | {op['skip_rate']*100:.3f}% | {op['skipped_per_query_mean']:.3f} | {op['retain_mislabeled_skipped']} | {op['retain_false_skip_rate']*100:.4f}% | {op['skip_precision_safe']*100:.3f}% | {op['queries_with_retain_false_skip']}/1000 |")
 report+=['', 'Retain FNR = 错误跳过的 y=1 数 / 全部 y=1 数。Skip precision = 正确跳过的 y=0 数 / 所有预测跳过数。两者分母不同。验证集上的风险上限不是测试集安全保证；尤其 validation zero-error 不能证明新 query 零风险。','', '这些结果是原始 baseline 状态上的离线分类结果，尚未把模型放入 DiskANN 搜索环，不能将 skip rate 当作真实在线节省率，也不能推断 Recall@10 不变。y=1 是保守标签，误跳过它表示违背 oracle 保留规则，不一定等于召回损失；真正影响需下一步在线 gate 重跑验证。','', '## 特征与 artifact','', 'LR 的 standardized coefficients、XGBoost 的 feature importance 分别保存在模型子目录 CSV。每个模型有 train/validation/test p_retain、阈值、完整 metrics、每-query 统计与 threshold curve。保存的模型已加载回验，与测试概率逐项一致。requirements.lock.txt 固定本次依赖；provenance.json 记录 split、标签口径和版本。','', '运行：在 learned-gate 中执行 `.venv/bin/python experiments/sift1m/train_state_baselines.py`。仅重新运行推理时使用 `predict_state.py`，输入必须按 provenance 中的17维顺序；LR bundle 含标准化参数，XGBoost 直接用 raw state。','', '实现文档：[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)、[XGBoost sklearn interface](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html)。本报告的全部数值来自本地训练与测试。']
 (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
 dump(OUT/'COMPLETE.json',{'complete':True,'models':['logistic_regression','xgboost'],'state_only':True,'query_disjoint':True,'validation_selected_thresholds':True,'saved_models_verified':True})
 print('STATE BASELINES COMPLETE',flush=True)
 for name,m in summaries.items(): print(name,json.dumps(m['splits']['test'],indent=2),flush=True)
