#!/usr/bin/env python3
"""Effective XGBoost model: gain, held-out permutation AUC drop, exact TreeSHAP."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import csv,json,hashlib,time,importlib.metadata
from pathlib import Path
import numpy as np
import xgboost as xgb
from scipy.special import expit
from scipy.stats import spearmanr,rankdata
from sklearn.metrics import roc_auc_score,log_loss
E=Path(__file__).resolve().parent;DATA=E/'training_data_l100';OUT=E/'state_feature_importance'

def dump(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
    OUT.mkdir(exist_ok=True);(OUT/'COMPLETE.json').unlink(missing_ok=True)
    manifest=json.loads((DATA/'manifest.json').read_text());names=manifest['pre_io_state_columns'];assert len(names)==17
    x=np.load(DATA/'test/pre_io_state.npy',mmap_mode='r');y=np.load(DATA/'test/y.npy',mmap_mode='r')
    meta=np.load(DATA/'test/metadata.npy',mmap_mode='r');qids,qi=np.unique(meta['query_id'],return_inverse=True)
    assert x.shape==(111841,17) and len(qids)==1000 and np.array_equal(y,meta['label'])
    original=xgb.Booster();original.load_model(E/'state_baselines/xgboost/model.ubj')
    best=int(original.attr('best_iteration'));booster=original[:best+1];booster.set_param({'nthread':4})
    booster.save_model(OUT/'effective_model.ubj')
    original_rounds=original.num_boosted_rounds()
    p=booster.inplace_predict(np.array(x,copy=True))
    np.testing.assert_array_equal(p,np.load(E/'state_baselines/xgboost/test_p_retain.npy'))
    baseline_auc=float(roc_auc_score(y,p));baseline_loss=float(log_loss(y,p))
    print('Prediction verified; rounds',best+1,'test AUC',baseline_auc,flush=True)
    scores={kind:booster.get_score(importance_type=kind) for kind in ['gain','total_gain','weight','cover']}
    gain=np.array([scores['gain'].get(f'f{i}',0) for i in range(17)])
    total_gain=np.array([scores['total_gain'].get(f'f{i}',0) for i in range(17)])
    normalized_gain=gain/gain.sum();normalized_total_gain=total_gain/total_gain.sum()
    constants=np.ptp(x,axis=0)==0
    repeats=10;rng=np.random.default_rng(42);permutation=np.zeros((17,repeats));loss_inc=np.zeros_like(permutation)
    work=np.array(x,copy=True)
    for j,name in enumerate(names):
        started=time.perf_counter()
        if not constants[j]:
            for repeat in range(repeats):
                order=rng.permutation(len(x));work[:,j]=x[order,j]
                perm_p=booster.inplace_predict(work)
                permutation[j,repeat]=baseline_auc-roc_auc_score(y,perm_p)
                loss_inc[j,repeat]=log_loss(y,perm_p)-baseline_loss
            work[:,j]=x[:,j]
        np.save(OUT/'permutation_auc_drops.npy',permutation)
        np.save(OUT/'permutation_logloss_increases.npy',loss_inc)
        print('Permutation',name,'mean AUC drop',permutation[j].mean(),'seconds',time.perf_counter()-started,flush=True)
    # Exact native TreeSHAP, all test rows, no approximate contribution algorithm.
    contrib=np.lib.format.open_memmap(OUT/'test_shap_values_with_bias.npy',mode='w+',dtype='<f4',shape=(len(x),18))
    max_margin_error=0.0;max_probability_error=0.0
    for i in range(0,len(x),4096):
        d=xgb.DMatrix(np.array(x[i:i+4096],copy=True));s=booster.predict(d,pred_contribs=True,approx_contribs=False)
        margin=booster.predict(d,output_margin=True)
        err=float(abs(s.sum(axis=1)-margin).max());max_margin_error=max(max_margin_error,err)
        np.testing.assert_allclose(s.sum(axis=1),margin,rtol=1e-5,atol=1e-5)
        pp=expit(s.sum(axis=1));max_probability_error=max(max_probability_error,float(abs(pp-p[i:i+len(s)]).max()))
        np.testing.assert_allclose(pp,p[i:i+len(s)],rtol=1e-5,atol=2e-6)
        contrib[i:i+len(s)]=s
        if i%20480==0:print('Exact TreeSHAP',i+len(s),'/',len(x),flush=True)
    contrib.flush()
    shap=np.asarray(contrib[:,:17]);mean_abs=np.abs(shap).mean(axis=0,dtype='float64')
    shap_signed=shap.mean(axis=0,dtype='float64')
    # Equal-query summaries and cluster bootstrap uncertainty for SHAP importance.
    counts=np.bincount(qi);query_abs=np.zeros((1000,17))
    for j in range(17):query_abs[:,j]=np.bincount(qi,weights=np.abs(shap[:,j]))/counts
    np.save(OUT/'query_mean_abs_shap.npy',query_abs)
    boots=np.random.default_rng(42).integers(0,1000,size=(1000,1000))
    boot_mean=np.array([query_abs[ix].mean(axis=0) for ix in boots])
    ci=np.percentile(boot_mean,[2.5,97.5],axis=0)
    variable=np.flatnonzero(~constants)
    pearson=np.full((17,17),np.nan);spearman=np.full((17,17),np.nan)
    pearson[np.ix_(variable,variable)]=np.corrcoef(x[:,variable],rowvar=False)
    spearman[np.ix_(variable,variable)]=spearmanr(x[:,variable]).statistic
    correlations=[]
    for j in variable:
        for k in variable:
            if k<=j:continue
            if abs(spearman[j,k])>=0.7:
                correlations.append(dict(feature_a=names[j],feature_b=names[k],spearman=float(spearman[j,k]),pearson=float(pearson[j,k])))
    correlations.sort(key=lambda r:abs(r['spearman']),reverse=True)
    np.savez(OUT/'feature_correlations.npz',pearson=pearson,spearman=spearman)
    dump(OUT/'correlated_feature_pairs.json',correlations)
    means=permutation.mean(axis=1);stds=permutation.std(axis=1,ddof=1)
    ranks={kind:rankdata(-vals,method='min').astype(int) for kind,vals in
           [('gain',normalized_gain),('permutation',means),('shap',mean_abs)]}
    rows=[]
    for j,name in enumerate(names):
        direction=None if constants[j] or np.ptp(shap[:,j])==0 else float(spearmanr(x[:,j],shap[:,j]).statistic)
        rows.append(dict(feature=name,constant_on_test=bool(constants[j]),gain=float(gain[j]),gain_fraction=float(normalized_gain[j]),
            total_gain_fraction=float(normalized_total_gain[j]),split_count=int(scores['weight'].get(f'f{j}',0)),
            permutation_auc_drop_mean=float(means[j]),permutation_auc_drop_std=float(stds[j]),
            permutation_logloss_increase_mean=float(loss_inc[j].mean()),shap_mean_abs_logodds=float(mean_abs[j]),
            shap_mean_signed_logodds=float(shap_signed[j]),shap_query_mean_abs_logodds=float(query_abs[:,j].mean()),
            shap_query_mean_abs_95ci_low=float(ci[0,j]),shap_query_mean_abs_95ci_high=float(ci[1,j]),
            shap_feature_value_spearman=direction,gain_rank=int(ranks['gain'][j]),permutation_rank=int(ranks['permutation'][j]),shap_rank=int(ranks['shap'][j])))
    dump(OUT/'importance.json',rows)
    with (OUT/'importance.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(sorted(rows,key=lambda r:r['shap_rank']))
    dump(OUT/'provenance.json',dict(model='existing state-only XGBoost',original_rounds=original_rounds,effective_rounds=best+1,
        seed=42,test_rows=len(x),test_queries=1000,feature_names=names,baseline_auc=baseline_auc,baseline_logloss=baseline_loss,
        permutation_repeats=repeats,permutation_method='independent row shuffle of one raw feature; fixed model; all held-out rows',
        permutation_std='variation across shuffles, not confidence interval or training-seed uncertainty',
        shap_method='native exact XGBoost TreeSHAP pred_contribs=True, approx_contribs=False; tree-path-dependent, log-odds retain output',
        shap_rows=len(x),shap_bootstrap_unit='query',shap_bootstrap_repeats=1000,
        max_shap_additivity_error=max_margin_error,max_shap_probability_error=max_probability_error,
        original_model_sha256=hashlib.sha256((E/'state_baselines/xgboost/model.ubj').read_bytes()).hexdigest(),
        data_manifest_sha256=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest(),
        versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','xgboost-cpu','scikit-learn']}))
    lines=['# 17-state-feature importance: XGBoost','','使用已训练模型实际预测的前250轮，全部 held-out test（111841 expansion、1000 query），无重新训练/选参。',
        f'预测逐项复现保存概率，test AUC={baseline_auc:.6f}。Gain 用平均分裂增益归一化；另保存 total_gain 和 split count。',
        'Permutation：逐个特征随机打乱，重复10次，重要性为 baseline AUC − 打乱后AUC。也记录 logloss 增量。',
        'SHAP：XGBoost 原生精确 TreeSHAP（tree-path-dependent），全部测试样本，mean(|SHAP|) 以 retain log-odds 衡量；正SHAP推向保留，负SHAP推向跳过。', '',
        '| Feature | Gain % | Permutation ΔAUC ± SD | Mean abs SHAP | Gain rank | Perm rank | SHAP rank |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(rows,key=lambda r:r['shap_rank']):
        lines.append(f'| {r["feature"]} | {100*r["gain_fraction"]:.3f} | {r["permutation_auc_drop_mean"]:.6f} ± {r["permutation_auc_drop_std"]:.6f} | {r["shap_mean_abs_logodds"]:.6f} | {r["gain_rank"]} | {r["permutation_rank"]} | {r["shap_rank"]} |')
    lines += ['', '三种方法衡量不同量，数值不可直接相加或相互比较。Gain 受分裂选择影响；Permutation 打乱相关特征会产生不自然状态；TreeSHAP 的路径依赖背景不是因果解释。',
        'search_L 和 beam_width 在本数据中为常量，零重要性不能说明改变L/beam没有效果。has_best_exact 是否被模型使用可从 split_count 和 SHAP 查看。',
        'query_mean_abs SHAP及按query bootstrap区间保存在CSV；常规mean_abs按expansion加权。Permutation的SD只反映打乱随机性。',
        '历史 feature_importance.csv 使用了导出booster所有280轮；本分析统一250轮，因此gain数值可能略有不同。',
        '完整SHAP（最后一列是bias）、相关系数、10轮permutation数值和有效模型均已保留。',
        '重现：`.venv/bin/python experiments/sift1m/analyze_state_features.py`。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    dump(OUT/'COMPLETE.json',dict(complete=True,gain=True,permutation=True,exact_treeshap=True,all_test_rows=True,prediction_and_shap_additivity_verified=True))
    print('STATE FEATURE IMPORTANCE COMPLETE',flush=True)

if __name__=='__main__':main()
