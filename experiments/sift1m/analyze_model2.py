#!/usr/bin/env python3
"""Verify M2 inference and estimate paired AUC uncertainty by resampling queries."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from predict_model2 import interaction_features, predict_pairs

E = Path(__file__).resolve().parent
DATA = E/'training_data_l100'
OUT = E/'model2'

def main():
    assert json.loads((OUT/'COMPLETE.json').read_text())['complete']
    manifest = json.loads((DATA/'manifest.json').read_text())
    meta = np.load(DATA/'test/metadata.npy', mmap_mode='r')
    y = np.load(DATA/'test/y.npy', mmap_mode='r')
    query = np.load(DATA/'test/query_vectors.npy', mmap_mode='r')
    spec = manifest['base_vectors']
    base = np.memmap(spec['path'], mode='r', dtype=spec['dtype'], offset=spec['offset_bytes'], shape=tuple(spec['shape']))
    bundle = joblib.load(OUT/'model.joblib')
    scores = {name:np.load(E/folder/'test_p_retain.npy')
              for name,folder in [('M1','model1'),('M2','model2'),('state MLP','state_mlp')]}
    with threadpool_limits(limits=4):
        for i in range(0,len(meta),8192):
            m = meta[i:i+8192]
            actual = predict_pairs(query[m['query_row']],base[m['node_id']],bundle)
            assert np.array_equal(actual,scores['M2'][i:i+len(m)])
    # Independent hand calculation checks feature ordering and raw interactions.
    q=np.ones((2,128), dtype='float32')*2
    v=np.ones((2,128), dtype='float32')*3
    f=interaction_features(q,v)
    np.testing.assert_array_equal(f,np.concatenate((q,v,np.ones_like(q),np.ones_like(q)*6),axis=1))
    for badq,badv in [(q[:,:127],v),(q,np.full_like(v,np.nan))]:
        try: interaction_features(badq,badv)
        except ValueError: pass
        else: raise AssertionError('Invalid input accepted')
    provenance = [json.loads((E/f/'provenance.json').read_text()) for f in ('model1','model2','state_mlp')]
    assert len({p['data_manifest_sha256'] for p in provenance})==1
    assert all(p['splits']==provenance[0]['splits'] for p in provenance)
    assert all(p['parameters']==provenance[0]['parameters'] for p in provenance)
    for folder in ('model1','state_mlp'):
        old=json.loads((E/folder/'metrics.json').read_text())['fit']
        new=json.loads((OUT/'metrics.json').read_text())['fit']
        assert all(old[k]==new[k] for k in ('max_epochs','patience','min_delta','selection'))
    _, qi=np.unique(meta['query_id'],return_inverse=True)
    nquery=int(qi.max())+1
    # Weighted rank AUC with exact tie handling; query bootstrap preserves within-query dependence.
    prepared={}
    for name,p in scores.items():
        order=np.argsort(p,kind='stable')
        starts=np.r_[0,np.flatnonzero(np.diff(p[order]))+1]
        prepared[name]=(order,starts)
    def auc(name, query_weight):
        order,starts=prepared[name]
        w=query_weight[qi[order]]
        pos=np.add.reduceat(w*y[order],starts)
        neg=np.add.reduceat(w*(1-y[order]),starts)
        return float(np.sum(pos*(np.cumsum(neg)-0.5*neg))/(pos.sum()*neg.sum()))
    for name,p in scores.items():
        np.testing.assert_allclose(auc(name,np.ones(nquery)),roc_auc_score(y,p),rtol=1e-12)
    rng=np.random.default_rng(42)
    samples={n:[] for n in scores}
    for _ in range(500):
        weights=np.bincount(rng.integers(0,nquery,nquery),minlength=nquery)
        for name in scores: samples[name].append(auc(name,weights))
    ci={n:dict(auc=float(roc_auc_score(y,scores[n])),query_bootstrap_95pct_ci=np.percentile(a,[2.5,97.5]).tolist()) for n,a in samples.items()}
    diffs={}
    for a,b in [('M2','M1'),('state MLP','M2')]:
        delta=np.array(samples[a])-samples[b]
        diffs[f'{a} minus {b}']=dict(delta_auc=ci[a]['auc']-ci[b]['auc'],query_bootstrap_95pct_ci=np.percentile(delta,[2.5,97.5]).tolist())
    result=dict(seed=42,bootstrap_repeats=500,resampling_unit='query',
                thresholds_and_models_fixed=True,includes_training_seed_uncertainty=False,
                all_raw_test_predictions_exactly_verified=True,identical_splits_and_training_protocol=True,
                auc=ci,paired_auc_differences=diffs)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    gap=ci['state MLP']['auc']-ci['M1']['auc']
    recovered=(ci['M2']['auc']-ci['M1']['auc'])/gap
    appendix=['','## M1 vs M2 interpretation','',
              f'M2-M1 AUC 差为 {diffs["M2 minus M1"]["delta_auc"]:.6f}；按 query 配对 bootstrap 的 95% 区间为 {diffs["M2 minus M1"]["query_bootstrap_95pct_ci"]}。',
              f'M2 弥补 M1 到 state MLP 的 AUC 差距约 {recovered*100:.1f}%（描述性比例）。',
              '交互项不引入新信息；提升意味着这种表示使关系更容易被当前 MLP 学到。剩余差距不能单凭本次实验归因于信息不足。',
              'M1 参数量 41217，M2 参数量 73985；输入扩展与首层容量变化共同构成此次对照。单个训练 seed 的结果尚不覆盖训练随机性。',
              'bootstrap 区间只描述固定模型在新 query 上的采样不确定性。',
              '独立验证：全部 111841 原始测试 q/v 对复现导出概率，特征顺序及公式核对通过，三个 MLP 的数据源与训练配置一致。',
              '重现分析：`.venv/bin/python experiments/sift1m/analyze_model2.py`。']
    report=OUT/'REPORT.md'
    text=report.read_text().split('\n## M1 vs M2 interpretation')[0]
    report.write_text(text+'\n'.join(appendix)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
