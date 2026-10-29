#!/usr/bin/env python3
"""Verify M3 raw-input inference and report held-out risk operating points."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
import csv
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from predict_model3 import predict_joint, joint_features

E=Path(__file__).resolve().parent
OUT=E/'model3'
DATA=E/'training_data_l100'

def main():
    assert json.loads((OUT/'COMPLETE.json').read_text())['complete']
    metrics=json.loads((OUT/'metrics.json').read_text())
    manifest=json.loads((DATA/'manifest.json').read_text())
    folder=DATA/'test'
    meta=np.load(folder/'metadata.npy',mmap_mode='r')
    q=np.load(folder/'query_vectors.npy',mmap_mode='r')
    state=np.load(folder/'pre_io_state.npy',mmap_mode='r')
    base_spec=manifest['base_vectors']
    base=np.memmap(base_spec['path'],mode='r',dtype=base_spec['dtype'],offset=base_spec['offset_bytes'],shape=tuple(base_spec['shape']))
    bundle=joblib.load(OUT/'model.joblib')
    expected=np.load(OUT/'test_p_retain.npy')
    with threadpool_limits(limits=4):
        for i in range(0,len(meta),8192):
            m=meta[i:i+8192]
            actual=predict_joint(q[m['query_row']],base[m['node_id']],state[i:i+len(m)],bundle)
            assert np.array_equal(actual,expected[i:i+len(m)])
    for source,indices in [('model2',slice(0,512)),('state_mlp',slice(512,529))]:
        other=np.load(E/source/'normalization.npz')
        for k in ('mean','scale'):
            np.testing.assert_array_equal(bundle[k][indices],other[k])
    provenance=json.loads((OUT/'provenance.json').read_text())
    for source in ('model1','model2','state_mlp'):
        old=json.loads((E/source/'provenance.json').read_text())
        assert old['data_manifest_sha256']==provenance['data_manifest_sha256']
        assert old['splits']==provenance['splits']
        assert old['parameters']==provenance['parameters']
    # Strict threshold convention: validation counts cannot exceed the requested budget.
    rows=[]
    for target,name in [(0.001,'validation_fnr_0.1pct'),(0.005,'validation_fnr_0.5pct'),(0.01,'validation_fnr_1pct')]:
        v=metrics['splits']['validation']['operating'][name]
        t=metrics['splits']['test']['operating'][name]
        assert v['retain_mislabeled_skipped']<=int(np.floor(target*v['retain_y1']))
        rows.append(dict(validation_target_retain_fnr=target,threshold=metrics['operating_points'][name]['threshold'],
            validation_actual_retain_fnr=v['retain_false_skip_rate'],validation_skip_rate=v['skip_rate'],
            test_auc=metrics['splits']['test']['ranking']['roc_auc_retain'],
            test_retain_fnr=t['retain_false_skip_rate'],test_skip_rate=t['skip_rate'],
            test_skipped=t['skipped'],test_safe_skipped=t['safe_skipped'],
            test_retain_false_skip_count=t['retain_mislabeled_skipped'],
            test_skipped_per_query=t['skipped_per_query_mean']))
    with (OUT/'risk_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT/'verification.json').write_text(json.dumps(dict(all_111841_raw_test_predictions_exactly_verified=True,
        state_and_interaction_normalization_match_prior_models=True,
        same_splits_and_training_parameters=True,validation_risk_budgets_verified=True),indent=2)+'\n')
    lines=['','## M3 requested risk summary','',
           f'Test AUC = {rows[0]["test_auc"]:.6f}；111841 个 expansion，1000 个 query。', '',
           '| 验证目标 RetainFNR | 验证实际 FNR | Test RetainFNR | Test SkipRate | Test 跳过数 | Test 误跳过数 | 平均跳过/query |',
           '|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["validation_target_retain_fnr"]*100:.1f}% | {r["validation_actual_retain_fnr"]*100:.4f}% | {r["test_retain_fnr"]*100:.4f}% | {r["test_skip_rate"]*100:.3f}% | {r["test_skipped"]} | {r["test_retain_false_skip_count"]} | {r["test_skipped_per_query"]:.3f} |')
    lines += ['', 'RetainFNR 分母为测试集 47641 个保守 retain 标签；SkipRate 分母为全部 111841 个 expansion。',
              '0.5% 基线对照也使用各自保存的 validation 概率定阈值，再评估 test，没有按 test 调阈值；结果保存在 comparison.json。',
              '推理入口 predict_model3.py 使用原始 q/v/state。全部测试预测与训练导出逐项一致，标准化与 M2/state-only 输入分块一致。',
              '分析重现：`.venv/bin/python experiments/sift1m/analyze_model3.py`。']
    report=OUT/'REPORT.md'
    prior=report.read_text().split('\n## M3 requested risk summary')[0]
    report.write_text(prior+'\n'.join(lines)+'\n')
    print(json.dumps(dict(fit=metrics['fit'],rows=rows),indent=2))

if __name__=='__main__':
    main()
