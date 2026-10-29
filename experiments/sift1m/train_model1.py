#!/usr/bin/env python3
"""Query/candidate-only MLP, with query-disjoint validation early stopping."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
import copy
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from train_state_baselines import dump, ranking, operating, threshold_from_validation, per_query_csv, TARGETS

E = Path(__file__).resolve().parent
DATA = E / 'training_data_l100'
OUT = E / 'model1'

def predict(model, x):
    return np.concatenate([model.predict_proba(x[i:i+8192])[:, 1]
                           for i in range(0, len(x), 8192)]).astype('float64')

def main(input_view='qv'):
    if input_view not in ('qv', 'state', 'interaction', 'joint'):
        raise ValueError('Unknown input_view')
    state_only = input_view == 'state'
    joint = input_view == 'joint'
    interaction = input_view in ('interaction', 'joint')
    targets = TARGETS + [('validation_fnr_0.5pct',0.005)] if joint else TARGETS
    output_name = {'qv':'model1', 'state':'state_mlp', 'interaction':'model2', 'joint':'model3'}[input_view]
    display_name = {'qv':'M1 q-v MLP', 'state':'state-only MLP', 'interaction':'M2 interaction MLP', 'joint':'M3 joint MLP'}[input_view]
    input_order = ('q[128];candidate_v[128];abs(q-v)[128];q*v[128]' if interaction
                   else 'q[128];candidate_v[128]')
    if joint:
        input_order += ';pre_io_state[17]'
    OUT = E / output_name
    width = 17 if state_only else 529 if joint else 512 if interaction else 256
    OUT.mkdir(exist_ok=True)
    (OUT / 'COMPLETE.json').unlink(missing_ok=True)
    assert json.loads((DATA / 'COMPLETE.json').read_text())['complete']
    manifest = json.loads((DATA / 'manifest.json').read_text())
    if not state_only:
        b = manifest['base_vectors']
        base = np.memmap(b['path'], mode='r', dtype=b['dtype'], offset=b['offset_bytes'], shape=tuple(b['shape']))
    splits = {}
    for name in ('train', 'validation', 'test'):
        folder = DATA / name
        meta = np.load(folder / 'metadata.npy', mmap_mode='r')
        y = np.load(folder / 'y.npy', mmap_mode='r')
        if state_only:
            x = np.array(np.load(folder/'pre_io_state.npy', mmap_mode='r'), dtype='float32')
        else:
            q = np.load(folder / 'query_vectors.npy', mmap_mode='r')
            x = np.empty((len(y), width), dtype='float32')
            for i in range(0, len(y), 8192):
                m = meta[i:i+8192]
                x[i:i+len(m), :128] = q[m['query_row']]
                x[i:i+len(m), 128:256] = base[m['node_id']]
                if interaction:
                    # Derive interactions from raw vectors before standardization.
                    a = x[i:i+len(m), :128]
                    v = base[m['node_id']]
                    x[i:i+len(m), 128:256] = v
                    x[i:i+len(m), 256:384] = np.abs(a-v)
                    x[i:i+len(m), 384:512] = a*v
            if joint:
                x[:,512:] = np.load(folder/'pre_io_state.npy', mmap_mode='r')
        assert x.shape == (len(y), width)
        assert np.isfinite(x).all() and np.array_equal(y, meta['label'])
        splits[name] = dict(x=x, y=y, meta=meta)
        print('Loaded', name, x.shape, flush=True)
    qs = [set(s['meta']['query_id'].tolist()) for s in splits.values()]
    assert not (qs[0]&qs[1] or qs[0]&qs[2] or qs[1]&qs[2])
    x = splits['train']['x']
    mean = x.mean(axis=0, dtype='float64').astype('float32')
    # Chunked moments avoid allocating a second full training matrix.
    ss = np.zeros(width, dtype='float64')
    for i in range(0, len(x), 8192):
        delta = x[i:i+8192].astype('float64') - mean
        ss += (delta * delta).sum(axis=0)
    scale = np.sqrt(ss / len(x)).astype('float32')
    scale[scale == 0] = 1
    if state_only:
        saved_stats = np.load(DATA/'pre_io_state_train_normalization.npz')
        np.testing.assert_allclose(mean, saved_stats['mean'], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(scale, saved_stats['scale'], rtol=1e-6, atol=1e-6)
    np.savez(OUT / 'normalization.npz', mean=mean, scale=scale)
    for s in splits.values():
        s['x'] -= mean
        s['x'] /= scale
    model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                          batch_size=2048, learning_rate_init=0.001, alpha=0.0001,
                          shuffle=True, random_state=42, early_stopping=False)
    history = []
    best_loss, best_epoch, stale = float('inf'), 0, 0
    started = time.perf_counter()
    with threadpool_limits(limits=4):
        for epoch in range(1, 61):
            t = time.perf_counter()
            model.partial_fit(x, splits['train']['y'], classes=np.array([0, 1]))
            vp = predict(model, splits['validation']['x'])
            vr = ranking(splits['validation']['y'], vp)
            history.append(dict(epoch=epoch, train_loss=float(model.loss_), validation=vr,
                                seconds=time.perf_counter()-t))
            if vr['log_loss'] < best_loss - 0.0001:
                best_loss, best_epoch, stale = vr['log_loss'], epoch, 0
                best_model = copy.deepcopy(model)
                joblib.dump(dict(model=best_model, mean=mean, scale=scale,
                                 input_order=manifest['pre_io_state_columns'] if state_only else input_order), OUT / 'model.joblib')
            else:
                stale += 1
            dump(OUT / 'training_history.json', history)
            print(f'Epoch {epoch}: loss={model.loss_:.6f}, val loss={vr["log_loss"]:.6f}, '
                  f'val AUC={vr["roc_auc_retain"]:.6f}, best={best_epoch}, stale={stale}', flush=True)
            if stale >= 8:
                break
        elapsed = time.perf_counter()-started
        model = best_model
        assert [a.shape for a in model.coefs_] == [(width,128), (128,64), (64,1)]
        assert model.out_activation_ == 'logistic'
        probabilities = {n: predict(model, s['x']) for n,s in splits.items()}
        chosen = {'default_0.5': {'threshold':0.5, 'selection_split':'fixed'}}
        for name, risk in targets:
            chosen[name] = threshold_from_validation(splits['validation']['y'], probabilities['validation'], risk)
            chosen[name]['decision'] = f'skip iff p(retain|{input_view}) < threshold'
        dump(OUT / 'thresholds.json', chosen)
        metrics = dict(model=display_name, architecture=[width,128,64,1],
                       parameter_count=int(sum(a.size for a in model.coefs_+model.intercepts_)),
                       fit=dict(training_seconds=elapsed, best_epoch=best_epoch, epochs_run=len(history),
                                max_epochs=60, patience=8, min_delta=0.0001,
                                selection='validation logloss only'),
                       operating_points=chosen, splits={})
        for name,s in splits.items():
            p = probabilities[name]
            assert np.isfinite(p).all()
            np.save(OUT / f'{name}_p_retain.npy', p)
            metrics['splits'][name] = dict(ranking=ranking(s['y'], p), operating={})
            for op, c in chosen.items():
                metrics['splits'][name]['operating'][op] = operating(s['y'], p, c['threshold'], s['meta']['query_id'])
                if name != 'train':
                    per_query_csv(OUT / f'{name}_per_query_{op}.csv', s['meta'], s['y'], p, c['threshold'])
        bundle = joblib.load(OUT / 'model.joblib')
        actual = predict(bundle['model'], splits['test']['x'][:8192])
        assert np.array_equal(actual, probabilities['test'][:8192])
        metrics['saved_model_roundtrip_verified'] = True
    versions = {n:importlib.metadata.version(n) for n in ('numpy','scipy','scikit-learn','joblib','threadpoolctl','xgboost-cpu')}
    dump(OUT / 'provenance.json', dict(seed=42, input=manifest['pre_io_state_columns'] if state_only else input_order + '; train-only per-coordinate standardization',
         forbidden_inputs='IDs, neighbor observations, post-score state and oracle metadata' if joint else 'q/v vectors, IDs, neighbor observations, post-score state and oracle metadata' if state_only else 'state, distances, IDs, neighbor observations and oracle metadata',
         sample_unit='candidate node expansion', label_definition=manifest['label_definition'],
         splits=manifest['splits'], query_disjoint=True, test_used_for_training_or_selection=False,
         data_manifest_sha256=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest(),
         parameters=model.get_params(), versions=versions))
    (OUT / 'requirements.lock.txt').write_text(''.join(f'{n}=={v}\n' for n,v in versions.items()))
    dump(OUT / 'metrics.json', metrics)
    baselines = json.loads((E/'state_baselines/results.json').read_text())
    models = {display_name:metrics, **baselines}
    if state_only or interaction:
        models['q-v MLP'] = json.loads((E/'model1/metrics.json').read_text())
    if interaction:
        models['state-only MLP'] = json.loads((E/'state_mlp/metrics.json').read_text())
    if joint:
        models['M2 interaction MLP'] = json.loads((E/'model2/metrics.json').read_text())
        # Recalibrate comparison thresholds exclusively from saved validation scores.
        for m in models.values():
            if 'validation_fnr_0.5pct' not in m['splits']['test']['operating']:
                folder = {'model1_qv_mlp':'model1', 'M1 q-v MLP':'model1',
                          'state_only_mlp':'state_mlp', 'state-only MLP':'state_mlp',
                          'M2 interaction MLP':'model2', 'logistic_regression':'state_baselines/logistic_regression',
                          'xgboost':'state_baselines/xgboost'}[m['model']]
                vp = np.load(E/folder/'validation_p_retain.npy')
                tp = np.load(E/folder/'test_p_retain.npy')
                threshold = threshold_from_validation(splits['validation']['y'], vp, 0.005)
                m['splits']['test']['operating']['validation_fnr_0.5pct'] = operating(
                    splits['test']['y'], tp, threshold['threshold'], splits['test']['meta']['query_id'])
        dump(OUT/'comparison.json', {n:m['splits']['test'] for n,m in models.items()})
    lines = ['# ' + display_name, '',
             ('输入仅为与 LR/XGBoost 相同的 17 维 pre_io_state。17→128→64→1。' if state_only
              else '输入为 [q;v;|q-v|;q⊙v;pre_io_state]，529→128→64→1。' if joint
              else '输入为原始向量构造的 [q;v;|q-v|;q⊙v]，512→128→64→1。' if interaction
              else '输入仅为 x=[q;v]，v 为待展开候选节点。256→128→64→1。') + 'ReLU/ReLU/sigmoid，BCE + Adam，batch=2048，lr=0.001。',
             '沿用 L=100、seed=42、8000/1000/1000 个互斥 query。只用训练集拟合标准化；验证集 logloss 早停选模型，验证集选阈值；测试集仅最终评估。', '',
             f'最佳 epoch={best_epoch}，运行 {len(history)} epochs，参数量={metrics["parameter_count"]}。', '',
             '| 模型 | Train AUC | Validation AUC | Test AUC | Test logloss |',
             '|---|---:|---:|---:|---:|']
    for n,m in models.items():
        r=m['splits']; lines.append(f'| {n} | {r["train"]["ranking"]["roc_auc_retain"]:.6f} | {r["validation"]["ranking"]["roc_auc_retain"]:.6f} | {r["test"]["ranking"]["roc_auc_retain"]:.6f} | {r["test"]["ranking"]["log_loss"]:.6f} |')
    lines += ['', '| 模型 | 验证目标 FNR | Test skip rate | Test retain FNR | 误跳过数 | 受影响 query |', '|---|---|---:|---:|---:|---:|']
    for n,m in models.items():
        for op in chosen:
            a=m['splits']['test']['operating'][op]
            lines.append(f'| {n} | {op} | {a["skip_rate"]*100:.3f}% | {a["retain_false_skip_rate"]*100:.4f}% | {a["retain_mislabeled_skipped"]} | {a["queries_with_retain_false_skip"]}/1000 |')
    a = metrics['splits']['test']['operating']
    lines += ['', f'{input_view} MLP：test AUC={metrics["splits"]["test"]["ranking"]["roc_auc_retain"]:.4f}。验证目标 FNR=0.1%/1% 时，测试跳过比例为 {a["validation_fnr_0.1pct"]["skip_rate"]*100:.3f}%/{a["validation_fnr_1pct"]["skip_rate"]*100:.3f}%。',
              '这里的 retain FNR 是保守保留标签误跳过率，不是 Recall@10 损失。验证零错误不保证测试零错误。',
              '各 MLP 的隐藏层、优化器、batch、seed、早停和数据划分一致，输入维度/特征和对应训练标准化统计不同。state MLP 与 LR/XGBoost 使用相同原始状态特征。',
              'M2 的交互特征是 q/v 的确定性函数，没有添加新的外部信息；但输入扩展也增加首层参数，M1/M2 不是参数量严格相等的对照。',
              '单一结构的失败不能证明 q-v 在信息论意义上不充分，也可能受模型容量、优化或样本量限制。',
              'skip rate 是 baseline 轨迹上的离线比例，需在线搜索验证召回、I/O 和延迟。', '',
              '重现：在 learned-gate 目录执行 `.venv/bin/python experiments/sift1m/' + ('train_state_mlp.py' if state_only else 'train_model3.py' if joint else 'train_model2.py' if interaction else 'train_model1.py') + '`.']
    (OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    dump(OUT / 'COMPLETE.json', dict(complete=True, query_disjoint=True, qv_only=not state_only and not joint, state_only=state_only, joint=joint,
                                  saved_model_verified=True, validation_selected_thresholds=True))
    print(input_view.upper() + ' MLP COMPLETE', json.dumps(metrics['splits']['test']), flush=True)

if __name__ == '__main__':
    main()
