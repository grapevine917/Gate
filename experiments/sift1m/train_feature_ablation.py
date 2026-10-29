#!/usr/bin/env python3
"""Five nested state MLP ablations, with the established query split/protocol."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='4'
import copy,csv,hashlib,json,time
from pathlib import Path
import joblib,numpy as np
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from train_model1 import predict
from train_state_baselines import ranking,operating,threshold_from_validation,dump

E=Path(__file__).resolve().parent; DATA=E/'training_data_l100'; OUT=E/'feature_ablation'
ORDER=['candidate_queue_rank','completed_expansions','visited_count',
       'candidate_pq_over_best','unexpanded_count']
TARGETS=[0.001,0.0025,0.005,0.01,0.02]

def main():
    OUT.mkdir(exist_ok=True); (OUT/'COMPLETE.json').unlink(missing_ok=True)
    manifest=json.loads((DATA/'manifest.json').read_text()); names=manifest['pre_io_state_columns']
    splits={s:dict(raw=np.load(DATA/s/'pre_io_state.npy',mmap_mode='r'),
                  y=np.load(DATA/s/'y.npy'),meta=np.load(DATA/s/'metadata.npy'),
                  ids=np.load(DATA/s/'query_ids.npy')) for s in ('train','validation','test')}
    qs=[set(s['ids'].tolist()) for s in splits.values()]
    assert not(qs[0]&qs[1] or qs[0]&qs[2] or qs[1]&qs[2])
    for s in splits.values():
        assert np.array_equal(s['y'],s['meta']['label'])
    summary=[]
    for count in (1,2,3,5,17):
        label=f'Top{count}' if count<17 else 'All17'; folder=OUT/label;folder.mkdir(exist_ok=True)
        features=ORDER[:count] if count<17 else names
        indices=[names.index(f) for f in features]
        x={n:np.array(s['raw'][:,indices],dtype='float32',order='C') for n,s in splits.items()}
        mean=x['train'].mean(axis=0,dtype='float64').astype('float32')
        ss=np.zeros(count,dtype='float64')
        for i in range(0,len(x['train']),8192):
            d=x['train'][i:i+8192].astype('float64')-mean; ss+=(d*d).sum(axis=0)
        scale=np.sqrt(ss/len(x['train'])).astype('float32'); scale[scale==0]=1
        for n in x:x[n]-=mean;x[n]/=scale
        model=MLPClassifier(hidden_layer_sizes=(128,64),activation='relu',solver='adam',
            batch_size=2048,learning_rate_init=0.001,alpha=0.0001,shuffle=True,
            random_state=42,early_stopping=False)
        best_loss=float('inf');stale=0;history=[];started=time.perf_counter()
        print('Training',label,features,flush=True)
        with threadpool_limits(limits=4):
            for epoch in range(1,61):
                model.partial_fit(x['train'],splits['train']['y'],classes=np.array([0,1]))
                p=predict(model,x['validation']); val=ranking(splits['validation']['y'],p)
                history.append(dict(epoch=epoch,train_loss=float(model.loss_),validation=val))
                if val['log_loss']<best_loss-0.0001:
                    best_loss=val['log_loss'];best_epoch=epoch;best_model=copy.deepcopy(model);stale=0
                else:stale+=1
                dump(folder/'training_history.json',history)
                print(label,epoch,'val AUC',round(val['roc_auc_retain'],6),'best',best_epoch,'stale',stale,flush=True)
                if stale>=8:break
            model=best_model; elapsed=time.perf_counter()-started
            probs={n:predict(model,a) for n,a in x.items()}
        bundle=dict(model=model,mean=mean,scale=scale,feature_indices=indices,
                    feature_names=features,input_order=features)
        joblib.dump(bundle,folder/'model.joblib')
        np.savez(folder/'normalization.npz',mean=mean,scale=scale)
        # Select all thresholds before reporting test operating metrics.
        thresholds={f'fnr_{risk*100:g}pct':threshold_from_validation(
            splits['validation']['y'],probs['validation'],risk) for risk in TARGETS}
        dump(folder/'thresholds.json',thresholds)
        metrics=dict(model=label,architecture=[count,128,64,1],features=features,
            feature_indices=indices,parameter_count=sum(w.size for w in model.coefs_+model.intercepts_),
            fit=dict(best_epoch=best_epoch,epochs_run=len(history),training_seconds=elapsed,
                     max_epochs=60,patience=8,min_delta=0.0001),splits={})
        for n,s in splits.items():
            np.save(folder/f'{n}_p_retain.npy',probs[n])
            metrics['splits'][n]=dict(ranking=ranking(s['y'],probs[n]),operating={
                op:operating(s['y'],probs[n],c['threshold'],s['meta']['query_id'])
                for op,c in thresholds.items()})
        reload=joblib.load(folder/'model.joblib')
        np.testing.assert_array_equal(predict(reload['model'],x['test'][:8192]),probs['test'][:8192])
        # Mapped export uses raw live state; no q/v vectors or node IDs in model input.
        with (folder/'model.bin').open('wb') as f:
            f.write(b'ABGATE01');np.array([count],dtype='<u4').tofile(f)
            np.array(indices,dtype='<u4').tofile(f)
            for a in (mean,scale):a.astype('<f4').tofile(f)
            for w,b in zip(model.coefs_,model.intercepts_):
                np.ascontiguousarray(w,dtype='<f4').tofile(f);b.astype('<f4').tofile(f)
        dump(folder/'metrics.json',metrics)
        row=dict(model=label,n_features=count,features=';'.join(features),
                 test_auc=metrics['splits']['test']['ranking']['roc_auc_retain'],best_epoch=best_epoch)
        for risk in (.001,.01):
            op=f'fnr_{risk*100:g}pct';m=metrics['splits']['test']['operating'][op]
            suffix=f'{risk*100:g}pct'
            row[f'tau_{suffix}']=thresholds[op]['threshold']
            row[f'test_skip_rate_{suffix}']=m['skip_rate']
            row[f'test_retain_fnr_{suffix}']=m['retain_false_skip_rate']
        summary.append(row);dump(OUT/'offline_summary.json',summary)
        print('COMPLETE',label,row,flush=True)
        del x
    with (OUT/'offline_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    dump(OUT/'provenance.json',dict(
        architecture='d-128-64-1 ReLU/ReLU/sigmoid; BCE Adam',seed=42,batch_size=2048,
        learning_rate=.001,alpha=.0001,standardization='train only',
        split=manifest['splits'],manifest_sha256=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest(),
        feature_order=ORDER,selection='Top5 is mean absolute SHAP order from prior state XGBoost analysis',
        limitation='Feature ranking was observed on these test queries; ablation is exploratory, not a fresh feature-selection holdout.',
        threshold_selection='validation only; strict p_retain < tau',
        label_definition=manifest['label_definition'],
        width_changes_parameter_count=True))
    lines=['# State feature ablation','',
        '五组重新训练 d→128→64→1 MLP；沿用8000/1000/1000互斥 query、train-only标准化、Adam、seed=42、batch=2048、lr=.001、alpha=.0001。验证logloss早停（最多60 epochs，patience=8，min_delta=.0001）。',
        'Top5按上一实验 mean |SHAP| 排名；All17保持原始列顺序。阈值只在validation选择，测试集报告实际FNR。',
        '特征排名曾在这一测试集上观察，因此本次属于探索性消融；训练和阈值选择均不使用test，但最终确认应使用新query。',
        '', '| Model | Test AUC | SkipRate@val FNR .1% | actual Test FNR | SkipRate@val FNR 1% | actual Test FNR |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["model"]} | {r["test_auc"]:.6f} | {r["test_skip_rate_0.1pct"]*100:.3f}% | {r["test_retain_fnr_0.1pct"]*100:.4f}% | {r["test_skip_rate_1pct"]*100:.3f}% | {r["test_retain_fnr_1pct"]*100:.4f}% |')
    lines+=['','## Features','']+[f'- {r["model"]}: {r["features"]}' for r in summary]
    lines+=['','FNR对应保守retain标签误跳过率，不是在线Recall损失保证；SkipRate为样本加权比例。输入维度改变也改变首层参数量，隐藏层与其他训练设置固定。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    dump(OUT/'COMPLETE.json',dict(complete=True,models=5,offline_only=True))

if __name__=='__main__':main()
