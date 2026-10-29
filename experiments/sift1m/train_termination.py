#!/usr/bin/env python3
"""Top3 and All17 P(safe-to-stop) MLPs on real beam-boundary labels."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
import copy,csv,json,time
from pathlib import Path
import joblib,numpy as np
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from train_model1 import predict
from train_state_baselines import ranking,dump
E=Path(__file__).resolve().parent;OUT=E/'learned_termination'
TOP3=['candidate_queue_rank','completed_expansions','visited_count']

def main():
    assert json.loads((OUT/'DATA_COMPLETE.json').read_text())['complete']
    (OUT/'TRAINING_COMPLETE.json').unlink(missing_ok=True)
    manifest=json.loads((OUT/'data_manifest.json').read_text());names=manifest['state_columns']
    splits={n:dict(raw=np.load(OUT/'data'/n/'state.npy'),
                  y=np.load(OUT/'data'/n/'y_safe_stop.npy'),
                  qids=np.load(OUT/'data'/n/'query_ids.npy')) for n in ('train','validation','test')}
    q=[set(s['qids'].tolist()) for s in splits.values()]
    assert not(q[0]&q[1] or q[0]&q[2] or q[1]&q[2])
    summary=[]
    for label,features in [('Top3',TOP3),('All17',names)]:
        indices=[names.index(f) for f in features];d=len(indices)
        folder=OUT/label;folder.mkdir(exist_ok=True)
        x={n:np.array(s['raw'][:,indices],dtype='float32',order='C') for n,s in splits.items()}
        mean=x['train'].mean(axis=0,dtype='float64').astype('float32')
        scale=np.sqrt(np.mean((x['train'].astype('float64')-mean)**2,axis=0)).astype('float32');scale[scale==0]=1
        for a in x.values():a-=mean;a/=scale
        model=MLPClassifier(hidden_layer_sizes=(128,64),activation='relu',solver='adam',
            batch_size=2048,learning_rate_init=.001,alpha=.0001,shuffle=True,random_state=42,early_stopping=False)
        best_loss=float('inf');stale=0;history=[];started=time.perf_counter()
        print('Training ET',label,'P(safe-to-stop)',flush=True)
        with threadpool_limits(limits=4):
            for epoch in range(1,61):
                model.partial_fit(x['train'],splits['train']['y'],classes=np.array([0,1]))
                val=ranking(splits['validation']['y'],predict(model,x['validation']))
                history.append(dict(epoch=epoch,train_loss=float(model.loss_),validation=val))
                if val['log_loss']<best_loss-.0001:
                    best_loss=val['log_loss'];best_epoch=epoch;best_model=copy.deepcopy(model);stale=0
                else:stale+=1
                dump(folder/'training_history.json',history)
                print(label,epoch,'val safe AUC',round(val['roc_auc_retain'],6),'best',best_epoch,'stale',stale,flush=True)
                if stale>=8:break
            model=best_model;probs={n:predict(model,a) for n,a in x.items()}
        bundle=dict(model=model,mean=mean,scale=scale,feature_indices=indices,feature_names=features,
                    output='P(safe-to-stop)',label_definition=manifest['safe_to_stop_label'])
        joblib.dump(bundle,folder/'model.joblib');np.savez(folder/'normalization.npz',mean=mean,scale=scale)
        with (folder/'model.bin').open('wb') as f:
            f.write(b'ABGATE01');np.array([d],dtype='<u4').tofile(f);np.array(indices,dtype='<u4').tofile(f)
            for a in (mean,scale):a.astype('<f4').tofile(f)
            for w,b in zip(model.coefs_,model.intercepts_):
                np.ascontiguousarray(w,dtype='<f4').tofile(f);b.astype('<f4').tofile(f)
        metrics=dict(model='ET '+label,features=features,architecture=[d,128,64,1],
            output='P(safe-to-stop)',parameter_count=sum(a.size for a in model.coefs_+model.intercepts_),
            fit=dict(best_epoch=best_epoch,epochs_run=len(history),training_seconds=time.perf_counter()-started,
                     max_epochs=60,patience=8,min_delta=.0001),splits={})
        for n,s in splits.items():
            np.save(folder/f'{n}_p_safe_stop.npy',probs[n]);r=ranking(s['y'],probs[n])
            # ranking helper's ROC key says retain, but this model's positive label is safe-to-stop.
            metrics['splits'][n]=dict(auc_safe_stop=r['roc_auc_retain'],log_loss=r['log_loss'],
                ap_safe_stop=r['average_precision_retain'],brier=r['brier'],samples=len(s['y']),safe_fraction=float(s['y'].mean()))
        reload=joblib.load(folder/'model.joblib')
        np.testing.assert_array_equal(predict(reload['model'],x['test']),probs['test'])
        dump(folder/'metrics.json',metrics)
        summary.append(dict(model=label,test_auc=metrics['splits']['test']['auc_safe_stop'],
            validation_auc=metrics['splits']['validation']['auc_safe_stop'],best_epoch=best_epoch))
        dump(OUT/'training_summary.json',summary)
        print('ET MODEL COMPLETE',label,summary[-1],flush=True)
    dump(OUT/'training_provenance.json',dict(seed=42,architecture='d-128-64-1, ReLU/ReLU/sigmoid',
        optimizer='Adam BCE',batch_size=2048,lr=.001,alpha=.0001,normalization='train-only',
        query_disjoint=True,validation_selection='logloss early stopping',test_used_for_fit_or_selection=False,
        output='safe=1; stop iff p_safe >= tau',
        features='same named Top3 / All17 fields as expansion pruning; new beam-boundary dataset and new target',
        no_expansion_pruning_label_reuse=True))
    dump(OUT/'TRAINING_COMPLETE.json',dict(complete=True,models=2,safe_stop_positive_label=True))

if __name__=='__main__':main()
