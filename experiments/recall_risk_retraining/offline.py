from common import *
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
assert (D/'SELECTION_LOCK.json').exists()
metrics=[];predictions={}
for split in ['validation','test']:
    a=labels(split);x=np.ascontiguousarray(a['state']);y=(a['keep_hits']>a['skip_hits']).astype(int);w=a['weight'];check=D/f'{split}_states.bin'
    with check.open('wb') as f:f.write(struct.pack('<I',len(x)));x.tofile(f)
    for variant in ['old','proxy','risk']:
        models=A/'models_train100' if variant=='old' else D/'models'/variant;output=D/f'{split}_{variant}_scores.bin'
        subprocess.run([str(A/'native/score'),str(models),str(check),str(output)],check=True)
        with output.open('rb') as f:n=np.fromfile(f,'<u4',1)[0];scores=np.fromfile(f,'<f4').reshape(n,2)
        assert len(scores)==len(a) and np.isfinite(scores).all()
        for j,family in enumerate(['LR','MLP']):
            s=scores[:,j];key=(variant,family);predictions[(split,*key)]=s
            row=dict(split=split,variant=variant,family=family,samples=len(a),weighted_risk_fraction=float(np.average(y,weights=w)),risk_ROC_AUC=float(roc_auc_score(y,s,sample_weight=w)),risk_AP=float(average_precision_score(y,s,sample_weight=w)),risk_Brier=float(brier_score_loss(y,s,sample_weight=w)) if variant=='risk' else None)
            if split=='validation':
                ix=np.argsort(s[y==1]);ss=s[y==1][ix];ww=w[y==1][ix];allowed=.01*ww.sum();cum=np.cumsum(ww);idx=min(int(np.searchsorted(cum,allowed,side='right')),len(ss)-1);row['tau_at_validation_risk_FNR1pct']=float(ss[idx])
            else:
                val=next(r for r in metrics if r['split']=='validation' and r['variant']==variant and r['family']==family);tau=val['tau_at_validation_risk_FNR1pct'];skip=s<tau
                row.update(tau_at_validation_risk_FNR1pct=tau,test_risk_FNR=float(np.sum(w[(y==1)&skip])/np.sum(w[y==1])),test_safe_skip_fraction=float(np.sum(w[(y==0)&skip])/np.sum(w[y==0])),test_skip_fraction=float(np.average(skip,weights=w)))
            metrics.append(row)
    # Separate teacher contribution and proxy/causal disagreements by context/stratum.
    summaries=[]
    for policy in range(3):
        for st in range(3):
            b=a[(a['policy']==policy)&(a['stratum']==st)];risk=b['keep_hits']>b['skip_hits'];ww=b['weight'];summaries.append(dict(policy=policy,stratum=st,samples=len(b),population_weight=float(ww.sum()),risk_fraction=float(np.average(risk,weights=ww)),proxy_positive_fraction=float(np.average(b['proxy'],weights=ww)),proxy_positive_but_safe_fraction=float(np.average((b['proxy']==1)&~risk,weights=ww)),proxy_negative_but_harmful_fraction=float(np.average((b['proxy']==0)&risk,weights=ww)),mean_signed_read_saving=float(np.average(b['keep_reads']-b['skip_reads'],weights=ww))))
    dump(D/f'{split}_label_diagnostics.json',summaries)
dump(D/'offline_metrics.json',metrics)
# All rows get the same CSV schema, with blanks for inapplicable fields.
keys=list(dict.fromkeys(k for r in metrics for k in r));save(D/'offline_metrics.csv',[{k:r.get(k) for k in keys} for r in metrics]);dump(D/'OFFLINE_COMPLETE.json',dict(complete=True,test_metrics=[r for r in metrics if r['split']=='test']));print('OFFLINE COMPLETE',flush=True)
