from common import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Additional nested feature controls, frozen before held-out performance is computed.
a=load();x=a['x'];y=a['U'].astype(int);mask=a['partition']==0
sets={'current_state_only':[0]+list(range(1,18)), 'current_state_history_query':[0]+list(range(1,18))+list(range(18,25))+[25,26,27]}
dump(R/'ABLATION_PROTOCOL.json',dict(sets={k:[FEATURES[i] for i in v] for k,v in sets.items()},fit='same 500 fit queries only; C=1; train-only scaler',purpose='history adds information beyond current state; Query adds information beyond state and history; predictive association, not causality'))
models={}
for name,c in sets.items():
 scaler=StandardScaler().fit(x[mask][:,c]);lr=LogisticRegression(C=1,max_iter=3000,random_state=SEED).fit(scaler.transform(x[mask][:,c]),y[mask]);w=lr.coef_[0]/scaler.scale_;coef=np.zeros(x.shape[1]);coef[c]=w;models[name]=dict(weights=coef.tolist(),intercept=float(lr.intercept_[0]-w@scaler.mean_))
dump(R/'ablations.json',models)
log('ADDITIONAL NESTED FEATURE CONTROLS FIT ON FIT QUERIES ONLY')
