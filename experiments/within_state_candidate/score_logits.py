from common import *
import joblib
from scipy.special import expit
r=dict(np.load(R/'records.npz'));shape=(len(r['state']),3,3);raw=np.empty(shape);canonical=np.empty(shape);errors={}
dump(R/'AUXILIARY_SCORE_PROTOCOL.json',dict(purpose='avoid concluding lack of ranking information solely from float32 sigmoid saturation; primary scores remain actual native probabilities',score='pre-sigmoid MLP logit, recomputed from frozen float32 coefficients and same normalization; no fitting, labels or new threshold selection',primary='native probability',auxiliary='unsaturated logit',canonical='same global snapshot control; not a deployed search'))
for gi,g in enumerate(GROUPS):
 for si,s in enumerate(SEEDS):
  bundle=joblib.load(FG/'models'/f'{g}_s{s}/model.joblib');model=bundle['model']
  for name,states,out in [('natural',r['state'],raw),('canonical',r['canonical_state'],canonical)]:
   values=[]
   for start in range(0,len(states),8192):
    z=np.array(states[start:start+8192,bundle['feature_indices']],np.float32,order='C');z-=bundle['mean'];z/=bundle['scale']
    for j,(w,b) in enumerate(zip(model.coefs_,model.intercepts_)):
     z=z@w+b
     if j<2:z=np.maximum(z,0)
    values.append(z[:,0].copy())
   out[:,gi,si]=np.concatenate(values)
  miss=~r['cached'];err=float(np.max(np.abs(expit(raw[miss,gi,si])-r['native_scores'][miss,gi,si])));assert err<3e-6;errors[f'{g}_s{s}']=err
# A mathematically identical Global input must not acquire a ranking from BLAS tile rounding.
_,first,inverse=np.unique(np.column_stack([r['query_row'],r['beam_step']]),axis=0,return_index=True,return_inverse=True)
oldp=r['canonical_scores'][:,0,:].copy();oldz=canonical[:,0,:].copy()
r['canonical_scores'][:,0,:]=r['canonical_scores'][first[inverse],0,:]
canonical[:,0,:]=canonical[first[inverse],0,:]
dump(R/'canonical_numerical_control.json',dict(method='evaluate one representative per identical beam Global snapshot and broadcast to its candidates, without consulting labels',max_probability_tile_rounding_difference=float(np.max(np.abs(oldp-r['canonical_scores'][:,0,:]))),max_logit_tile_rounding_difference=float(np.max(np.abs(oldz-canonical[:,0,:])))))
np.savez_compressed(R/'records.npz',**r)
np.savez_compressed(R/'logit_scores.npz',natural=raw,canonical=canonical);dump(R/'logit_verification.json',errors);log('AUXILIARY PRE-SIGMOID SCORES VERIFIED')
