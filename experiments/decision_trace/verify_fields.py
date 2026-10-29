"""Independent Python MLP forward and native wire/behavior acceptance checks."""
from pathlib import Path
import numpy as np,json,struct,os
for key in ['MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
from accept import O,R,D,run,identical,CONTROL
from convert import convert

def main():
 checks={};baseline=run('extra_control',CONTROL);old=run('extra_original_binary',O/'reference/search_disk_index_original');identical(baseline,old);checks['original_binary_exact']=True
 with open(R/'experiments/sift1m/online_state_mlp/model.bin','rb') as f:
  assert f.read(8)==b'STGATE01'
  read=lambda n:np.fromfile(f,'<f4',n)
  mean=read(17);scale=read(17);weights=[];bias=[]
  for x,y in [(17,128),(128,64),(64,1)]:weights.append(read(x*y).reshape(x,y));bias.append(read(y))
  assert not f.read(1)
 for split in ['history','test']:
  a=np.load(O/(split+'_npz/L100.npz'));rng=np.random.default_rng(20261019);idx=rng.choice(len(a['query_id']),min(2000,len(a['query_id'])),replace=False)
  x=(a['state_features'][idx]-mean)/scale
  for j,(w,b) in enumerate(zip(weights,bias)):
   x=x@w+b
   if j<2:x=np.maximum(x,0)
  prediction=1/(1+np.exp(-x[:,0]));error=float(np.max(np.abs(prediction-a['model_score'][idx])));assert error<2e-5
  ids=np.load(D/(split+'_query_ids.npy')); cached=a['cache_hit'].astype(bool)
  checks[split]=dict(records=len(a['query_id']),cache_hit_decisions=int(cached.sum()),cache_hit_skips=int(a['model_decision'][cached].sum()),python_score_max_error=error,ids_mapped_exact=bool(set(a['query_id'])==set(ids)))
 # Multi-L serialization, default row IDs, explicit overwrite and malformed query-ID guards.
 args=['--experiment_mode','decision_trace','--decision_trace_prefix',str(O/'multi'),'--search_list','75','100']
 run('extra_multi_L',extra=args);convert(O/'multi.manifest.json',O/'multi_npz')
 assert [x['L'] for x in json.loads((O/'multi.manifest.json').read_text())['records']]==[75,100]
 assert np.array_equal(np.load(O/'multi_npz/queries.npz')['query_ids'],np.arange(1000))
 run('extra_overwrite',extra=args+['--decision_trace_overwrite','true']);convert(O/'multi.manifest.json',O/'multi_npz')
 (O/'bad_ids.txt').write_text('1\n1\n')
 run('extra_duplicate_ids',extra=['--experiment_mode','decision_trace','--decision_trace_query_ids',str(O/'bad_ids.txt')],expect=1)
 (O/'bad_count.txt').write_text('1\n')
 run('extra_bad_id_count',extra=['--experiment_mode','decision_trace','--decision_trace_query_ids',str(O/'bad_count.txt')],expect=1)
 checks.update(multi_L=True,default_row_ids=True,overwrite_guard=True,explicit_overwrite=True,duplicate_ids_rejected=True,bad_count_rejected=True)
 (O/'field_verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2),flush=True)
if __name__=='__main__':main()
