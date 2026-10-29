from pathlib import Path
import os,json,subprocess,numpy as np
ROOT=Path(__file__).resolve().parent;LG=ROOT.parent.parent
ns={'__file__':str(ROOT/'train.py')};exec((ROOT/'train.py').read_text().split('summary={}')[0],ns)
checks={}
for ds in ['sift','deep']:
 q=ns['matrix'](ROOT/ds/'test/query.fbin')[:8]
 with open(ROOT/ds/'test/gt.bin','rb') as f:n,d=np.fromfile(f,np.uint32,2);gt=np.fromfile(f,np.uint32,count=int(n)*int(d)).reshape(int(n),int(d))[:8]
 folder=ROOT/ds/'audit';folder.mkdir(exist_ok=True)
 for name,a in [('query',q),('gt',gt)]:
  with open(folder/f'{name}.bin','wb') as f:np.array(a.shape,np.uint32).tofile(f);a.tofile(f)
 for model in ['state_lr','state_tree','state_query_lr']:
  out=folder/model;env=os.environ.copy()
  for k in list(env):
   if k.startswith('DISKANN_'):env.pop(k)
  env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='1',DISKANN_M3_MODEL=str(ROOT/ds/model/'model.bin'),DISKANN_M3_TAU='0',DISKANN_M3_AUDIT=str(out)+'.audit')
  args=[str(LG/'DiskANN/build/apps/search_disk_index'),'--data_type','float','--dist_fn','l2','--index_path_prefix',str(LG/'experiments'/('sift1m' if ds=='sift' else 'deep1m')/'index'),'--query_file',str(folder/'query.bin'),'--gt_file',str(folder/'gt.bin'),'-K','10','-L','100','--result_path',str(out),'-W','4','-T','4','--num_nodes_to_cache','10000']
  with open(str(out)+'.log','w') as f:subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  audit=np.fromfile(str(out)+'.audit',np.float32).reshape(-1,19)
  states=audit[:,1:18];inputs=np.concatenate([states,np.repeat(q[:1],len(states),axis=0)],axis=1) if model=='state_query_lr' else states
  expected=ns['native_scores'](ROOT/ds/model/'model.bin',inputs);expected[states[:,10]!=0]=1
  error=float(np.max(np.abs(expected-audit[:,18])));assert error<1e-5,(ds,model,error)
  checks[ds+'/'+model]={'audited_candidates':len(states),'max_retain_score_error':error}
 # Old baseline request parity, including query row identity.
 old=LG/'experiments'/('sift1m/feature_ablation/online/r0_baseline.csv' if ds=='sift' else 'deep1m/online/r0_baseline_L100.csv')
 import csv
 original=list(csv.DictReader(open(old)));current=list(csv.DictReader(open(ROOT/ds/'online/test/r0_baseline.csv')))
 assert all(a['ssd_reads']==b['ssd_reads'] and a['expansions']==b['expansions'] for a,b in zip(original,current))
 checks[ds+'/baseline_request_parity']=True
# Exercise old ABGATE01 format after extending the reader.
env=os.environ.copy()
for k in list(env):
 if k.startswith('DISKANN_'):env.pop(k)
out=ROOT/'sift/audit/legacy_top3'
env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='1',DISKANN_M3_MODEL=str(LG/'experiments/sift1m/feature_ablation/Top3/model.bin'),DISKANN_M3_TAU='0.03665834292769432')
args=[str(LG/'DiskANN/build/apps/search_disk_index'),'--data_type','float','--dist_fn','l2','--index_path_prefix',str(LG/'experiments/sift1m/index'),'--query_file',str(ROOT/'sift/test/query.fbin'),'--gt_file',str(ROOT/'sift/test/gt.bin'),'-K','10','-L','100','--result_path',str(out),'-W','4','-T','4','--num_nodes_to_cache','10000']
with open(str(out)+'.log','w') as f:subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
for suffix in ['_100_idx_uint32.bin','_100_dists_float.bin']:
 old=LG/'experiments/sift1m/feature_ablation/online'/('r0_Top3_fnr_0.5pct'+suffix)
 assert old.read_bytes()==Path(str(out)+suffix).read_bytes()
checks['legacy_top3_results_exact_parity']=True
(ROOT/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
