"""Freeze label thresholds on validation; measure causal online policy on test."""
from pathlib import Path
import csv,json,os,subprocess,numpy as np
ROOT=Path(__file__).resolve().parent;LG=ROOT.parent.parent;BIN=LG/'DiskANN/build/apps/search_disk_index'
classification=json.loads((ROOT/'classification_summary.json').read_text())
def readbin(path,dtype):
 with open(path,'rb') as f:n,d=np.fromfile(f,np.uint32,2);return np.fromfile(f,dtype,count=int(n)*int(d)).reshape(int(n),int(d))
def execute(ds,split,name,tau,rep=0):
 out=ROOT/ds/'online'/split/f'r{rep}_{name}';out.parent.mkdir(parents=True,exist_ok=True)
 model_name=name.split('__')[0]
 env=os.environ.copy()
 for k in list(env):
  if k.startswith('DISKANN_'):env.pop(k)
 env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='1',DISKANN_QUERY_STATS=str(out)+'.csv')
 if name!='baseline':env.update(DISKANN_M3_MODEL=str(ROOT/ds/model_name/'model.bin'),DISKANN_M3_TAU=str(tau),DISKANN_M3_AUDIT=str(out)+'.audit')
 args=[str(BIN),'--data_type','float','--dist_fn','l2','--index_path_prefix',str(LG/'experiments'/('sift1m' if ds=='sift' else 'deep1m')/'index'),'--query_file',str(ROOT/ds/split/'query.fbin'),'--gt_file',str(ROOT/ds/split/'gt.bin'),'-K','10','-L','100','--result_path',str(out),'-W','4','-T','4','--num_nodes_to_cache','10000']
 (Path(str(out)+'.command.json')).write_text(json.dumps({'args':args,'env':{k:v for k,v in env.items() if k.startswith('DISKANN_') or k in ['OMP_NUM_THREADS','MKL_NUM_THREADS']}},indent=2)+'\n')
 with open(str(out)+'.log','w') as f:subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 return out

def stats(prefix,baseline,ds,split):
 ids=readbin(str(prefix)+'_100_idx_uint32.bin',np.uint32);dists=readbin(str(prefix)+'_100_dists_float.bin',np.float32)
 gt=readbin(ROOT/ds/split/'gt.bin',np.uint32)[:,:10]
 bids=readbin(str(baseline)+'_100_idx_uint32.bin',np.uint32);bdists=readbin(str(baseline)+'_100_dists_float.bin',np.float32)
 hits=lambda a:np.array([len(set(x)&set(g)) for x,g in zip(a,gt)])
 rec,brec=hits(ids),hits(bids)
 rows=list(csv.DictReader(open(str(prefix)+'.csv')));br=list(csv.DictReader(open(str(baseline)+'.csv')))
 reads=sum(int(r['ssd_reads']) for r in rows);base_reads=sum(int(r['ssd_reads']) for r in br)
 return {'submitted_ssd_read_requests':reads,'reference_submitted_requests':base_reads,'net_io_saving_requests':base_reads-reads,'net_io_saving_ratio':1-reads/base_reads,
         'recall_at_10':float(rec.mean()/10),'reference_recall_at_10':float(brec.mean()/10),'recall_loss_pp':float((brec.mean()-rec.mean())*10),
         'quality_degraded_queries':int((rec<brec).sum()),'quality_improved_queries':int((rec>brec).sum()),
         'strict_result_preserved_queries':int(((ids==bids).all(axis=1)&(dists==bdists).all(axis=1)).sum()),
         'mean_expansions':float(np.mean([int(r['expansions']) for r in rows])),
         'mean_latency_us':float(np.mean([float(r['latency_us']) for r in rows])),
         'gate_skipped_per_query':float(np.mean([int(r['gate_skipped']) for r in rows]))}
summary={}
for ds in ['sift','deep']:
 valbase=execute(ds,'validation','baseline',0)
 validation={};choices={}
 for name,info in classification[ds]['models'].items():
  entries={}
  for key,t in info['thresholds'].items():
   tag=name+'__'+key;p=execute(ds,'validation',tag,t['retain_threshold'])
   entries[key]=stats(p,valbase,ds,'validation')
  validation[name]=entries
  # Chosen exclusively by actual validation quality and request counts.
  for budget in [0,0.1]:
   valid=[(key,r) for key,r in entries.items() if r['recall_loss_pp']<=budget+1e-9 and r['net_io_saving_ratio']>0]
   choices[name+f'__budget_{budget:g}pp']=None if not valid else min(valid,key=lambda kr:kr[1]['submitted_ssd_read_requests'])[0]
 selected_model=max(classification[ds]['models'],key=lambda n:classification[ds]['models'][n]['metrics']['validation']['average_precision'])
 test={}
 testbase=execute(ds,'test','baseline',0)
 # Evaluate every frozen classifier/threshold pair to expose quality drift; never select on test.
 for name,info in classification[ds]['models'].items():
  for key,t in info['thresholds'].items():
   tag=name+'__'+key;p=execute(ds,'test',tag,t['retain_threshold']);test[tag]=stats(p,testbase,ds,'test')
 for name,key in choices.items():
  model_name=name.split('__')[0]
  test[name]=stats(testbase,testbase,ds,'test') if key is None else test[model_name+'__'+key]
 summary[ds]={'validation':validation,'validation_selected_thresholds':choices,'validation_selected_model_by_ap':selected_model,'test':test}
 (ROOT/'online_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
