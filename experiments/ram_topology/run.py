from pathlib import Path
import numpy as np,json,collections,subprocess,os,random,csv,time
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'ram_navigation';LG=ROOT.parent.parent
ns={'__file__':str(OLD/'build.py')};exec((OLD/'build.py').read_text().split('manifest={}')[0],ns)
en={'__file__':str(OLD/'evaluate.py')};exec((OLD/'evaluate.py').read_text().split('summary={}')[0],en);en['ROOT']=ROOT
for ds in ['sift','deep']:(ROOT/f'{ds}_bfs.json').write_bytes((OLD/f'{ds}_bfs.json').read_bytes())
def save(ds,name,selected,table,degree):
 packed=np.full(len(selected)*16,0xffffffff,np.uint32);flat=np.concatenate([table[i,:int(degree[i])] for i in range(len(selected))]);packed[:len(flat)]=flat
 offset=np.r_[0,np.cumsum(degree[:-1])].astype(np.uint32);ns['write_graph'](ROOT/ds/name,selected,packed,offset,True)
def geometry(v,degree):
 n=len(v);norm=np.sum(v*v,axis=1);near=np.empty((n,128),np.int32)
 for lo in range(0,n,256):
  d=norm[lo:lo+256,None]+norm[None,:]-2*v[lo:lo+256]@v.T;d[np.arange(len(d)),np.arange(lo,lo+len(d))]=np.inf
  ix=np.argpartition(d,128,axis=1)[:,:128];near[lo:lo+len(d)]=np.take_along_axis(ix,np.argsort(np.take_along_axis(d,ix,axis=1),axis=1),axis=1)
 table=np.full((n,64),0xffffffff,np.uint32)
 for i in range(n):
  k=int(degree[i]);pool=near[i];sd=np.sum((v[pool]-v[i])**2,axis=1);remaining=np.ones(len(pool),bool);chosen=[]
  while remaining.any() and len(chosen)<k:
   at=np.flatnonzero(remaining)[0];node=int(pool[at]);chosen.append(node);remaining&=1.2*np.sum((v[pool]-v[node])**2,axis=1)>sd;remaining[at]=False
  for node in pool:
   if len(chosen)==k:break
   if int(node) not in chosen:chosen.append(int(node))
  table[i,:k]=chosen
 return table
summary={}
for ds in ['sift','deep']:
 directory=ROOT/ds;directory.mkdir(exist_ok=True);selected=ns['ids'](str(OLD/ds/'B3.ids'));base=ns['matrix'](LG/('data/sift/sift_base.fbin' if ds=='sift' else 'data/deep1M_base.fbin'));v=np.asarray(base[selected]);q=ns['matrix'](OLD/ds/'history.fbin')
 disk=LG/'experiments'/('sift1m' if ds=='sift' else 'deep1m')/'index_disk.index';original,degree=ns['original_edges'](disk,selected,v.shape[1],64)
 for i in range(len(selected)):assert len(set(map(int,original[i,:degree[i]])))==degree[i] and i not in original[i,:degree[i]]
 print(ds,'geometry',flush=True);geom=geometry(v,degree);save(ds,'Original',selected,original,degree);save(ds,'Geometry',selected,geom,degree)
 args=[str(en['BIN']),'profile',str(disk).replace('_disk.index',''),str(directory/'Geometry'),str(OLD/ds/'history.fbin'),str(directory/'train_geometry'),'64']
 with open(directory/'train_geometry.log','w') as f:subprocess.run(args,env=en['ENV'],stdout=f,stderr=subprocess.STDOUT,check=True)
 paths=ns['read_profile'](directory/'train_geometry.profile');assert len(paths)==len(q);owners,dmin=ns['nearest'](q,v);pairs=collections.defaultdict(collections.Counter)
 for row,path in paths.items():
  target=int(owners[row]);dist=np.sum((v[path]-q[row])**2,axis=1)
  for source,dd in zip(path,dist):
   if int(source)!=target and dd>dmin[row]*1.05:pairs[int(source)][target]+=float((dd-dmin[row])/max(dd,1e-12))
 graphs={'Original':original,'Geometry':geom};historical_counts={}
 for fraction in [.25,.5,1.]:
  name='History'+str(int(fraction*100));table=geom.copy();count=0
  for source,candidates in pairs.items():
   k=int(degree[source]);quota=min(k,int(np.ceil(k*fraction)));ranked=sorted(candidates,key=lambda t:(-candidates[t],t))[:quota];chosen=ranked.copy()
   for target in geom[source,:k]:
    if len(chosen)==k:break
    if int(target) not in chosen:chosen.append(int(target))
   table[source,:k]=chosen;count+=len(ranked)
  graphs[name]=table;historical_counts[name]=count;save(ds,name,selected,table,degree)
 manifest={'nodes':len(selected),'effective_edges':int(degree.sum()),'degree_zero_nodes':int((degree==0).sum()),'same_per_source_degrees':True,'history_queries':len(q),'history_selected_slots':historical_counts,'root_reachability':{name:ns['reach'](t,degree) for name,t in graphs.items()}}
 np.save(directory/'degree.npy',degree);(directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 validation={}
 for method in graphs:
  print(ds,'validation',method,flush=True);validation[method]=en['execute'](ds,method,'validation',64,method)
 feasible=[m for m in historical_counts if validation[m]['recall_at_10']>=validation['Geometry']['recall_at_10']-1e-12]
 chosen=min(feasible,key=lambda m:validation[m]['submitted_requests']) if feasible else min(historical_counts,key=lambda m:(-validation[m]['recall_at_10'],validation[m]['submitted_requests']))
 methods=['Original','Geometry',chosen];runs={m:[] for m in methods}
 for repeat in range(3):
  order=methods.copy();random.Random(20261014+repeat).shuffle(order)
  for method in order:
   print(ds,'test',repeat,method,flush=True);runs[method].append(en['execute'](ds,method,'test',64,f'r{repeat}_{method}'))
 aggregate={}
 for method,rr in runs.items():
  assert len({r['submitted_requests'] for r in rr})==1 and len({r['recall_at_10'] for r in rr})==1
  aggregate[method]={k:float(np.median([r[k] for r in rr])) for k in ['mean_requests','recall_at_10','mean_latency_us','mean_navigation_us','mean_navigation_distances','mean_navigation_expansions','qps']};aggregate[method]['memory_bytes']=rr[0]['memory']['cache_and_navigation_bytes'];aggregate[method]['qps_runs']=[r['qps'] for r in rr]
 assert len({a['memory_bytes'] for a in aggregate.values()})==1
 hist=np.load(OLD/ds/'history_query_ids.npy');val=np.load(LG/'experiments/io_predictability'/ds/'validation/query_ids.npy');test=np.load(LG/'experiments/io_predictability'/ds/'test/query_ids.npy');assert not(set(hist)&set(val) or set(hist)&set(test) or set(val)&set(test))
 a=list(csv.DictReader(open(runs['Geometry'][0]['prefix']+'.csv')));b=list(csv.DictReader(open(runs[chosen][0]['prefix']+'.csv')))
 delta=np.array([int(x['submitted_requests'])-int(y['submitted_requests']) for x,y in zip(a,b)]);hits=np.array([int(y['hits'])-int(x['hits']) for x,y in zip(a,b)]);rng=np.random.default_rng(20261014);samples=rng.integers(0,len(a),(5000,len(a)))
 paired={'queries_recall_lower':int((hits<0).sum()),'queries_recall_higher':int((hits>0).sum()),'queries_fewer_requests':int((delta>0).sum()),'queries_more_requests':int((delta<0).sum()),'mean_request_saving':float(delta.mean()),'mean_request_saving_ci95':np.quantile(delta[samples].mean(axis=1),[.025,.975]).tolist(),'recall_difference_pp_ci95':(np.quantile(hits[samples].mean(axis=1),[.025,.975])*10).tolist()}
 summary[ds]={'manifest':manifest,'validation':validation,'selected_history':chosen,'validation_recall_feasible':bool(feasible),'test':aggregate,'paired_history_vs_geometry':paired,'raw_test_runs':runs,'query_disjoint':True}
 (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(ds,aggregate,paired,flush=True)
