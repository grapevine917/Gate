#!/usr/bin/env python3
"""Collect all four budgets, annotate exact distances, validate every field."""
import os,sys,json,struct,csv,subprocess,bisect
from pathlib import Path
import numpy as np
from read_trace import EXP,EDGE
V=Path(__file__).resolve().parent; E=V.parent; P=E.parents[1]; D=P/'DiskANN'; LS=[20,50,100,150]
ENV=os.environ.copy(); ENV.update(MKL_NUM_THREADS='4',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
for k in ['DISKANN_SKIP_PREFIX','DISKANN_SKIP_ENTIRE_NODE','DISKANN_TRACE_PREFIX','DISKANN_TRACE_VERSION']: ENV.pop(k,None)
def binmat(path,dtype):
 with open(path,'rb') as f:
  n,d=struct.unpack('<II',f.read(8)); a=np.fromfile(f,dtype=dtype)
 assert len(a)==n*d; return a.reshape(n,d)
def run(args,name,extra=None):
 env=ENV.copy(); env.update(extra or {})
 (V/(name+'.command.json')).write_text(json.dumps({'args':args,'env':extra or {}},indent=2))
 with open(V/name,'w') as log: subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
def search(name,extra):
 args=[str(D/'build/apps/search_disk_index'),'--data_type','float','--dist_fn','l2','--index_path_prefix',str(E/'index'),'--query_file',str(P/'data/sift/sift_query.fbin'),'--gt_file',str(P/'data/sift/gt100.bin'),'-K','10','-L',*map(str,LS),'--result_path',str(V/name),'-W','4','-T','4','--num_nodes_to_cache','10000']
 run(args,name+'.log',extra)
def parity(name,L):
 for suffix,dtype in [('idx_uint32','<u4'),('dists_float','<f4')]:
  a=binmat(V/f'{name}_{L}_{suffix}.bin',dtype); b=binmat(E/f'baseline_{L}_{suffix}.bin',dtype); assert np.array_equal(a,b),(name,L,suffix)
if __name__=='__main__':
 if '--analyze-only' not in sys.argv:
  run(['cmake','--build',str(D/'build'),'--target','search_disk_index','-j','4'],'build.log')
  print('Compiled v2. Checking disabled logger and collecting all queries.',flush=True)
  search('logger_off',{})
  search('results',{'DISKANN_TRACE_VERSION':'2','DISKANN_TRACE_PREFIX':str(V/'trace')})
 for L in LS: parity('logger_off',L); parity('results',L)
 print('All 40000 query results and distances match original baseline exactly.',flush=True)
 base=np.memmap(P/'data/sift/sift_base.fbin',mode='r',dtype='<f4',offset=8,shape=(1000000,128))
 queries=np.memmap(P/'data/sift/sift_query.fbin',mode='r',dtype='<f4',offset=8,shape=(10000,128))
 gt=binmat(P/'data/sift/gt100.bin','<u4'); summary=[]
 sample=(V/'edges_sample.csv').open('w',newline=''); writer=csv.writer(sample); writer.writerow(['L',*EDGE.names])
 replay_ids=set(map(int,np.linspace(0,9999,32,dtype=int)))
 for L in LS:
  final=binmat(V/f'results_{L}_idx_uint32.bin','<u4'); rows=[]; seen=set(); edge_count=0
  counts={k:0 for k in ['inserted','expanded_edges','led_to_improvement','final_topk_edges','direct_pq_improvements','direct_exact_improvements']}
  with open(V/f'trace_{L}.trace','r+b') as f:
   assert f.read(8)==b'LGTRACE2'
   while True:
    raw=f.read(20)
    if not raw: break
    assert len(raw)==20; q,l,visited,n,m=struct.unpack('<5I',raw)
    assert l==L and q not in seen; seen.add(q)
    ex=np.frombuffer(f.read(n*24),dtype=EXP); offset=f.tell(); buf=bytearray(f.read(m*88)); assert len(buf)==m*88
    ed=np.frombuffer(buf,dtype=EDGE)
    assert m==int(ex['degree'].sum()) and n==len(set(map(int,ex['node'])))
    assert np.all(ed['query_id']==q)
    fresh=((ed['flags']&1)!=0); admitted=ed['neighbor_was_inserted'].astype(bool)
    assert np.array_equal(admitted,((ed['flags']&2)!=0)) and np.all(~admitted|fresh)
    expected_visited=1+np.cumsum(fresh,dtype=np.uint32)-fresh
    assert np.array_equal(ed['visited_count'],expected_visited) and visited==int(fresh.sum())+1
    expected_queue=np.minimum(L,1+np.cumsum(admitted,dtype=np.uint32)-admitted)
    assert np.array_equal(ed['queue_size'],expected_queue)
    assert int(admitted.sum())==int(ex['admitted'].sum()) and int(fresh.sum())==int(ex['fresh'].sum())
    steps=np.repeat(np.arange(n,dtype=np.uint32),ex['degree'])
    assert np.array_equal(ed['step_id'],steps)
    assert np.array_equal(ed['current_node_id'],ex['node'][steps])
    assert np.array_equal(ed['current_distance_exact'],ex['exact'][steps])
    expected_exact_best=np.minimum.accumulate(ex['exact'])
    assert np.array_equal(ed['current_best_exact_distance'],expected_exact_best[steps])
    best0=ed['current_best_distance'][0]
    pq_prefix=np.minimum.accumulate(np.minimum(best0,np.where(admitted,ed['neighbor_distance_pq'],np.inf)))
    expected_pq=np.r_[np.float32(best0),pq_prefix[:-1]]
    assert np.array_equal(ed['current_best_distance'],expected_pq)
    assert np.array_equal(ed['neighbor_improved_best_pq'].astype(bool),admitted & (ed['neighbor_distance_pq']<ed['current_best_distance']))
    starts=np.r_[0,np.cumsum(ex['degree'])[:-1]].astype(int)
    for step,start in enumerate(starts):
     a=ed[start:start+int(ex['degree'][step])]
     assert np.array_equal(a['neighbor_position'],np.arange(len(a)))
     order=np.lexsort((a['neighbor_position'],a['neighbor_id'],a['neighbor_distance_pq']))
     ranks=np.empty(len(a),dtype=np.uint32); ranks[order]=np.arange(len(a))
     assert np.array_equal(a['neighbor_rank'],ranks)
     assert np.all(a['current_distance_pq']==a['current_distance_pq'][0])
     assert np.all(a['beam_step']==a['beam_step'][0])
    assert np.all(np.diff(ed['beam_step'].astype(np.int64))>=0)
    # Exact distance annotations use unique IDs so repeated edges do not repeat vector work.
    ids,inverse=np.unique(ed['neighbor_id'],return_inverse=True); assert ids.max()<len(base)
    distances=((base[ids]-queries[q])**2).sum(axis=1)
    ed['neighbor_distance_exact']=distances[inverse]
    parent=dict(zip(map(int,ed['neighbor_id'][fresh]),map(int,ed['current_node_id'][fresh])))
    previous=np.r_[np.float32(np.inf),expected_exact_best[:-1]]
    improved=set(map(int,ex['node'][(ex['exact']<previous)&(np.arange(n)>0)])); led=set()
    for node in improved:
     while node not in led:
      led.add(node)
      if node not in parent: break
      node=parent[node]
    assert np.array_equal(ed['neighbor_was_expanded'].astype(bool),np.isin(ed['neighbor_id'],ex['node']))
    assert np.array_equal(ed['neighbor_in_final_topk'].astype(bool),np.isin(ed['neighbor_id'],final[q]))
    assert np.array_equal(ed['neighbor_improved_best_exact'].astype(bool),fresh & np.isin(ed['neighbor_id'],list(improved)))
    assert np.array_equal(ed['neighbor_led_to_improvement'].astype(bool),fresh & np.isin(ed['neighbor_id'],list(led)))
    # All scored current-node distances independently checked from base vectors.
    current_exact=((base[ex['node']]-queries[q])**2).sum(axis=1)
    assert np.array_equal(current_exact,ex['exact'])
    assert np.all(np.isfinite(ed['neighbor_distance_exact']))
    # Detailed bounded-queue admission replay on 32 predetermined queries per L.
    if q in replay_ids:
     queue=[(float(ed['current_distance_pq'][0]),int(ex['node'][0]))]; visited_ids={int(ex['node'][0])}
     for x in ed:
      node=int(x['neighbor_id']); is_fresh=node not in visited_ids
      assert is_fresh==bool(x['flags']&1); visited_ids.add(node)
      item=(float(x['neighbor_distance_pq']),node)
      expected_insert=is_fresh and (len(queue)<L or not(queue[-1]<item))
      assert expected_insert==bool(x['neighbor_was_inserted'])
      if expected_insert:
       bisect.insort(queue,item)
       if len(queue)>L: queue.pop()
    # Commit annotations only after validation of this complete query.
    end=f.tell(); f.seek(offset); f.write(buf); f.seek(end)
    if q==0:
     for x in ed[:200]: writer.writerow([L,*[x[k].item() for k in EDGE.names]])
    for k,field in [('inserted','neighbor_was_inserted'),('expanded_edges','neighbor_was_expanded'),('led_to_improvement','neighbor_led_to_improvement'),('final_topk_edges','neighbor_in_final_topk'),('direct_pq_improvements','neighbor_improved_best_pq'),('direct_exact_improvements','neighbor_improved_best_exact')]: counts[k]+=int(ed[field].sum())
    rows.append({'query_id':q,'L':L,'visited':visited,'expanded':n,'edges':m,'useful':int((ex['admitted']>0).sum()),'useless':int((ex['admitted']==0).sum()),'inserted_edges':int(admitted.sum()),'direct_exact_improvements':len(improved),'discovery_edges_led_to_improvement':int(ed['neighbor_led_to_improvement'].sum())}); edge_count+=m
    if len(seen)%2000==0: print(f'L={L}: enriched and validated {len(seen)}/10000 queries',flush=True)
   f.flush(); os.fsync(f.fileno())
  assert seen==set(range(10000)); rows.sort(key=lambda x:x['query_id'])
  with open(V/f'per_query_{L}.csv','w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
  recall=np.mean([len(set(x)&set(y[:10]))/10 for x,y in zip(final,gt)])
  summary.append({'L':L,'queries':10000,'edges':edge_count,'recall_at_10':float(recall),'baseline_ids_identical':True,'baseline_distances_identical':True,'logger_off_parity':True,'all_fields_validated':True,'queue_replay_queries':sorted(replay_ids),'neighbor_exact_distances_complete':True,'label_counts':counts})
  (V/'validation.json').write_text(json.dumps(summary,indent=2))
 sample.close()
 (V/'COMPLETE.json').write_text(json.dumps({'version':2,'L':LS,'queries_per_L':10000,'validated':True,'exact_neighbor_distance_annotations_complete':True},indent=2))
 print('V2 COMPLETE',flush=True)
