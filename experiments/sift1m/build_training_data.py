#!/usr/bin/env python3
"""L100 expansion samples and neighbor contexts; strictly query-disjoint splits."""
import sys,os,json,csv,hashlib,bisect
from pathlib import Path
import numpy as np
E=Path(__file__).resolve().parent; P=E.parents[1]; V=E/'v2'; OUT=E/'training_data_l100'
sys.path.insert(0,str(V)); from read_trace import blocks,EDGE
SEED=42; DEGREE=64; WIDTH=4; L=100
PRE=['candidate_pq_distance','best_pq_distance','worst_pq_distance','candidate_queue_rank','queue_size','unexpanded_count','visited_count','completed_expansions','beam_step','beam_position','cached_node','best_exact_before_beam','has_best_exact','candidate_pq_over_best','candidate_pq_over_worst','search_L','beam_width']
SCAN=['current_distance_pq','current_distance_exact','current_best_pq_distance','current_best_exact_distance','queue_size','visited_count','step_id','beam_step','cached_node','degree','candidate_pq_over_best','candidate_exact_over_best']
META=np.dtype([('sample_id','<u8'),('query_id','<u4'),('query_row','<u4'),('step_id','<u4'),('beam_step','<u4'),('node_id','<u4'),('label','u1'),('oracle_zero_admission','u1'),('oracle_in_final_topk','u1')])
NEIGHBOR=np.dtype(EDGE.descr+[('expansion_sample_id','<u8'),('expansion_row','<u8'),('parent_y','u1')])
def binmat(path,dtype):
 with open(path,'rb') as f:
  n,d=np.frombuffer(f.read(8),dtype='<u4'); a=np.fromfile(f,dtype=dtype)
 assert len(a)==int(n)*int(d); return a.reshape(int(n),int(d))
def create(path,dtype,shape): return np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape)
def replay(q,ex,ed):
 """Rebuild selection state BEFORE cache lookup/read using only earlier events."""
 n=len(ex); pre=np.empty((n,len(PRE)),dtype='<f4'); scan=np.empty((n,len(SCAN)),dtype='<f4')
 assert np.all(ex['degree']==DEGREE) and len(ed)==n*DEGREE
 first=ed[::DEGREE]; assert np.array_equal(first['current_node_id'],ex['node'])
 seed=int(ex['node'][0]); queue=[(float(first['current_distance_pq'][0]),seed)]
 expanded=set(); visited_count=1; unexpanded=1; completed=0; best_exact=np.inf
 start=0
 while start<n:
  beam=int(first['beam_step'][start]); end=start+1
  while end<n and int(first['beam_step'][end])==beam: end+=1
  beam_ids=list(map(int,ex['node'][start:end])); cache={int(ex['node'][i]):int(ex['cached'][i]) for i in range(start,end)}
  chosen=[]; state={}
  for pos in range(WIDTH):
   index=next((i for i,(_,node) in enumerate(queue) if node not in expanded),None)
   if index is None: break
   pq,node=queue[index]; assert node in cache,(q,beam,node,beam_ids)
   best=queue[0][0]; worst=queue[-1][0]
   state[node]=[pq,best,worst,index,len(queue),unexpanded,visited_count,completed,beam,pos,cache[node],float(best_exact) if np.isfinite(best_exact) else 0.0,int(np.isfinite(best_exact)),pq/max(best,1e-12),pq/max(worst,1e-12),L,WIDTH]
   chosen.append(node); expanded.add(node); unexpanded-=1
  expected=[x for x in chosen if cache[x]]+[x for x in chosen if not cache[x]]
  assert expected==beam_ids,(q,beam,expected,beam_ids)
  for i in range(start,end):
   x=first[i]; node=int(ex['node'][i]); pre[i]=state[node]
   assert np.float32(state[node][0])==x['current_distance_pq']
   assert len(queue)==int(x['queue_size']) and visited_count==int(x['visited_count'])
   assert np.float32(queue[0][0])==x['current_best_distance']
   best_exact=min(best_exact,float(ex['exact'][i])); assert np.float32(best_exact)==x['current_best_exact_distance']
   scan[i]=[x['current_distance_pq'],x['current_distance_exact'],x['current_best_distance'],x['current_best_exact_distance'],x['queue_size'],x['visited_count'],i,beam,ex['cached'][i],ex['degree'][i],float(x['current_distance_pq'])/max(float(x['current_best_distance']),1e-12),float(x['current_distance_exact'])/max(float(x['current_best_exact_distance']),1e-12)]
   neighborhood=ed[i*DEGREE:(i+1)*DEGREE]
   for edge in neighborhood[neighborhood['neighbor_was_inserted']!=0]:
    item=(float(edge['neighbor_distance_pq']),int(edge['neighbor_id']))
    assert item[1] not in expanded and (len(queue)<L or not(queue[-1]<item))
    bisect.insort(queue,item); unexpanded+=1
    if len(queue)>L:
     evicted=queue.pop()
     if evicted[1] not in expanded: unexpanded-=1
   visited_count+=int(ex['fresh'][i]); completed+=1
  assert unexpanded==sum(node not in expanded for _,node in queue)
  start=end
 assert completed==n and not any(node not in expanded for _,node in queue)
 return pre,scan,visited_count
if __name__=='__main__':
 assert json.loads((V/'COMPLETE.json').read_text())['validated']
 source_rows=list(csv.DictReader((V/'per_query_100.csv').open())); source_rows.sort(key=lambda x:int(x['query_id']))
 n_by_q=np.array([int(x['expanded']) for x in source_rows],dtype=np.int64); assert len(n_by_q)==10000
 global_offsets=np.r_[0,np.cumsum(n_by_q)]
 masks=[set() for _ in range(10000)]
 for line in (E/'full_skip_100.mask').open():
  q,node=map(int,line.split()); assert node not in masks[q]; masks[q].add(node)
 safe_by_q=np.array([len(x) for x in masks],dtype=np.int64)
 assert n_by_q.sum()==1117633 and safe_by_q.sum()==642134
 assert int(safe_by_q.sum())*DEGREE==41096576
 perm=np.random.default_rng(SEED).permutation(10000)
 ids={'train':np.sort(perm[:8000]),'validation':np.sort(perm[8000:9000]),'test':np.sort(perm[9000:])}
 assert all(not(set(ids[a])&set(ids[b])) for a,b in [('train','validation'),('train','test'),('validation','test')])
 assert set(np.concatenate(list(ids.values())))==set(range(10000))
 OUT.mkdir(exist_ok=True)
 (OUT/'COMPLETE.json').unlink(missing_ok=True)
 split_by_q={}; query_local={}; files={}; split_offsets={}; edge_offsets={}; safe_offsets={}; summaries={}
 queries=binmat(P/'data/sift/sift_query.fbin','<f4'); final=binmat(V/'results_100_idx_uint32.bin','<u4')
 base_manifest={'path':str(P/'data/sift/sift_base.fbin'),'dtype':'<f4','offset_bytes':8,'shape':[1000000,128]}
 np.save(OUT/'query_split.npy',np.full(10000,255,dtype='u1'))
 split_codes=np.load(OUT/'query_split.npy',mmap_mode='r+')
 for code,(name,qids) in enumerate(ids.items()):
  folder=OUT/name; folder.mkdir(exist_ok=True); count=int(n_by_q[qids].sum()); safe=int(safe_by_q[qids].sum())
  np.save(folder/'query_ids.npy',qids.astype('<u4')); np.save(folder/'query_vectors.npy',queries[qids])
  offsets=np.r_[0,np.cumsum(n_by_q[qids])].astype('<u8'); np.save(folder/'query_expansion_offsets.npy',offsets)
  np.save(folder/'query_neighbor_offsets.npy',(offsets*DEGREE).astype('<u8'))
  so=np.r_[0,np.cumsum(safe_by_q[qids])*DEGREE].astype('<u8'); np.save(folder/'query_safe_neighbor_offsets.npy',so)
  for i,q in enumerate(qids):
   q=int(q); split_by_q[q]=name; query_local[q]=i; split_offsets[q]=int(offsets[i]); edge_offsets[q]=int(offsets[i])*DEGREE; safe_offsets[q]=int(so[i]); split_codes[q]=code
  files[name]={'metadata':create(folder/'metadata.npy',META,(count,)),'y':create(folder/'y.npy','u1',(count,)), 'pre_io_state':create(folder/'pre_io_state.npy','<f4',(count,len(PRE))), 'post_score_state':create(folder/'post_score_state.npy','<f4',(count,len(SCAN))), 'neighbors':create(folder/'neighbors.npy',NEIGHBOR,(count*DEGREE,)), 'safe_skip_neighbors':create(folder/'safe_skip_neighbors.npy',NEIGHBOR,(safe*DEGREE,))}
  summaries[name]={'queries':len(qids),'expansion_samples':count,'safe_to_skip_y0':safe,'retain_y1':count-safe,'neighbor_observations':count*DEGREE,'safe_skip_neighbor_observations':safe*DEGREE}
 split_codes.flush()
 seen=set(); checked_safe=0; edge_total=0
 for q,l,visited,ex,ed in blocks(V/'trace_100.trace'):
  assert l==L and q not in seen; seen.add(q); n=len(ex); assert n==n_by_q[q]
  name=split_by_q[q]; f=files[name]; offset=split_offsets[q]; erow=edge_offsets[q]; srow=safe_offsets[q]
  node_ids=ex['node']; in_final=np.isin(node_ids,final[q]); zero_admit=(ex['admitted']==0)
  safe=zero_admit & ~in_final; oracle_nodes=set(map(int,node_ids[safe])); assert oracle_nodes==masks[q]
  labels=(~safe).astype('u1'); assert int(safe.sum())==safe_by_q[q]
  pre,scan,replayed_visited=replay(q,ex,ed); assert replayed_visited==visited
  assert np.all(np.isfinite(pre)) and np.all(np.isfinite(scan))
  meta=np.empty(n,dtype=META); meta['sample_id']=global_offsets[q]+np.arange(n,dtype=np.uint64); meta['query_id']=q; meta['query_row']=query_local[q]; meta['step_id']=np.arange(n); meta['beam_step']=ed[::DEGREE]['beam_step']; meta['node_id']=node_ids; meta['label']=labels; meta['oracle_zero_admission']=zero_admit; meta['oracle_in_final_topk']=in_final
  f['metadata'][offset:offset+n]=meta; f['y'][offset:offset+n]=labels; f['pre_io_state'][offset:offset+n]=pre; f['post_score_state'][offset:offset+n]=scan
  nr=np.empty(len(ed),dtype=NEIGHBOR)
  for field in EDGE.names: nr[field]=ed[field]
  steps=ed['step_id']; nr['expansion_sample_id']=global_offsets[q]+steps; nr['expansion_row']=offset+steps; nr['parent_y']=labels[steps]
  assert np.array_equal(nr['current_node_id'],node_ids[steps]) and np.all(nr['query_id']==q)
  f['neighbors'][erow:erow+len(nr)]=nr
  safe_edges=nr[nr['parent_y']==0]; assert len(safe_edges)==int(safe.sum())*DEGREE
  f['safe_skip_neighbors'][srow:srow+len(safe_edges)]=safe_edges
  checked_safe+=len(safe_edges); edge_total+=len(nr)
  if len(seen)%1000==0: print(f'Generated and replay-validated {len(seen)}/10000 queries',flush=True)
 assert seen==set(range(10000)) and checked_safe==41096576 and edge_total==71528512
 for name,arrays in files.items():
  for a in arrays.values(): a.flush()
  folder=OUT/name; y=np.load(folder/'y.npy',mmap_mode='r'); meta=np.load(folder/'metadata.npy',mmap_mode='r')
  assert np.array_equal(y,meta['label']) and int((y==0).sum())==summaries[name]['safe_to_skip_y0']
  assert set(map(int,np.unique(meta['query_id'])))==set(map(int,ids[name]))
  neighbors=np.load(folder/'neighbors.npy',mmap_mode='r'); safen=np.load(folder/'safe_skip_neighbors.npy',mmap_mode='r')
  # All exported edges checked against parent samples in bounded-size readback chunks.
  for edges in [neighbors,safen]:
   for a in range(0,len(edges),1000000):
    chunk=edges[a:a+1000000]; parent=meta[chunk['expansion_row']]
    assert np.array_equal(chunk['parent_y'],parent['label'])
    assert np.array_equal(chunk['query_id'],parent['query_id'])
    assert np.array_equal(chunk['current_node_id'],parent['node_id'])
    assert np.array_equal(chunk['expansion_sample_id'],parent['sample_id'])
    assert np.array_equal(chunk['step_id'],parent['step_id'])
    assert np.all(np.isfinite(chunk['neighbor_distance_exact']))
   if edges is safen: assert np.all(edges['parent_y']==0)
 train_stats={}
 for view,columns in [('pre_io_state',PRE),('post_score_state',SCAN)]:
  x=np.load(OUT/'train'/f'{view}.npy',mmap_mode='r'); mean=x.mean(axis=0,dtype=np.float64); std=x.std(axis=0,dtype=np.float64); scale=np.where(std>1e-12,std,1.0)
  np.savez(OUT/f'{view}_train_normalization.npz',mean=mean.astype('<f4'),scale=scale.astype('<f4'))
  train_stats[view]={'columns':columns,'mean':mean.tolist(),'std':std.tolist(),'scale':scale.tolist(),'fit_split':'train'}
 manifest={'version':1,'L':100,'seed':SEED,'query_split_ratios':{'train':0.8,'validation':0.1,'test':0.1},'splits':summaries,'total_expansion_samples':1117633,'total_safe_to_skip':642134,'total_retain':475499,'total_neighbor_observations':71528512,'total_safe_skip_neighbor_observations':41096576,'source_trace':str(V/'trace_100.trace'),'source_trace_version':2,'source_query_results':str(V/'results_100_idx_uint32.bin'),'oracle_mask':str(E/'full_skip_100.mask'),'base_vectors':base_manifest,'pre_io_state_columns':PRE,'post_score_state_columns':SCAN,'normalization':train_stats,'label_definition':'y=0 iff baseline expansion admitted zero new candidates AND current node is not in baseline final top10; exact full-skip oracle mask verified identical results for all queries. y=1 means conservatively retain, not proof of necessity.','primary_sample_unit':'one candidate node expansion','neighbor_unit':'edge observation joined to parent expansion; parent_y is not an independently certified neighbor-edge skip label','pre_io_replay':'Full baseline candidate queue and original beam slot selection reconstructed and checked against every traced expansion. Uses only history from previous processed expansions; no current-node exact distance or future labels.','query_disjoint_verified':True,'all_expansion_selection_states_replay_verified':True,'all_neighbor_parent_joins_verified':True,'state_features_exclude_future_labels':True,'raw_v_vector_cost':'v vectors are resolved from existing base file for offline model input; fetching uncached full v on SSD has cost, and is not a free pre-IO feature.'}
 for key,path in [('oracle_mask_sha256',E/'full_skip_100.mask'),('query_vectors_sha256',P/'data/sift/sift_query.fbin'),('validation_sha256',V/'validation.json')]: manifest[key]=hashlib.sha256(path.read_bytes()).hexdigest()
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
 (OUT/'COMPLETE.json').write_text(json.dumps({'complete':True,'queries':10000,'expansions':1117633,'neighbor_observations':71528512,'safe_skip_neighbor_observations':41096576,'query_disjoint':True,'parent_joins_validated':True,'beam_replay_validated':True},indent=2))
 print(json.dumps(summaries,indent=2),flush=True); print('TRAINING DATA COMPLETE',flush=True)
