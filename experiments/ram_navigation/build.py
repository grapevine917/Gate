"""Static history only; equal-size navigation tables; no neural training."""
from pathlib import Path
import numpy as np,json,struct,collections,subprocess,os,time
ROOT=Path(__file__).resolve().parent;LG=ROOT.parent.parent;BIN=LG/'DiskANN/build/apps/search_ram_navigation'
E=np.dtype([('node','<u4'),('degree','<u4'),('fresh','<u4'),('admitted','<u4'),('exact','<f4'),('cached','<u4')])
def matrix(path):
 with open(path,'rb') as f:n,d=np.fromfile(f,np.uint32,2)
 return np.memmap(path,mode='r',dtype=np.float32,offset=8,shape=(int(n),int(d)))
def ids(path):
 with open(path,'rb') as f:n=np.fromfile(f,np.uint32,1)[0];return np.fromfile(f,np.uint32,int(n))
def write_graph(prefix,selected,edges,degrees=None,csr=False):
 with open(str(prefix)+'.ids','wb') as f:np.array([len(selected)],np.uint32).tofile(f);np.asarray(selected,np.uint32).tofile(f)
 if degrees is None:degrees=np.full(len(selected),16,np.uint32)
 with open(str(prefix)+'.edges','wb') as f:np.array([len(selected),16|(0x80000000 if csr else 0),0],np.uint32).tofile(f);np.asarray(degrees,np.uint32).tofile(f);np.asarray(edges,np.uint32).tofile(f)
def ordered(root,selected):return np.array([root]+sorted(set(map(int,selected))-{root}),np.uint32)
def dist_to(q,v):return np.maximum(0,np.sum(q*q,axis=1)+float(v@v)-2*(q@v))
def nearest(q,v):
 out=np.empty(len(q),np.int32);best=np.empty(len(q),np.float32);norm=np.sum(v*v,axis=1)
 for lo in range(0,len(q),256):
  qq=q[lo:lo+256];d=np.sum(qq*qq,axis=1)[:,None]+norm[None,:]-2*qq@v.T;ix=np.argmin(d,axis=1);out[lo:lo+len(qq)]=ix;best[lo:lo+len(qq)]=np.maximum(0,d[np.arange(len(qq)),ix])
 return out,best

def history(ds,train_ids):
 path=LG/('experiments/sift1m/trace_100.trace' if ds=='sift' else 'experiments/deep1m/models/L100/trace_100.trace')
 freq=np.zeros(1000000,np.int64);bestnode={};seen=set();want=set(map(int,train_ids))
 with open(path,'rb') as f:
  magic=f.read(8);edge_size=16 if magic==b'LGTRACE1' else 88 if magic==b'LGTRACE2' else 0;assert edge_size
  while True:
   h=f.read(20)
   if not h:break
   q,L,vis,nexp,nedge=struct.unpack('<5I',h);e=np.frombuffer(f.read(nexp*24),E);f.seek(nedge*edge_size,1)
   if q in want:
    assert q not in seen and L==100;seen.add(q);np.add.at(freq,e['node'],1);bestnode[q]=int(e['node'][np.argmin(e['exact'])])
 assert seen==want
 return freq,np.array([bestnode[int(q)] for q in train_ids],np.uint32)

def original_edges(path,selected,dimension,edge_cap=16):
 disk=np.memmap(path,mode='r',dtype=np.uint8);_,_,root,size,persector=np.frombuffer(disk[8:48],'<u8');assert persector>0
 lookup={int(n):i for i,n in enumerate(selected)};table=np.full((len(selected),edge_cap),0xffffffff,np.uint32);degrees=np.zeros(len(selected),np.uint32)
 for i,node in enumerate(selected):
  offset=(int(node)//int(persector)+1)*4096+(int(node)%int(persector))*int(size)
  degree=int(np.frombuffer(disk[offset+dimension*4:offset+dimension*4+4],'<u4')[0]);neighbors=np.frombuffer(disk[offset+dimension*4+4:offset+dimension*4+4+degree*4],'<u4')
  resident=[lookup[int(x)] for x in neighbors if int(x) in lookup][:edge_cap];table[i,:len(resident)]=resident;degrees[i]=len(resident)
 return table,degrees

def rebuilt(v,seed=42):
 n=len(v);table=np.empty((n,16),np.uint32);norm=np.sum(v*v,axis=1);near=np.empty((n,64),np.int32)
 for lo in range(0,n,256):
  d=norm[lo:lo+256,None]+norm[None,:]-2*v[lo:lo+256]@v.T;d[np.arange(len(d)),np.arange(lo,lo+len(d))]=np.inf
  chosen=np.argpartition(d,64,axis=1)[:,:64];near[lo:lo+len(d)]=np.take_along_axis(chosen,np.argsort(np.take_along_axis(d,chosen,axis=1),axis=1),axis=1)
 rng=np.random.default_rng(seed);cycle=rng.permutation(n);nxt=np.empty(n,np.int32);prv=np.empty(n,np.int32);nxt[cycle]=np.roll(cycle,-1);prv[cycle]=np.roll(cycle,1)
 for i in range(n):
  pool=np.unique(np.concatenate([near[i],rng.integers(0,n,32)]));pool=pool[(pool!=i)&(pool!=nxt[i])&(pool!=prv[i])]
  selfdist=np.sum((v[pool]-v[i])**2,axis=1);order=np.lexsort((pool,selfdist));pool=pool[order];selfdist=selfdist[order];remaining=np.ones(len(pool),bool);chosen=[]
  while remaining.any() and len(chosen)<14:
   at=np.flatnonzero(remaining)[0];node=int(pool[at]);chosen.append(node);d=np.sum((v[pool]-v[node])**2,axis=1);remaining&=1.2*d>selfdist;remaining[at]=False
  for node in pool:
   if len(chosen)==14:break
   if int(node) not in chosen:chosen.append(int(node))
  assert len(chosen)==14;table[i]=chosen+[int(nxt[i]),int(prv[i])]
 return table

def read_profile(path):
 out={}
 with open(path,'rb') as f:
  while True:
   h=f.read(8)
   if not h:break
   q,n=struct.unpack('<2I',h);out[q]=np.frombuffer(f.read(n*4),'<u4').copy()
 return out

def profile(ds,prefix,label):
 directory=ROOT/ds;output=directory/label;env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='1');env.pop('RAM_USE_ORIGINAL_EDGES',None)
 args=[str(BIN),'profile',str(LG/'experiments'/('sift1m' if ds=='sift' else 'deep1m')/'index'),str(prefix),str(directory/'history.fbin'),str(output),'64']
 with open(str(output)+'.log','w') as f:subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 return read_profile(str(output)+'.profile')

def coverage(root,hot,n,q,bestnodes,base):
 selected=set(map(int,hot[:int(n*.7)]));selected.add(root);_,best=nearest(q,np.asarray(base[np.array(sorted(selected))]));scale=np.maximum(np.sum(q*q,axis=1),1e-12);closed=np.zeros(len(q),bool)
 while len(selected)<n:
  priority=best/scale;priority[closed]=-1;at=int(np.argmax(priority))
  if priority[at]<0:break
  node=int(bestnodes[at]);closed[at]=True
  if node in selected:continue
  selected.add(node);best=np.minimum(best,dist_to(q,base[node]))
 for node in hot:
  if len(selected)==n:break
  selected.add(int(node))
 return ordered(root,selected)

def reach(table,degrees):
 seen={0};front=[0]
 while front:
  node=front.pop()
  for x in table[node,:degrees[node]]:
   if int(x) not in seen:seen.add(int(x));front.append(int(x))
 return len(seen)

manifest={}
for ds in ['sift','deep']:
 start=time.perf_counter();directory=ROOT/ds;directory.mkdir(exist_ok=True);meta=json.loads((ROOT/f'{ds}_bfs.json').read_text());root=meta['root_id'];n=9000 if ds=='sift' else 8850
 train_ids=np.load(LG/('experiments/sift1m/training_data_l100/train/query_ids.npy' if ds=='sift' else 'experiments/deep1m/queries/train/query_ids.npy'))
 base=matrix(LG/('data/sift/sift_base.fbin' if ds=='sift' else 'data/deep1M_base.fbin'));query=matrix(LG/('data/sift/sift_query.fbin' if ds=='sift' else 'data/deep1M_query.fbin'));q=np.asarray(query[train_ids]).copy()
 with open(directory/'history.fbin','wb') as f:np.array(q.shape,np.uint32).tofile(f);q.tofile(f)
 np.save(directory/'history_query_ids.npy',train_ids)
 print(ds,'extract history',flush=True);freq,bestnodes=history(ds,train_ids);hot=np.lexsort((np.arange(1000000),-freq));np.save(directory/'frequency.npy',freq);np.save(directory/'historical_best_nodes.npy',bestnodes)
 b2=ordered(root,hot[:n]);b3=coverage(root,hot,n,q,bestnodes,base);assert len(b2)==n and len(b3)==n
 cfgs={};disk=LG/'experiments'/('sift1m' if ds=='sift' else 'deep1m')/'index_disk.index'
 for name,selected in [('B2',b2),('B3',b3)]:
  edges,degree=original_edges(disk,selected,base.shape[1]);full,full_degree=original_edges(disk,selected,base.shape[1],64);packed=np.full(n*16,0xffffffff,np.uint32);flat=np.concatenate([full[i,:int(full_degree[i])] for i in range(n)]);assert len(flat)<=len(packed);packed[:len(flat)]=flat;offset=np.concatenate([[0],np.cumsum(full_degree[:-1])]).astype(np.uint32);write_graph(directory/name,selected,packed,offset,csr=True);cfgs[name]={'nodes':n,'edge_slots':n*16,'valid_edges':int(full_degree.sum()),'reachable_from_root':reach(full,full_degree),'truncated16_valid_edges':int(degree.sum()),'truncated16_reachable_from_root':reach(edges,degree),'runtime_edge_source':'CSR of all original resident-to-resident edges, no degree truncation'}
 print(ds,'build independent graph',flush=True);v=np.asarray(base[b3]);edges4=rebuilt(v);write_graph(directory/'B4',b3,edges4);cfgs['B4']={'nodes':n,'valid_edges':n*16,'reachable_from_root':reach(edges4,np.full(n,16))};assert cfgs['B4']['reachable_from_root']==n
 print(ds,'joint node update from historical RAM navigation',flush=True);paths=profile(ds,directory/'B4','joint_round0');owners,_=nearest(q,v);visits=np.zeros(n,np.int64)
 for path in paths.values():np.add.at(visits,path,1)
 owns=np.bincount(owners,minlength=n);score=freq[b3]/max(1,freq[b3].max())+visits/max(1,visits.max())+owns/max(1,owns.max())
 bestpath=np.array([np.min(np.sum((v[paths[i]]-q[i])**2,axis=1)) for i in range(len(q))]);desired=np.sum((np.asarray(base[bestnodes])-q)**2,axis=1);gap=np.maximum(0,bestpath-desired)/np.maximum(np.sum(q*q,axis=1),1e-12)
 remove=[int(i) for i in np.argsort(score,kind='stable') if int(b3[i])!=root];add=[];selected=set(map(int,b3));seen=set()
 for at in np.argsort(-gap,kind='stable'):
  node=int(bestnodes[at])
  if node not in selected and node not in seen:add.append(node);seen.add(node)
  if len(add)>=int(n*.05):break
 final=(selected-{int(b3[i]) for i in remove[:len(add)]})|set(add);b5=ordered(root,final);assert len(b5)==n
 v5=np.asarray(base[b5]);edges5=rebuilt(v5);write_graph(directory/'B5_geometry',b5,edges5)
 print(ds,'joint edge update from historical routing feedback',flush=True);paths=profile(ds,directory/'B5_geometry','joint_round1');owners,dmin=nearest(q,v5);pairs=collections.defaultdict(collections.Counter)
 for row,path in paths.items():
  target=int(owners[row]);dist=np.sum((v5[path]-q[row])**2,axis=1)
  for source,dd in zip(path,dist):
   source=int(source)
   if source!=target and dd>dmin[row]*1.05 and target not in edges5[source]:pairs[source][target]+=float((dd-dmin[row])/max(dd,1e-12))
 changed=0
 for source,targets in pairs.items():
  target=min(targets,key=lambda t:(-targets[t],t));edges5[source,13]=target;changed+=1
 write_graph(directory/'B5',b5,edges5);cfgs['B5']={'nodes':n,'valid_edges':n*16,'nodes_replaced':len(add),'edges_replaced':changed,'reachable_from_root':reach(edges5,np.full(n,16))};assert cfgs['B5']['reachable_from_root']==n
 manifest[ds]={'budget_bytes':meta['bfs_native_cache_bytes'],'history_queries':len(train_ids),'nodes_B2_B5':n,'navigation_degree':16,'coverage_hot_fraction':.7,'joint_swap_fraction':.05,'configurations':cfgs,'build_wall_seconds':time.perf_counter()-start,'shared_nodes_B3_B4':bool(np.array_equal(b3,ids(str(directory/'B4')+'.ids')))}
 (ROOT/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(ds,'done',manifest[ds],flush=True)
