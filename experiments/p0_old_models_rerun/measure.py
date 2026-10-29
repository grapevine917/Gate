from common import *
class Experiment:
 def __init__(self,ds):
  self.ds=ds;self.C=catalog(ds);self.O=self.C['out'];self.R=self.C['parent_dir'];self.P=json.loads((self.O/'PROTOCOL.json').read_text());self.configs=json.loads((self.O/'configs.json').read_text());self.bykey={c['key']:c for c in self.configs};self.cache={}
  for p in [self.O/'logs',self.O/'runs/validation',self.O/'runs/test']:p.mkdir(parents=True,exist_ok=True)
  self.gt={}
  for split in ['validation','test']:
   with (self.R/'data'/f'{split}.gt').open('rb') as f:n,k=np.fromfile(f,'<u4',2);self.gt[split]=np.fromfile(f,'<u4').reshape(int(n),int(k))[:,:10]
 def prefix(self,split,c,rep):return self.O/'runs'/split/f"{c['key']}_r{rep}"
 def complete(self,split,c,rep):
  p=self.prefix(split,c,rep)
  try:
   m=json.loads(p.with_suffix('.meta.json').read_text());return m['queries']==1000 and m['reader_bytes']==m['linux_read_bytes']==m['reader_requests']*4096 and m['swap_before_kib']==m['swap_after_kib']==0 and p.with_suffix('.ids').stat().st_size==80000 and len(p.with_suffix('.csv').read_text().splitlines())==1001
  except (OSError,ValueError,KeyError):return False
 def read(self,split,c,rep):
  key=(split,c['key'],rep)
  if key in self.cache:return self.cache[key]
  p=self.prefix(split,c,rep);m=json.loads(p.with_suffix('.meta.json').read_text());a=np.genfromtxt(p.with_suffix('.csv'),delimiter=',',names=True);ids=np.fromfile(p.with_suffix('.ids'),'<u8').reshape(1000,10)
  valid=ids<self.C['N'];missing=ids==np.iinfo(np.uint32).max;assert np.all(valid|missing) and np.all(np.diff(missing.astype('i1'),axis=1)>=0)
  assert all(len(set(row[mask]))==int(mask.sum()) for row,mask in zip(ids,valid))
  hits=np.array([len(set(row)&set(gt)) for row,gt in zip(ids,self.gt[split])]);np.testing.assert_array_equal(hits,a['hits'])
  assert m['linux_read_bytes']==m['reader_bytes']==m['reader_requests']*4096==int(a['reads'].sum())*4096 and m['swap_before_kib']==m['swap_after_kib']==0
  if c['kind'] in [1,2,4]:np.testing.assert_array_equal(a['expansions'],a['gate_evaluated']-a['skips'])
  r=c|dict(split=split,repeat=rep,total_hits=int(hits.sum()),recall=float(hits.sum()/100),p50_ms=float(np.percentile(a['latency_us'],50)/1000),p95_ms=float(np.percentile(a['latency_us'],95)/1000),p99_ms=float(np.percentile(a['latency_us'],99)/1000),mean_ms=float(a['latency_us'].mean()/1000),qps=float(m['qps']),physical_bytes=float(m['linux_read_bytes']/1000),reads=float(a['reads'].mean()),expansions=float(a['expansions'].mean()),skips=float(a['skips'].mean()),model_calls=float(a['model_calls'].mean()),et_calls=float(a['et_evaluated'].mean()),et_stop_fraction=float(a['et_stopped'].mean()),io_ms=float(a['io_us'].mean()/1000),io_rounds=float(a['io_rounds'].mean()),missing_result_slots=int(missing.sum()),short_result_queries=int(missing.any(axis=1).sum()),rss_kib=m['rss_after_kib'],peak_rss_kib=m['peak_rss_kib'],ids_sha256=sha(p.with_suffix('.ids')))
  self.cache[key]=r;return r
 def load(self,split,configs,rep=0,ordered=False):
  todo=[c for c in configs if not self.complete(split,c,rep)]
  if not ordered:random.Random(SEED+rep+(split=='test')*100).shuffle(todo)
  batches=[todo[i:i+16] for i in range(0,len(todo),16)]
  for bi,batch in enumerate(batches):
   digest=hashlib.sha256(' '.join(c['key'] for c in batch).encode()).hexdigest()[:12];tag=f'{split}_r{rep}_{digest}';configfile=self.O/'logs'/f'{tag}.configs'
   configfile.write_text(''.join(f"{c['key']} {c['kind']} {c['model']} {c['L']} {c['tau']:.17g} {c['et_tau']:.17g} {c['rank']} {c['exp']}\n" for c in batch))
   cmd=[self.O/'native/search',self.C['index_path'],self.R/'data'/f"{split}.{self.C['query_ext']}",self.R/'data'/f'{split}.gt',self.R/'models',self.C['vector_path'],configfile,self.O/'runs'/split,str(rep)]
   status=dict(dataset=self.ds,phase=split,repeat=rep,batch=bi+1,batches=len(batches),keys=[c['key'] for c in batch],log=str(self.O/'logs'/f'{tag}.log'),updated=time.strftime('%Y-%m-%d %H:%M:%S'))
   dump(self.O/'STATUS.json',status);dump(D/'STATUS.json',status);print(time.strftime('%H:%M:%S'),self.ds,split,rep,'batch',bi+1,'/',len(batches),flush=True)
   with (self.O/'logs'/f'{tag}.log').open('w') as f:subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,check=True)
   for c in batch:
    p=self.prefix(split,c,rep);m=json.loads(p.with_suffix('.meta.json').read_text())
    if m['linux_read_bytes']!=m['reader_bytes'] or m['reader_requests']*4096!=m['reader_bytes']:
     journal=json.loads((self.O/'IO_REJECTIONS.json').read_text()) if (self.O/'IO_REJECTIONS.json').exists() else [];attempt=1+sum(r['split']==split and r['key']==c['key'] and r['repeat']==rep for r in journal);assert attempt<=3,('repeatedIO mismatch',c['key'],m)
     dest=self.O/'rejected_io'/f"{split}_{c['key']}_r{rep}_attempt{attempt}";dest.mkdir(parents=True,exist_ok=True)
     for ext in ['csv','ids','meta.json']:shutil.move(str(p.with_suffix('.'+ext)),str(dest/p.with_suffix('.'+ext).name))
     journal.append(dict(split=split,key=c['key'],repeat=rep,attempt=attempt,metadata=m,archive=str(dest)));dump(self.O/'IO_REJECTIONS.json',journal);self.load(split,[c],rep,True)
  return [self.read(split,c,rep) for c in configs]
 def audit_sources(self):assert all(sha(p)==h for p,h in self.P['source_hashes'].items())
