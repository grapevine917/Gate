from common import *
def snapshot():
 out={}
 for ds in CATALOG:
  prefix=catalog(ds)['index_path'];files={}
  for p in sorted(prefix.parent.glob(prefix.name+'*')):
   if not p.is_file():continue
   st=p.stat()
   with p.open('rb') as f:head=f.read(4096)
   files[str(p)]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,header4096_sha256=hashlib.sha256(head).hexdigest())
  assert files;out[ds]=files
 return out
if __name__=='__main__':
 current=snapshot();p=D/'INDEX_LOCK.json'
 if p.exists():assert json.loads(p.read_text())==current
 else:dump(p,current)
 print('INDEX METADATA/HEADER LOCK VERIFIED',flush=True)
