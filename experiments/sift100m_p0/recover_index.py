"""Resume saved DiskANN shards using bounded I/O buffers; keep original shards."""
from pathlib import Path
import datetime, fcntl, hashlib, json, os, struct, subprocess, sys, traceback
R=Path(__file__).resolve().parent
ROOT=R.parents[1]
os.chdir(ROOT)
lock=(R/'index_recovery.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def dump(name,info):
 p=R/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(info,indent=2)+'\n');t.replace(p)
def status(phase,**kw):
 dump('STATUS.json',dict(phase=phase,complete=False,pid=os.getpid(),updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
try:
 if not (R/'INDEX_COMPLETE.json').exists():
  status('index_recovery_precheck')
  for i in range(17):
   p=R/f'index/ann_mem.index_tempFiles_subshard-{i}_mem.index'
   with p.open('rb') as f:expected,degree,medoid,frozen=struct.unpack('<QIIQ',f.read(24))
   assert expected==p.stat().st_size and degree<=64 and frozen==0,(p,expected,p.stat().st_size)
   ids=p.with_name(p.name.removesuffix('_mem.index')+'_ids_uint32.bin')
   with ids.open('rb') as f:n,d=struct.unpack('<II',f.read(8))
   assert d==1 and ids.stat().st_size==8+4*n
  src=(ROOT/'DiskANN/src/disk_utils.cpp').read_text()
  start=src.index('int merge_shards(');end=src.index('\n// TODO:',start)
  merge=src[start:end].replace('int merge_shards(', 'int recovery_merge_shards(').replace('BUFFER_SIZE_FOR_CACHED_IO','(16ULL * 1024 * 1024)')
  code='#include "logger.h"\n#include "disk_utils.h"\n#include "partition.h"\nnamespace diskann {\n'+merge+'\n}\n'
  code+='''int main(int argc,char**argv) {
 if(argc!=3)return 2;
 std::string p=argv[1], base=argv[2];
 diskann::recovery_merge_shards(p+"_mem.index_tempFiles_subshard-","_mem.index",p+"_mem.index_tempFiles_subshard-","_ids_uint32.bin",17,64,p+"_mem.index",p+"_disk.index_medoids.bin",false,"");
 diskann::create_disk_layout<uint8_t>(base,p+"_mem.index",p+"_disk.index");
 gen_random_slice<uint8_t>(base,p+"_sample",0.001);
 return 0;
}\n'''
  cpp=R/'native/recover_index.cpp';cpp.write_text(code)
  template=json.loads((R/'build.commands.json').read_text())[-1]
  si=template.index(str(R/'score.cpp'));oi=template.index('-L/opt/intel/oneapi/mkl/latest/lib/intel64')
  libs=[x.replace(str(R/'native/libdiskann.a'),str(ROOT/'DiskANN/build/src/libdiskann.a')) for x in template[oi:]]
  cmd=[*template[:si],str(cpp),'-o',str(R/'native/recover_index'),*libs]
  dump('index_recovery.command.json',dict(argv=cmd,source_sha256=hashlib.sha256(cpp.read_bytes()).hexdigest(),cache_bytes_per_stream=16777216,shards_preserved=True,original_merge_algorithm=True))
  status('index_recovery_compiling')
  subprocess.run(cmd,check=True)
  status('index_merge_and_disk_layout',current_log='index_recovery.log')
  subprocess.run([str(R/'native/recover_index'),str(R/'index/ann'),str(ROOT/'data/sift100m/base.100M.u8bin')],check=True)
  p=R/'index/ann_disk.index'
  with p.open('rb') as f:
   rows,cols=struct.unpack('<II',f.read(8));meta=struct.unpack('<'+'Q'*(min(rows*cols,16)),f.read(8*min(rows*cols,16)))
  assert meta[0]==100000000 and meta[1]==128 and p.stat().st_size%4096==0,(rows,cols,meta,p.stat().st_size)
  dump('INDEX_COMPLETE.json',dict(passed=True,complete=True,recovered_from_shards=True,shards=17,cache_bytes_per_stream=16777216,disk_index_bytes=p.stat().st_size,metadata=list(meta),completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 status('index_complete_starting_pipeline')
 subprocess.run([sys.executable,str(R/'run_pipeline.py')],check=True)
except BaseException as e:
 status('failed',error=str(e),traceback=traceback.format_exc())
 raise
