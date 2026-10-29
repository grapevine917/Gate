from common import *
import shutil,shlex,subprocess,sys
N=R/'native';N.mkdir(exist_ok=True);D=LG/'DiskANN'
for name in ['online_gate.h','online_termination.h','cost_profile.h']:shutil.copy(E1/'native'/name,N/name)
shutil.copy(R/'override.h',N/'override.h')
src=(E1/'native/pq_flash_index.cpp').read_text().replace('#include "online_gate.h"','#include "online_gate.h"\n#include "override.h"')
needle='if(memo_candidate && online_skip_memo->audit && !online_skip_memo->lookup_only) {'
assert src.count(needle)==1
src=src.replace(needle,'e2_override_score(nbr.id,state,online_last_score);\n                '+needle)
(N/'pq_flash_index.cpp').write_text(src)
flags=(D/'build/src/CMakeFiles/diskann.dir/flags.make').read_text();opts=[]
for key in ['CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS']:opts+=shlex.split(next(l.split('=',1)[1] for l in flags.splitlines() if l.startswith(key+' =')))
def run(cmd):
 log('BUILD '+cmd[0])
 with (R/'build.log').open('a') as f:
  f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':cmd})+'\n');f.flush();subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
if '--link-only' not in sys.argv:
 run(['g++','-I'+str(N),*opts,'-c',str(N/'pq_flash_index.cpp'),'-o',str(N/'pq_flash_index.cpp.o')])
 shutil.copy(LG/'DiskANN/build/src/libdiskann.a',N/'libdiskann.a');run(['ar','r',str(N/'libdiskann.a'),str(N/'pq_flash_index.cpp.o')])
link=shlex.split((D/'build/apps/CMakeFiles/collect_io_labels.dir/link.txt').read_text());libs=link[link.index('-L/opt/intel/oneapi/mkl/latest/lib/intel64'):];libs=[str(N/'libdiskann.a') if s=='../src/libdiskann.a' else s for s in libs]
run(['g++','-I'+str(N),*opts,str(R/'intervene.cpp'),'-o',str(N/'intervene'),*libs,'-latomic']);log('BUILD COMPLETE')
