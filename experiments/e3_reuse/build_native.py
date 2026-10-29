from common import *
import shutil,shlex,sys
N=R/'native';N.mkdir(exist_ok=True);D=LG/'DiskANN'
for name in ['online_gate.h','online_termination.h','cost_profile.h']:shutil.copy(E1/'native'/name,N/name)
shutil.copy(R/'reuse.h',N/'reuse.h')
src=(E1/'native/pq_flash_index.cpp').read_text().replace('#include "online_gate.h"','#include "online_gate.h"\n#include "reuse.h"')
needle='''if(online_model_calls && online_gate && state[10]==0)++*online_model_calls;
                { E1Scope timer(1); online_last_score=(online_miss_only && state[10]!=0)?1.0f:(online_gate?online_gate->score(query_float,nbr.id,state):float(!heuristic_skip)); }'''
assert src.count(needle)==1
src=src.replace(needle,'''bool e3_reused=e3_before(nbr.id,state,online_last_score);
                if(!e3_reused){
                 if(online_model_calls && online_gate && state[10]==0)++*online_model_calls;
                 { E1Scope timer(1); online_last_score=(online_miss_only && state[10]!=0)?1.0f:(online_gate?online_gate->score(query_float,nbr.id,state):float(!heuristic_skip)); }
                }
                e3_after(nbr.id,state,online_last_score);''')
(N/'pq_flash_index.cpp').write_text(src)
flags=(D/'build/src/CMakeFiles/diskann.dir/flags.make').read_text();opts=[]
for key in ['CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS']:opts+=shlex.split(next(l.split('=',1)[1] for l in flags.splitlines() if l.startswith(key+' =')))
def build(cmd):
 log('BUILD '+cmd[0])
 with (R/'build.log').open('a') as f:f.write(json.dumps(list(map(str,cmd)))+'\n');f.flush();subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,check=True)
if '--link-only' not in sys.argv:
 build(['g++','-I'+str(N),*opts,'-c',N/'pq_flash_index.cpp','-o',N/'pq_flash_index.cpp.o']);shutil.copy(D/'build/src/libdiskann.a',N/'libdiskann.a');build(['ar','r',N/'libdiskann.a',N/'pq_flash_index.cpp.o'])
link=shlex.split((D/'build/apps/CMakeFiles/collect_io_labels.dir/link.txt').read_text());libs=link[link.index('-L/opt/intel/oneapi/mkl/latest/lib/intel64'):];libs=[str(N/'libdiskann.a') if v=='../src/libdiskann.a' else v for v in libs]
build(['g++','-I'+str(N),*opts,R/'search.cpp','-o',N/'search',*libs,'-latomic']);log('BUILD COMPLETE')
