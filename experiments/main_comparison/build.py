from pathlib import Path
import subprocess,shlex,shutil,json,sys
R=Path(__file__).resolve().parent;D=R.parents[1]/'DiskANN';N=R/'native';N.mkdir(exist_ok=True);E=R.parent/'e1_similarity/native'
for h in ['online_gate.h','online_termination.h']:shutil.copy(E/h,N/h)
shutil.copy(D/'src/pq_flash_index.cpp',N/'pq_flash_index.cpp')
f=(D/'build/src/CMakeFiles/diskann.dir/flags.make').read_text();opts=[]
for k in ['CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS']:opts+=shlex.split(next(x.split('=',1)[1] for x in f.splitlines() if x.startswith(k+' =')))
cmds=[]
def run(c):
 cmds.append(c);(R/'build.commands.json').write_text(json.dumps(cmds,indent=2))
 with (R/'build.log').open('a') as f:subprocess.run(c,stdout=f,stderr=subprocess.STDOUT,check=True)
if '--link-only' not in sys.argv:
 run(['g++','-I'+str(N),*opts,'-c',str(N/'pq_flash_index.cpp'),'-o',str(N/'pq_flash_index.cpp.o')]);shutil.copy(D/'build/src/libdiskann.a',N/'libdiskann.a');run(['ar','r',str(N/'libdiskann.a'),str(N/'pq_flash_index.cpp.o')])
l=shlex.split((D/'build/apps/CMakeFiles/collect_io_labels.dir/link.txt').read_text());libs=l[l.index('-L/opt/intel/oneapi/mkl/latest/lib/intel64'):];libs=[str(N/'libdiskann.a') if x=='../src/libdiskann.a' else x for x in libs]
for source,name in [('collect.cpp','collect'),('search.cpp','search'),('verify.cpp','verify')]:
 if (R/source).exists():run(['g++','-I'+str(N),*opts,str(R/source),'-o',str(N/name),*libs,'-latomic'])
print('BUILD COMPLETE')
