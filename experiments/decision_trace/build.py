from pathlib import Path
import subprocess,shlex,json,hashlib
root=Path(__file__).resolve().parents[2];app=root/'DiskANN/build/apps'; out=root/'experiments/decision_trace_acceptance';out.mkdir(exist_ok=True)
flags={}
for line in (app/'CMakeFiles/search_disk_index.dir/flags.make').read_text().splitlines():
 if line.startswith('CXX_'):
  k,v=line.split('=',1);flags[k.strip()]=shlex.split(v)
link=shlex.split((app/'CMakeFiles/search_disk_index.dir/link.txt').read_text())
link=[x for x in link if not x.startswith('-Wl,--dependency-file=')]
for name,src,dst in [('control',out/'reference/search_disk_index.cpp',out/'control'),('candidate',root/'DiskANN/apps/search_disk_index.cpp',app/'search_disk_index')]:
 obj=out/(name+'.o')
 command=['g++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-c',str(src),'-o',str(obj)]
 subprocess.run(command,cwd=app,check=True)
 command=[str(obj) if x.endswith('search_disk_index.cpp.o') else x for x in link];command[command.index('-o')+1]=str(dst);command+=['-latomic']
 subprocess.run(command,cwd=app,check=True)
 print(name,hashlib.sha256(dst.read_bytes()).hexdigest(),flush=True)
