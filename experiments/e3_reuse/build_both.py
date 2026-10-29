"""Independent extension: both Keep and Skip reuse; preserve original Skip-only runs."""
from common import *
import shutil,shlex
N=R/'both_native';N.mkdir(exist_ok=True);D=LG/'DiskANN'
h=(R/'reuse.h').read_text()
needle='\n};\ninline thread_local E3Context*'
assert h.count(needle)==1
extra='''
 uint32_t decision(uint32_t node,const float*s,float fresh) const {
  if(kind<10)return candidate(node,s)?1:0;
  auto it=bank->ranges.find(node);if(it==bank->ranges.end())return 0;
  if(kind==11){bool skip=fresh<tau;for(size_t i=it->second.first;i<it->second.second;++i)if(bool(bank->records[i].skip)==skip)return skip?1:2;return 0;}
  int base=kind-10,available=0,skips=0;double best=std::numeric_limits<double>::infinity();bool winner=false;bool found=false;double z[17];
  for(int k=0;k<17;++k)z[k]=(double(s[k])-bank->mean[k])/bank->denom[k];
  for(size_t i=it->second.first;i<it->second.second;++i){const auto&r=bank->records[i];int rank=-1;
   for(int k=0;k<3;++k)if(r.query==neighbours[k]&&query_dist[k]<=qcut){rank=k;break;}
   if(base==5){if(rank>=0){++available;skips+=r.skip;}continue;}
   if(base==4&&rank<0)continue;double d=0;for(int k=0;k<17;++k){double v=z[k]-r.z[k];d+=v*v;}
   if(d<best){best=d;winner=r.skip;found=true;}
  }
  if(base==5){if(available<votes)return 0;if(skips==available)return 1;if(skips==0)return 2;return 0;}
  return found&&best<=scut*scut?(winner?1:2):0;
 }
'''
h=h.replace(needle,extra+needle)
h=h.replace('c->false_skip+=r.fresh>=c->tau;score=0;','c->false_skip+=(r.reused==1&&r.fresh>=c->tau);score=r.reused==1?0:1;')
h=h.replace('bool reuse=c->candidate(node,s)&&(!c->oracle||score<c->tau);','uint32_t reuse=c->decision(node,s,score);if(c->oracle&&reuse&&((reuse==1)!=(score<c->tau)))reuse=0;')
h=h.replace('c->false_skip+=score>=c->tau;score=0;','c->false_skip+=(reuse==1&&score>=c->tau);score=reuse==1?0:1;')
(N/'reuse.h').write_text(h)
for name in ['online_gate.h','online_termination.h','cost_profile.h']:shutil.copy(E1/'native'/name,N/name)
src=(R/'native/pq_flash_index.cpp').read_text();(N/'pq_flash_index.cpp').write_text(src)
flags=(D/'build/src/CMakeFiles/diskann.dir/flags.make').read_text();opts=[]
for key in ['CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS']:opts+=shlex.split(next(l.split('=',1)[1] for l in flags.splitlines() if l.startswith(key+' =')))
def build(cmd):
 log('BUILD BOTH '+cmd[0])
 with (R/'both_build.log').open('a') as f:f.write(json.dumps(list(map(str,cmd)))+'\n');f.flush();subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,check=True)
build(['g++','-I'+str(N),*opts,'-c',N/'pq_flash_index.cpp','-o',N/'pq_flash_index.cpp.o']);shutil.copy(D/'build/src/libdiskann.a',N/'libdiskann.a');build(['ar','r',N/'libdiskann.a',N/'pq_flash_index.cpp.o'])
link=shlex.split((D/'build/apps/CMakeFiles/collect_io_labels.dir/link.txt').read_text());libs=link[link.index('-L/opt/intel/oneapi/mkl/latest/lib/intel64'):];libs=[str(N/'libdiskann.a') if v=='../src/libdiskann.a' else v for v in libs]
build(['g++','-I'+str(N),*opts,R/'search.cpp','-o',N/'search',*libs,'-latomic'])
p=json.loads((R/'PROTOCOL.json').read_text());qcut=p['query_cutoffs_train_only'][-1];configs=[{'key':'D0','kind':0,'qcut':1e30,'scut':1e30,'votes':2,'oracle':0}]
for kind in [13,14]:
 for i,s in enumerate(p['state_cutoffs_train_only'][:-1]):configs.append({'key':f'D{kind-10}_s{i}','kind':kind,'qcut':qcut,'scut':s,'votes':2,'oracle':0})
configs += [{'key':'D5','kind':15,'qcut':qcut,'scut':1e30,'votes':2,'oracle':0},{'key':'OracleA_both','kind':11,'qcut':1e30,'scut':1e30,'votes':2,'oracle':1}]
dump(R/'both_configs.json',configs);dump(R/'BOTH_PROTOCOL.json',{'scope':'separate extension permitting reuse of historical Keep as well as Skip, to avoid understating inference-saving potential','base':str(R/'PROTOCOL.json'),'validation_grid':configs,'test_used_to_choose_grid':False,'selection':'same 0.2pp validation budget and maximize ModelSaving; no test retuning','OracleA_both':'free fresh decision, reuse if any same-node history decision agrees, including Keep; unrestricted state/query distance within historical node coverage','OracleB_both':'quality-constrained portfolio over previously replayed Skip-only trajectories and validation-selected both-decision trajectories; exact within menu, not unrestricted combinatorial optimum','latency':'same offline-tape protocol; Keep reuse still reads/expands node; no inference call at reused Keep or Skip','FalseSkipRate':'only reused Skip where fresh Keep / all fresh Keep; Keep disagreement separately counted from tapes'})
log('BOTH BUILD AND GRID LOCKED')
