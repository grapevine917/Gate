from pathlib import Path
import subprocess,shlex,shutil
R=Path(__file__).resolve().parent; D=R.parents[1]/'DiskANN';N=R/'native';N.mkdir(exist_ok=True)
s=(D/'src/pq_flash_index.cpp').read_text()
s=s.replace('#include "online_gate.h"','#include "online_gate.h"\n#include "beam_policy.h"')
a='        uint32_t num_seen = 0;'
b='''        unsigned beam_skips=0;
        uint32_t beam_protect=std::numeric_limits<uint32_t>::max();
        std::vector<float> beam_scores;
        if(beam_policy==3 && online_gate) {
            auto preview=retset;
            float highest=-1; bool cached=false;
            for(unsigned pos=0;pos<beam_width && preview.has_unexpanded_node();++pos) {
                size_t rank=0,remaining=0;bool found=false;
                for(size_t j=0;j<preview.size();++j)if(!preview[j].expanded){++remaining;if(!found){rank=j;found=true;}}
                auto n=preview.closest_unexpanded();
                float best=preview[0].distance,worst=preview[preview.size()-1].distance;
                bool hit=_nhood_cache.count(n.id)!=0,exact=completed_expansions>0;
                float st[17]={n.distance,best,worst,float(rank),float(preview.size()),float(remaining),float(visited.size()),float(completed_expansions),float(hops),float(pos),float(hit),exact?online_best_exact:0.0f,float(exact),float(double(n.distance)/std::max(double(best),1e-12)),float(double(n.distance)/std::max(double(worst),1e-12)),float(l_search),float(beam_width)};
                float score=hit?1.0f:online_gate->score(query_float,n.id,st);
                beam_scores.push_back(score);cached|=hit;
                if(score>highest){highest=score;beam_protect=n.id;}
            }
            // Cached candidates are already guaranteed Keep.
            if(cached)beam_protect=std::numeric_limits<uint32_t>::max();
        }
        uint32_t num_seen = 0;'''
assert s.count(a)==1;s=s.replace(a,b)
a='online_last_score=(online_miss_only && state[10]!=0)?1.0f:(online_gate?online_gate->score(query_float,nbr.id,state):float(!heuristic_skip));'
assert s.count(a)==1;s=s.replace(a,'online_last_score=beam_policy==3?beam_scores.at(num_seen-1):((online_miss_only && state[10]!=0)?1.0f:(online_gate?online_gate->score(query_float,nbr.id,state):float(!heuristic_skip)));')
a='&& !(online_miss_only && _nhood_cache.count(nbr.id));'
assert s.count(a)==2;s=s.replace(a,a[:-1]+' && !((beam_policy==1 && beam_skips>=1) || (beam_policy==2 && beam_skips>=2) || (beam_policy==3 && nbr.id==beam_protect));')
s=s.replace('if(history_skip) {','if(history_skip) {\n                    ++beam_skips;')
(N/'pq_flash_index.cpp').write_text(s)
(N/'beam_policy.h').write_text('#pragma once\nnamespace diskann { inline thread_local int beam_policy=0; }\n')
for h in ['online_gate.h','online_termination.h']:shutil.copy(D/'include'/h,N/h)
c=(R.parent/'e1_similarity/collect.cpp').read_text().replace('#include "cost_profile.h"','#include "cost_profile.h"\n#include "beam_policy.h"').replace('diskann::online_gate=model.get();','diskann::beam_policy=std::stoi(argv[6]);diskann::online_gate=model.get();')
c=c.replace('if(std::string(argv[6])!="-")marks.reset(new diskann::OnlineNodeMarks(argv[6]));','')
(R/'collect.cpp').write_text(c)
f=(D/'build/src/CMakeFiles/diskann.dir/flags.make').read_text();opts=[]
for key in ['CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS']:opts+=shlex.split(next(l.split('=',1)[1] for l in f.splitlines() if l.startswith(key+' =')))
def run(a):
 with (R/'build.log').open('a') as out:subprocess.run(a,stdout=out,stderr=subprocess.STDOUT,check=True)
run(['g++','-I'+str(N),*opts,'-c',str(N/'pq_flash_index.cpp'),'-o',str(N/'pq_flash_index.cpp.o')])
shutil.copy(D/'build/src/libdiskann.a',N/'libdiskann.a');run(['ar','r',str(N/'libdiskann.a'),str(N/'pq_flash_index.cpp.o')])
l=shlex.split((D/'build/apps/CMakeFiles/collect_io_labels.dir/link.txt').read_text());libs=l[l.index('-L/opt/intel/oneapi/mkl/latest/lib/intel64'):];libs=[str(N/'libdiskann.a') if v=='../src/libdiskann.a' else v for v in libs]
run(['g++','-I'+str(N),*opts,str(R/'collect.cpp'),'-o',str(N/'collect'),*libs,'-latomic'])
print('build complete')
