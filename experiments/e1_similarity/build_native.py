"""Build isolated instrumented library; never change existing DiskANN sources/binaries."""
from pathlib import Path
import subprocess,shlex,shutil,json,sys
R=Path(__file__).resolve().parent;D=R.parents[1]/'DiskANN';N=R/'native';N.mkdir(exist_ok=True)
src=(D/'src/pq_flash_index.cpp').read_text()
src=src.replace('#include "online_gate.h"','#include "online_gate.h"\n#include "cost_profile.h"')
def replace(a,b):
 global src
 assert a in src,a
 src=src.replace(a,b)
replace('diskann::aggregate_coords(ids, n_ids, this->data, this->_n_chunks, pq_coord_scratch);','{ E1Scope timer(2); diskann::aggregate_coords(ids, n_ids, this->data, this->_n_chunks, pq_coord_scratch); }')
replace('diskann::pq_dist_lookup(pq_coord_scratch, n_ids, this->_n_chunks, pq_dists, dists_out);','{ E1Scope timer(0); diskann::pq_dist_lookup(pq_coord_scratch, n_ids, this->_n_chunks, pq_dists, dists_out); }')
replace('_pq_table.populate_chunk_distances(query_rotated, pq_dists);','{ E1Scope timer(0); _pq_table.populate_chunk_distances(query_rotated, pq_dists); }')
replace('reader->read(frontier_read_reqs, ctx);','{ E1Scope timer(3); reader->read(frontier_read_reqs, ctx); }')
replace('memcpy(data_buf, node_fp_coords, _disk_bytes_per_point);','{ E1Scope timer(2); memcpy(data_buf, node_fp_coords, _disk_bytes_per_point); }')
for coords in ['node_fp_coords_copy','data_buf']:
 a=f'cur_expanded_dist = _dist_cmp->compare(aligned_query_T, {coords}, (uint32_t)_aligned_dim);'
 replace(a,'{ E1Scope timer(0); '+a+' }')
replace('online_last_score=(online_miss_only && state[10]!=0)?1.0f:(online_gate?online_gate->score(query_float,nbr.id,state):float(!heuristic_skip));','{ E1Scope timer(1); online_last_score=(online_miss_only && state[10]!=0)?1.0f:(online_gate?online_gate->score(query_float,nbr.id,state):float(!heuristic_skip)); }')
(N/'pq_flash_index.cpp').write_text(src)
h=(D/'include/online_gate.h').read_text()
h=h.replace('float threshold=0;', '''bool forest_model=false;
    struct ForestNode { int left,right,feature; float value; };
    std::vector<std::vector<ForestNode>> forest;
    float forest_bias=0;
    float threshold=0;''')
h=h.replace('if(model_magic=="LRGATE01"', '''if(model_magic=="XGGATE01") {
            forest_model=true;state_only=true;uint32_t nt;
            f.read(reinterpret_cast<char*>(&forest_bias),4);f.read(reinterpret_cast<char*>(&nt),4);
            if(nt>10000)throw std::runtime_error("Invalid forest");
            forest.resize(nt);for(auto& t:forest){uint32_t nn;f.read(reinterpret_cast<char*>(&nn),4);if(nn>100000)throw std::runtime_error("Invalid tree");t.resize(nn);f.read(reinterpret_cast<char*>(t.data()),nn*sizeof(ForestNode));}
            if(!f)throw std::runtime_error("Truncated forest");return;
        }
        if(model_magic=="LRGATE01"''',1)
h=h.replace('if(linear_model || tree_model) {','''if(forest_model) {
            if(state[10]!=0)return 1;
            float z=forest_bias;
            for(const auto& t:forest){int i=0;while(t[i].left>=0)i=state[t[i].feature]<t[i].value?t[i].left:t[i].right;z+=t[i].value;}
            return 1.0f/(1.0f+std::exp(-z));
        }
        if(linear_model || tree_model) {''')
(N/'online_gate.h').write_text(h);shutil.copy(R/'cost_profile.h',N/'cost_profile.h')
shutil.copy(D/'include/online_termination.h',N/'online_termination.h')
flags=(D/'build/src/CMakeFiles/diskann.dir/flags.make').read_text()
opts=[]
for key in ['CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS']:
 opts+=shlex.split(next(l.split('=',1)[1] for l in flags.splitlines() if l.startswith(key+' =')))
def run(args):
 with (R/'build.log').open('a') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
if '--link-only' not in sys.argv:
 run(['g++','-I'+str(N),*opts,'-c',str(N/'pq_flash_index.cpp'),'-o',str(N/'pq_flash_index.cpp.o')])
 shutil.copy(D/'build/src/libdiskann.a',N/'libdiskann.a')
 run(['ar','r',str(N/'libdiskann.a'),str(N/'pq_flash_index.cpp.o')])
link=shlex.split((D/'build/apps/CMakeFiles/collect_io_labels.dir/link.txt').read_text())
libs=link[link.index('-L/opt/intel/oneapi/mkl/latest/lib/intel64'):]
libs=[str(N/'libdiskann.a') if v=='../src/libdiskann.a' else v for v in libs]
run(['g++','-I'+str(N),*opts,str(R/'collect.cpp'),'-o',str(N/'collect'),*libs,'-latomic'])
print('Built isolated E1 collector',flush=True)
