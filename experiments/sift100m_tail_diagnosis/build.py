"""Build an isolated profiler; never modify the frozen P0 artifacts."""
from pathlib import Path
import json, shutil, subprocess

D = Path(__file__).resolve().parent
R = D.parent / 'sift100m_p0'
N = D / 'native'
N.mkdir(exist_ok=True)
for name in ['online_gate.h', 'online_termination.h', 'libdiskann.a']:
    shutil.copy2(R / 'native' / name, N / name)
profile = r'''
#include <chrono>
namespace diskann {
struct TailProfile {
 bool timers=false;
 double rank_us=0, feature_us=0, score_us=0;
 uint64_t rounds=0, empty_rounds=0, queue_slots=0;
 std::vector<uint8_t>* record=nullptr;
 const std::vector<uint8_t>* replay=nullptr;
 size_t replay_pos=0;
 std::vector<uint32_t> nodes;
 std::vector<uint8_t> skipped;
};
inline thread_local TailProfile* tail_profile=nullptr;
inline double tail_now() { return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}
'''
with (N/'online_gate.h').open('a') as f: f.write(profile)
s=(R/'native/pq_flash_index.cpp').read_text()
def replace(old,new):
    global s
    assert s.count(old)==1, (old,s.count(old))
    s=s.replace(old,new)
replace('        // clear iteration state', '''        if(tail_profile)++tail_profile->rounds;
        // clear iteration state''')
replace('            // State is captured before marking this beam candidate expanded.', '''            double tail_rank_start=(tail_profile && tail_profile->timers)?tail_now():0;
            // State is captured before marking this beam candidate expanded.''')
replace('            Neighbor nbr;','''            if(tail_profile && tail_profile->timers)tail_profile->rank_us+=tail_now()-tail_rank_start;
            Neighbor nbr;''')
replace('            num_seen++;','''            num_seen++;
            if(tail_profile) {
                ++tail_profile->queue_slots;
                if(tail_profile->timers){tail_profile->nodes.push_back(nbr.id);tail_profile->skipped.push_back(0);}
                if(tail_profile->replay) {
                    if(tail_profile->replay_pos>=tail_profile->replay->size())throw std::runtime_error("replay exhausted");
                    bool skip=(*tail_profile->replay)[tail_profile->replay_pos++]!=0;
                    if(stats){++stats->n_gate_evaluated;if(skip)++stats->n_gate_skipped;}
                    if(skip)continue;
                }
            }''')
replace('                const float best=retset[0].distance, worst=retset[retset.size()-1].distance;', '''                double tail_feature_start=(tail_profile && tail_profile->timers)?tail_now():0;
                const float best=retset[0].distance, worst=retset[retset.size()-1].distance;''')
replace('                online_last_score=(online_miss_only', '''                if(tail_profile && tail_profile->timers)tail_profile->feature_us+=tail_now()-tail_feature_start;
                double tail_score_start=(tail_profile && tail_profile->timers)?tail_now():0;
                online_last_score=(online_miss_only''')
replace('                if(memo_candidate && online_skip_memo->audit && !online_skip_memo->lookup_only) {','''                if(tail_profile && tail_profile->timers)tail_profile->score_us+=tail_now()-tail_score_start;
                if(memo_candidate && online_skip_memo->audit && !online_skip_memo->lookup_only) {''')
replace('                if(history_skip) {','''                if(tail_profile && tail_profile->timers)tail_profile->skipped.back()=history_skip;
                if(tail_profile && tail_profile->record)tail_profile->record->push_back(history_skip);
                if(history_skip) {''')
replace('        // read nhoods of frontier ids','''        if(tail_profile && frontier.empty())++tail_profile->empty_rounds;
        // read nhoods of frontier ids''')
(N/'pq_flash_index.cpp').write_text(s)
s=(R/'search.cpp').read_text()
old='  auto search=[&](size_t i,diskann::QueryStats&s,uint64_t*ids,float*ds,uint32_t*calls){'
assert old in s
s=s.replace(old,'''  std::vector<std::vector<uint8_t>> tapes(nq);
  std::vector<std::string> modes={"profile","full","replay"};
  if(std::stoi(argv[8])%2)modes={"profile","replay","full"};
  for(const auto& mode:modes) {
  if(kind==0 && mode=="replay")continue;
  std::string mode_prefix=prefix+"_"+mode;
  std::vector<diskann::TailProfile> profiles(nq);
  bool formal=false;
  auto search=[&](size_t i,diskann::QueryStats&s,uint64_t*ids,float*ds,uint32_t*calls){''')
s=s.replace('   diskann::online_gate=gate;diskann::online_heuristic=kind==1?&heuristic:nullptr;', '''   diskann::online_gate=mode=="replay"?nullptr:gate;diskann::online_heuristic=(kind==1 && mode!="replay")?&heuristic:nullptr;
   diskann::TailProfile local;
   auto& prof=formal?profiles[i]:local;
   if(mode=="replay")prof.replay=&tapes[i];
   else if(formal && mode=="profile" && kind!=0){tapes[i].clear();prof.record=&tapes[i];}
   prof.timers=formal && mode=="profile";
   diskann::tail_profile=&prof;''')
s=s.replace('   diskann::online_gate=nullptr;', '''   if(mode=="replay" && prof.replay_pos!=tapes[i].size())throw std::runtime_error("replay length mismatch");
   diskann::tail_profile=nullptr;
   diskann::online_gate=nullptr;''')
s=s.replace('  std::vector<std::array<uint64_t,10>>results(nq);', '  formal=true;\n  std::vector<std::array<uint64_t,10>>results(nq);')
s=s.replace('prefix+".csv"','mode_prefix+".csv"').replace('prefix+".ids"','mode_prefix+".ids"').replace('prefix+".meta.json"','mode_prefix+".meta.json"')
s=s.replace('model_calls,latency_us\\n','model_calls,latency_us,io_us,cpu_partial_us,io_rounds,rounds,empty_rounds,queue_slots,rank_us,feature_us,score_us,cache_hits,comparisons\\n')
s=s.replace("<<calls[i]<<','<<us[i]<<'\\n'", "<<calls[i]<<','<<us[i]<<','<<s.io_us<<','<<s.cpu_us<<','<<s.n_hops<<','<<profiles[i].rounds<<','<<profiles[i].empty_rounds<<','<<profiles[i].queue_slots<<','<<profiles[i].rank_us<<','<<profiles[i].feature_us<<','<<profiles[i].score_us<<','<<s.n_cache_hits<<','<<s.n_cmps<<'\\n'")
s=s.replace('std::cout<<"CONFIG_DONE "<<key<<std::endl;', '''
  if(mode=="profile" && std::stoi(argv[8])==0){
   std::ofstream trace(mode_prefix+".trace.u32",std::ios::binary);
   for(uint32_t qi=0;qi<nq;++qi)for(size_t j=0;j<profiles[qi].nodes.size();++j){
    uint32_t row[3]={qi,profiles[qi].nodes[j],uint32_t(profiles[qi].skipped[j])};
    trace.write((char*)row,sizeof(row));
   }
  }
  std::cout<<"CONFIG_DONE "<<key<<" "<<mode<<std::endl;
  }''')
(D/'search.cpp').write_text(s)
cmds=json.loads((R/'build.commands.json').read_text())
def adapt(cmd):
    return [x.replace(str(R/'native'),str(N)) for x in cmd]
compile_cmd=adapt(cmds[0])
link_cmd=adapt(next(c for c in cmds if str(R/'search.cpp') in c))
link_cmd=[str(D/'search.cpp') if x==str(R/'search.cpp') else x for x in link_cmd]
commands=[compile_cmd,['ar','r',str(N/'libdiskann.a'),str(N/'pq_flash_index.cpp.o')],link_cmd]
(D/'build_commands.json').write_text(json.dumps(commands,indent=2)+'\n')
for i,cmd in enumerate(commands):
    print('BUILD',i,flush=True)
    with (D/f'build_{i}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
print('BUILD COMPLETE',flush=True)
