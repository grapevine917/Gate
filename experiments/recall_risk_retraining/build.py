from pathlib import Path
import json,shutil,subprocess
D=Path(__file__).resolve().parent;W=D.parent/'sift1m_wide_p0';N=D/'native';N.mkdir(exist_ok=True)
for name in ['online_gate.h','online_termination.h','libdiskann.a']:shutil.copy2(W/'native'/name,N/name)
with (N/'online_gate.h').open('a') as f:f.write('''
namespace diskann {
struct RecallHook {uint32_t step=0,node=0,seen=0,applied=0;bool skip=false;float state[17]{};};
inline thread_local RecallHook* recall_hook=nullptr;
inline void recall_override(uint32_t node,const float*state,float&score){
 if(!recall_hook)return;auto&h=*recall_hook;if(h.seen++!=h.step)return;
 if(node!=h.node || state[10]!=0 || std::memcmp(state,h.state,17*sizeof(float)))throw std::runtime_error("Recall branch prefix mismatch");
 score=h.skip?-1.0f:2.0f;++h.applied;
}
}
''')
s=(W/'native/pq_flash_index.cpp').read_text();needle='                online_last_score=(online_miss_only && state[10]!=0)?1.0f:(online_gate?online_gate->score(query_float,nbr.id,state):float(!heuristic_skip));'
assert s.count(needle)==1;s=s.replace(needle,needle+'\n                recall_override(nbr.id,state,online_last_score);');(N/'pq_flash_index.cpp').write_text(s)
commands=json.loads((W/'build_commands.json').read_text());adapt=lambda cmd:[x.replace(str(W/'native'),str(N)) for x in cmd]
core=adapt(commands[0]);link=adapt(next(c for c in commands if str(W/'collect.cpp') in c));link=[str(D/'collect.cpp') if x==str(W/'collect.cpp') else x for x in link]
cmds=[core,['ar','r',str(N/'libdiskann.a'),str(N/'pq_flash_index.cpp.o')],link];(D/'build_commands.json').write_text(json.dumps(cmds,indent=2)+'\n')
for i,c in enumerate(cmds):
 with (D/f'build_{i}.log').open('w') as f:subprocess.run(c,stdout=f,stderr=subprocess.STDOUT,check=True)
print('BUILD COMPLETE',flush=True)
