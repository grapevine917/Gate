"""Isolated fixed-weight LR with a progress-dependent threshold."""
from pathlib import Path
import json,shutil,subprocess
D=Path(__file__).resolve().parent;R=D.parent/'sift100m_p0';N=D/'native';N.mkdir(exist_ok=True)
for name in ['online_gate.h','online_termination.h','libdiskann.a']:shutil.copy2(R/'native'/name,N/name)
with (N/'online_gate.h').open('a') as f:f.write('''
namespace diskann {
struct ProgressGate { int rank, expansions; float late_tau; };
inline thread_local const ProgressGate* progress_gate=nullptr;
inline thread_local uint32_t progress_eligible=0, progress_extra_skips=0;
}
''')
s=(R/'native/pq_flash_index.cpp').read_text()
# Use one effective threshold for every decision and history update.
s=s.replace('online_gate->threshold','progress_tau')
needle='                const bool heuristic_skip=online_heuristic && online_heuristic->skip(candidate_rank,completed_expansions);'
assert s.count(needle)==1
s=s.replace(needle,'''                float progress_tau=online_gate?online_gate->threshold:0.0f;
                const bool progress_active=online_gate && progress_gate && int64_t(candidate_rank)>progress_gate->rank && int64_t(completed_expansions)>progress_gate->expansions;
                if(progress_active){progress_tau=std::max(progress_tau,progress_gate->late_tau);if(!_nhood_cache.count(nbr.id))++progress_eligible;}
'''+needle)
needle='                if(history_skip) {'
assert s.count(needle)==1
s=s.replace(needle,'''                if(history_skip && progress_active && online_last_score>=online_gate->threshold)++progress_extra_skips;
'''+needle)
(N/'pq_flash_index.cpp').write_text(s)
s=(R/'search.cpp').read_text()
s=s.replace('diskann::OnlineHeuristic heuristic{rank,expansions};','diskann::OnlineHeuristic heuristic{rank,expansions};diskann::ProgressGate progress{rank,expansions,et_tau};std::vector<uint32_t> eligible(nq),extra(nq);')
s=s.replace('   diskann::online_gate=gate;', '''   diskann::progress_gate=(kind==2 && rank>=0 && expansions>=0)?&progress:nullptr;
   diskann::progress_eligible=0;diskann::progress_extra_skips=0;
   diskann::online_gate=gate;''')
s=s.replace('   diskann::online_gate=nullptr;', '''   eligible[i]=diskann::progress_eligible;extra[i]=diskann::progress_extra_skips;
   diskann::progress_gate=nullptr;
   diskann::online_gate=nullptr;''')
s=s.replace('model_calls,latency_us\\n','model_calls,latency_us,io_us,io_rounds,progress_eligible,progress_extra_skips\\n')
s=s.replace("<<calls[i]<<','<<us[i]<<'\\n'", "<<calls[i]<<','<<us[i]<<','<<s.io_us<<','<<s.n_hops<<','<<eligible[i]<<','<<extra[i]<<'\\n'")
(D/'search.cpp').write_text(s)
cmds=json.loads((R/'build.commands.json').read_text())
adapt=lambda c:[x.replace(str(R/'native'),str(N)).replace(str(R/'search.cpp'),str(D/'search.cpp')) for x in c]
commands=[adapt(cmds[0]),['ar','r',str(N/'libdiskann.a'),str(N/'pq_flash_index.cpp.o')],adapt(next(c for c in cmds if str(R/'search.cpp') in c))]
(D/'build_commands.json').write_text(json.dumps(commands,indent=2)+'\n')
for i,c in enumerate(commands):
    print('BUILD',i,flush=True)
    with (D/f'build_{i}.log').open('w') as f:subprocess.run(c,stdout=f,stderr=subprocess.STDOUT,check=True)
print('BUILD COMPLETE',flush=True)
