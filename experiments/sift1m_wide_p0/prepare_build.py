"""Freeze the SIFT1M wide-queue experiment and build isolated native runners."""
from pathlib import Path
import hashlib,json,shutil,struct,subprocess
import numpy as np
D=Path(__file__).resolve().parent;R=D.parent/'main_comparison';N=D/'native';DATA=D/'data';M=D/'models'
for p in [N,DATA,M]:p.mkdir(exist_ok=True)
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
LS=[30,50,75,100,160,300,500,1000]
protocol=dict(dataset='SIFT1M',base_vectors=1000000,dimensions=128,index=str(D.parent/'sift1m/index'),L_training_and_learned_grid=LS,P0_A_L=300,old_P0_A_L=100,old_P0_B_max_L=300,beam=4,K=10,threads=4,MKL_threads=1,cache_nodes=10000,training_queries=1200,internal_holdout_queries=200,validation_queries=1000,test_queries=1000,seed=20261024,train_candidates_per_query_per_L=64,models='Fresh 17D MLP and LR trained on identical mixed-wide-L rows; fresh ET on the corresponding prefix rows; no prior model weights reused',MLP='17-128-64-1; internal query holdout checkpoint selection, maximum80epochs; refit all1200 queries with selected epoch count',label='retain iff admitted_neighbors>0 OR final baseline Top10; same conservative contribution proxy as original P0',P0_A='L300; shared Vanilla/Heuristic, Static MLP, Static LR, Dynamic MLP, Dynamic LR. Static threshold calibrated retain-FNR1% on validation L300. Dynamic chosen on validation L300 under Recall>=corresponding static default. Twelve balanced six-method cyclic timing repeats; shared run pool also used by P0-B.',P0_B='Nine families: DiskANN, Heuristic, MLP, LR, DynamicMLP, DynamicLR, ET, ET_MLP, ET_LR. One shared control pool;64 configs/family, 8Lx8presets except baseline64distinctL.',targets=[90,95,98,99,99.5,99.7,99.9],risks=[0,.001,.0025,.005,.01,.02,.05,.1],dynamic_presets=[[0,0,1,-1],[0,3,.72,1],[0,5,.5,.9],[0,6,.25,.75],[2,5,.5,.9],[3,6,.25,.75],[4,7,.25,.75],[1,7,.64,.92]],dynamic_rule='score < base_tau; when zero-based rank > r AND completed_expansions > e use max(base_tau, late_tau); cache hits protected',validation='Calibrate gate thresholds on full validation L300 baseline cache-miss states; freeze64/config grids; lock per-target per-objective choices before test',test='Full frozen grid descriptive frontiers; repeat validation-selected policies and all final objective frontiers to3 repeats until stable. No test-guided grid extension.',limits=['Previously inspected query split; exploratory follow-up, not an independent blind benchmark','P0-A L300 and wider mixed-L training both differ from original L100 study; not a pure queue-width causal ablation','Finite dynamic presets, not globally optimal','Candidate proxy FNR is not final Recall loss','ET trained on unpruned trajectories; joint groups may have trajectory shift'])
if (D/'PROTOCOL.json').exists():assert json.loads((D/'PROTOCOL.json').read_text())==protocol
else:dump(D/'PROTOCOL.json',protocol)
for split in ['train','validation','test']:
    for ext in ['fbin','gt']:
        dest=DATA/f'{split}.{ext}'
        if not dest.exists():shutil.copy2(R/'data'/dest.name,dest)
    shutil.copy2(R/'data'/f'{split}_ids.npy',DATA/f'{split}_ids.npy')
shutil.copy2(R/'data/internal_holdout_ids.npy',DATA/'internal_holdout_ids.npy')
shutil.copy2(R/'splits.json',D/'splits.json')
ids={s:set(np.load(DATA/f'{s}_ids.npy').tolist()) for s in ['train','validation','test']}
assert not ids['train']&ids['validation'] and not ids['train']&ids['test'] and not ids['validation']&ids['test']
(D/'L.txt').write_text(' '.join(map(str,LS))+'\n');(D/'calibration_L.txt').write_text('300\n')
with (M/'dummy.bin').open('wb') as f:
    f.write(b'LRGATE01');f.write(struct.pack('<I',0))
    for a in [np.zeros(17),np.ones(17),np.zeros(17),np.zeros(1)]:np.asarray(a,'<f4').tofile(f)
inputs=[R/'native/search',D.parent/'sift1m/index_disk.index',D.parent/'sift1m/index_pq_compressed.bin',D.parent/'sift1m/index_pq_pivots.bin',*[DATA/f'{s}.{e}' for s in ['train','validation','test'] for e in ['fbin','gt']]]
dump(D/'INPUT_LOCK.json',{str(p):sha(p) for p in inputs})
for name in ['online_gate.h','online_termination.h','libdiskann.a']:shutil.copy2(R/'native'/name,N/name)
with (N/'online_gate.h').open('a') as f:f.write('''
namespace diskann {
struct ProgressGate { int rank, expansions; float late_tau; };
inline thread_local const ProgressGate* progress_gate=nullptr;
inline thread_local uint32_t progress_eligible=0,progress_extra_skips=0;
}
''')
s=(R/'native/pq_flash_index.cpp').read_text().replace('online_gate->threshold','progress_tau')
old='                const bool heuristic_skip=online_heuristic && online_heuristic->skip(candidate_rank,completed_expansions);'
assert s.count(old)==1
s=s.replace(old,'''                float progress_tau=online_gate?online_gate->threshold:0;
                const bool progress_active=online_gate && progress_gate && int64_t(candidate_rank)>progress_gate->rank && int64_t(completed_expansions)>progress_gate->expansions;
                if(progress_active){progress_tau=std::max(progress_tau,progress_gate->late_tau);if(!_nhood_cache.count(nbr.id))++progress_eligible;}
'''+old)
s=s.replace('                if(history_skip) {','''                if(history_skip && progress_active && online_last_score>=online_gate->threshold)++progress_extra_skips;
                if(history_skip) {''')
(N/'pq_flash_index.cpp').write_text(s)
shutil.copy2(R/'collect.cpp',D/'collect.cpp')
s=(D.parent/'sift100m_p0/search.cpp').read_text().replace('uint8_t*q=nullptr;float*gd=nullptr;','float*q=nullptr,*gd=nullptr;').replace('load_aligned_bin<uint8_t>','load_aligned_bin<float>').replace('PQFlashIndex<uint8_t,uint32_t>','PQFlashIndex<float,uint32_t>')
a=s.index(' const char*names[]=');b=s.index(' std::ifstream configs',a)
s=s[:a]+''' const char*names[]={"","","state","lr","state","lr","","state","lr"};
 std::unique_ptr<diskann::OnlineGate> owned,et;
 if(first_kind>=2 && first_kind!=6)owned=std::make_unique<diskann::OnlineGate>(std::string(argv[4])+"/"+names[first_kind]+".bin","",0);
 if(first_kind>=6)et=std::make_unique<diskann::OnlineGate>(std::string(argv[4])+"/et.bin","",1.1);
'''+s[b:]
s=s.replace('float tau,et_tau;','float tau,et_tau,late_tau;').replace('>>rank>>expansions){','>>rank>>expansions>>late_tau){')
s=s.replace('diskann::OnlineHeuristic heuristic{rank,expansions};','diskann::OnlineHeuristic heuristic{rank,expansions};diskann::ProgressGate progress{rank,expansions,late_tau};std::vector<uint32_t>eligible(nq),extra(nq);')
s=s.replace('   diskann::online_gate=gate;','''   diskann::progress_gate=(kind==4||kind==5)?&progress:nullptr;
   diskann::progress_eligible=0;diskann::progress_extra_skips=0;
   diskann::online_gate=gate;''')
s=s.replace('   diskann::online_gate=nullptr;','''   eligible[i]=diskann::progress_eligible;extra[i]=diskann::progress_extra_skips;diskann::progress_gate=nullptr;
   diskann::online_gate=nullptr;''')
s=s.replace('model_calls,latency_us\\n','model_calls,latency_us,io_us,io_rounds,progress_eligible,progress_extra_skips\\n')
s=s.replace("<<calls[i]<<','<<us[i]<<'\\n'", "<<calls[i]<<','<<us[i]<<','<<s.io_us<<','<<s.n_hops<<','<<eligible[i]<<','<<extra[i]<<'\\n'")
(D/'search.cpp').write_text(s)
(D/'verify.cpp').write_text('''#include "online_gate.h"
#include <iostream>
int main(int argc,char**argv){if(argc!=2)return 2;mkl_set_num_threads_local(1);std::string m=argv[1];diskann::OnlineGate lr(m+"/lr.bin","",0),st(m+"/state.bin","",0),et(m+"/et.bin","",0);diskann::OnlineGate* models[]={&lr,&st,&et};std::ifstream f(m+"/check.bin",std::ios::binary);uint32_t n;f.read((char*)&n,4);float row[20],q[128]={},err[3]={};for(unsigned i=0;i<n;++i){f.read((char*)row,sizeof(row));for(int k=0;k<3;++k)err[k]=std::max(err[k],std::abs(models[k]->score(q,0,row)-row[17+k]));}if(!f)return 3;std::cout<<"{\\"LR\\":"<<err[0]<<",\\"MLP\\":"<<err[1]<<",\\"ET\\":"<<err[2]<<",\\"samples\\":"<<n<<"}\\n";for(float x:err)if(x>2e-5)return 4;}
''')
commands=json.loads((R/'build.commands.json').read_text());template=next(c for c in commands if str(R/'search.cpp') in c)
prefix=[x.replace(str(R/'native'),str(N)) for x in template[:template.index(str(R/'search.cpp'))]]
libs=template[template.index('-L/opt/intel/oneapi/mkl/latest/lib/intel64'):];libs=[x.replace(str(R/'native'),str(N)) for x in libs]
builds=[[*prefix,'-c',str(N/'pq_flash_index.cpp'),'-o',str(N/'pq_flash_index.cpp.o')],['ar','r',str(N/'libdiskann.a'),str(N/'pq_flash_index.cpp.o')]]+[[*prefix,str(D/f'{name}.cpp'),'-o',str(N/name),*libs] for name in ['collect','search','verify']]
dump(D/'build_commands.json',builds)
for i,c in enumerate(builds):
    print('BUILD',i,flush=True)
    with (D/f'build_{i}.log').open('w') as f:subprocess.run(c,stdout=f,stderr=subprocess.STDOUT,check=True)
dump(D/'BUILD_COMPLETE.json',dict(complete=True,search_sha256=sha(N/'search')))
print('BUILD COMPLETE',flush=True)
