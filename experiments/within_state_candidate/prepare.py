from common import *
import struct,joblib
log('PREPARE common baseline beam states, frozen model scores, contribution observations and exhaustive interventions')
m=np.load(DATA/'test/metadata.npy');state=np.load(DATA/'test/pre_io_state.npy');qids=np.load(DATA/'test/query_ids.npy');n=len(m);glob=GROUPS['Global'];norm=np.load(S/'training_data_l100/pre_io_state_train_normalization.npz');scale=norm['scale'].astype(float)
train=np.load(DATA/'train/pre_io_state.npy',mmap_mode='r');phase_cuts=np.quantile(train[train[:,10]==0,7],[.2,.4,.6,.8]).tolist();max_beam_distance=float(3/scale[5]);distance_cut=max_beam_distance+1e-7
proto=dict(experiment='Within-State Candidate Discrimination',dataset='SIFT1M',queries=1000,L=100,K=10,beam=4,cache_nodes=10000,frozen_model_source=str(FG/'models'),groups=GROUPS,seeds=SEEDS,no_training_or_threshold_selection=True,common_trajectory='all candidates retained / no pruning; existing actual native tau0 traces from all 9 frozen models, joined one-to-one with validated baseline expansion records',population='primary: cache-miss candidates actually scored by all models; cache-hit candidates recorded but protected and excluded from ranking/intervention primary sample',pairing='all unordered pairs of noncached candidates within same (query_id, beam_step); never cross-query pairs; no outcome-based pair selection',natural_exact='all 11 Global columns bit-identical; if empty report no support / NA rather than invent exact-state pairs',natural_near='same real beam; require all Global dimensions except unexpanded_count bit-identical; normalized Global L2 <= locked maximum of 3 / training std(unexpanded_count) + tolerance',global_normalization_source=str(DATA/'pre_io_state_train_normalization.npz'),global_distance_limit=distance_cut,canonical_control='use observed state from beam selection position0 as common Global vector; retain each candidate six original Candidate features; rescore frozen models only; not an actually deployed search or new training; slight canonical-input distribution shift disclosed',phase_cuts_train_only=phase_cuts,contribution_targets={'admitted_neighbors':'number of actually admitted neighbors during original expansion, observed direct marginal contribution in original processing order; not causal necessity','fresh_neighbors':'number of first-visited neighbors','conservative_retain':'admitted>0 or node in baseline final Top10; not proof of necessity','lost_hits':'baseline Top10 GT hits minus hits after exactly one forced Skip at this candidate; subsequent graph search completes normally','harmful_skip':'lost_hits>0','safe_positive_IO':'lost_hits<=0 and baseline_reads>after_reads'},counterfactual='exhaustive single-node intervention for every cache-miss baseline candidate; baseline model tau=0 (all retained), force score=-1 exactly once before decision; byte-identical original node, 17D pre-I/O state and native passive score checked; no reuse of E4-1 labels from a different pruned trajectory',intervention_model='Full17 seed42 used only as passive tau0 score carrier; thresholds disabled, so all three groups share identical original search path',sham='1000 fixed random cache-miss Keep->Keep interventions; same final IDs, distances, reads, expansions, neighbor comparisons required',metrics='graded Pairwise Accuracy on unequal contribution counts; within-beam binary conditional AUC on positive/negative pairs (equal to tie-adjusted pairwise accuracy on those pairs); score ties count 0.5, target ties excluded with support reported; query/beam macro metrics additionally saved',heuristic_controls=['negative candidate PQ distance','negative candidate queue rank'],confidence='1000 target-query cluster bootstrap draws, jointly resampling the three matched training seeds for group summaries and differences; individual pairs not treated independent',limitations=['Previously observed test set; exploratory diagnostic','Same-beam direct admission contributions depend on original processing order and overlapping neighborhoods','Single-node causal marginal labels cannot be summed as joint pruning effects','Canonical Global-only ties are a construction sanity check, not independent evidence against Global-only','Pairwise Accuracy / conditional AUC do not establish end-to-end Recall/latency gains'],CPU_affinity=CPUS)
if (R/'PROTOCOL.json').exists():assert json.loads((R/'PROTOCOL.json').read_text())==proto
else:dump(R/'PROTOCOL.json',proto)
keys={(int(v['query_id']),int(v['node_id'])):i for i,v in enumerate(m)};scores=np.zeros((n,3,3),float);score_errors={};carrier_steps=np.empty(n,np.uint32)
for gi,g in enumerate(GROUPS):
 for si,seed in enumerate(SEEDS):
  seen=np.zeros(n,bool);parts=[];indices=[]
  with (FG/'audit'/f'{g}_s{seed}_tau0.trace').open('rb') as f:
   while head:=f.read(8):
    row,count=struct.unpack('<II',head);a=np.frombuffer(f.read(count*76),'<f4').reshape(count,19);ix=np.array([keys[(int(qids[row]),int(node))] for node in a[:,0]]);assert not seen[ix].any();seen[ix]=True;assert np.array_equal(a[:,1:18],state[ix]);scores[ix,gi,si]=a[:,18]
    if g=='Full17' and seed==42:carrier_steps[ix]=np.arange(count)
  assert seen.all();bundle=joblib.load(FG/'models'/f'{g}_s{seed}/model.joblib');choose=np.random.default_rng(seed).choice(np.flatnonzero(state[:,10]==0),2000,replace=False);err=float(np.max(np.abs(predict(bundle,state[choose])-scores[choose,gi,si])));assert err<3e-6;score_errors[f'{g}_s{seed}']=err
# Read the actual edge observation table, without relying on the conservative binary training label.
edges=np.load(DATA/'test/neighbors.npy',mmap_mode='r');assert len(edges)==64*n;admit=np.zeros(n,np.uint32);fresh=np.zeros(n,np.uint32);discovery=np.zeros(n,np.uint32)
for start in range(0,n,8192):
 end=min(n,start+8192);a=edges[start*64:end*64].reshape(end-start,64);expected=np.arange(start,end,dtype=np.uint64)[:,None];assert np.all(a['expansion_row']==expected);assert np.all(a['current_node_id']==m['node_id'][start:end,None]);assert np.all(a['query_id']==m['query_id'][start:end,None]);assert np.all(a['beam_step']==m['beam_step'][start:end,None]);admit[start:end]=a['neighbor_was_inserted'].sum(1);fresh[start:end]=((a['flags']&1)!=0).sum(1);discovery[start:end]=a['neighbor_led_to_improvement'].sum(1)
assert np.array_equal(admit==0,m['oracle_zero_admission'].astype(bool));assert np.array_equal(((admit>0)|(m['oracle_in_final_topk']!=0)).astype(np.uint8),m['label'])
beams={}
for i,v in enumerate(m):beams.setdefault((int(v['query_id']),int(v['beam_step'])),[]).append(i)
canonical=state.copy();pairL=[];pairR=[];beamRows=[];max_delta=np.zeros(len(glob));exact_pairs=0
for bi,((qid,beam),ix) in enumerate(beams.items()):
 ix=np.array(ix,int);positions=state[ix,9].astype(int);assert len(ix)<=4 and len(np.unique(positions))==len(ix);first=ix[np.argmin(positions)];canonical[np.ix_(ix,glob)]=state[first,glob];valid=ix[state[ix,10]==0]
 pairs=0
 for j,a in enumerate(valid):
  for b in valid[j+1:]:
   delta=np.abs(state[a,glob]-state[b,glob]);max_delta=np.maximum(max_delta,delta);other=[k for k,v in enumerate(glob) if v!=5];assert np.all(delta[other]==0);distance=float(np.linalg.norm(delta/scale[glob]));assert distance<=distance_cut+1e-8;exact_pairs+=not delta.any();pairL.append(a);pairR.append(b);pairs+=1
 beamRows.append(dict(beam_row=bi,query_id=qid,beam_step=beam,candidates=len(ix),miss_candidates=len(valid),miss_pairs=pairs,phase_value=int(state[first,7]),phase_bin=int(np.searchsorted(phase_cuts,state[first,7],side='left')),snapshot_record=int(first)))
canonical_scores=np.zeros_like(scores)
for gi,g in enumerate(GROUPS):
 for si,seed in enumerate(SEEDS):canonical_scores[:,gi,si]=predict(joblib.load(FG/'models'/f'{g}_s{seed}/model.joblib'),canonical)
_,first_rows,beam_inverse=np.unique(np.column_stack([m['query_row'],m['beam_step']]),axis=0,return_index=True,return_inverse=True)
canonical_scores[:,0,:]=canonical_scores[first_rows[beam_inverse],0,:]
left=np.array(pairL,np.uint32);right=np.array(pairR,np.uint32);dist=np.linalg.norm((state[left][:,glob].astype(float)-state[right][:,glob])/scale[glob],axis=1);phase=np.searchsorted(phase_cuts,state[left,7],side='left').astype(np.uint8)
np.savez_compressed(R/'records.npz',state=state,canonical_state=canonical,query_id=m['query_id'],query_row=m['query_row'],beam_step=m['beam_step'],node_id=m['node_id'],sample_id=m['sample_id'],cached=state[:,10]!=0,admitted=admit,fresh=fresh,discovery_edges=discovery,conservative_retain=m['label'],in_final_top10=m['oracle_in_final_topk'],native_scores=scores,canonical_scores=canonical_scores,carrier_step=carrier_steps)
np.savez_compressed(R/'pairs.npz',left=left,right=right,query_id=m['query_id'][left],query_row=m['query_row'][left],beam_step=m['beam_step'][left],global_distance=dist,global_exact=dist==0,phase=phase,beam_gap=np.abs(state[left,9]-state[right,9]).astype(np.uint8))
csvsave(R/'beam_support.csv',beamRows)
# True marginal retrieval contribution under the common no-pruning path.
miss=np.flatnonzero(state[:,10]==0);np.save(R/'intervention_records.npy',miss)
def write_requests(path,ix,skip):
 with path.open('wb') as f:
  f.write(struct.pack('<I',len(ix)))
  for k,i in enumerate(ix):f.write(struct.pack('<5I18f',k,int(m['query_row'][i]),int(carrier_steps[i]),int(m['node_id'][i]),skip,float(scores[i,2,0]),*state[i]))
write_requests(R/'interventions.bin',miss,1);sham=np.random.default_rng(SEED).choice(miss,1000,replace=False);np.save(R/'sham_records.npy',sham);write_requests(R/'sham.bin',sham,0)
dump(R/'PREPARED.json',dict(complete=True,records=n,queries=1000,beams=len(beams),primary_miss_records=len(miss),within_beam_miss_pairs=len(left),natural_exact_pairs=int(exact_pairs),max_global_differences=dict(zip([str(g) for g in glob],max_delta.tolist())),max_normalized_global_distance=float(dist.max()),score_errors=score_errors,protocol_sha256=sha(R/'PROTOCOL.json')))
log(f'PREPARED {len(miss)} interventions and {len(left)} within-beam candidate pairs')
