from pathlib import Path
import json
R=Path(__file__).resolve().parent
P=dict(dataset='SIFT100M',base_shape=[100000000,128],base_dtype='uint8',index_R=64,index_Lbuild=100,PQ_bytes=34,index_build_RAM_GiB=8,index_build_threads=12,K=10,beam=4,threads=4,MKL_threads=1,cache_nodes=10000,train_queries=1200,internal_holdout_queries=200,validation_queries=1000,test_queries=1000,targets=[90,95,98,99,99.5,99.7],seed=20261030,
 L_pilot_candidates=[15,20,30,50,75,100,160,300,500,1000,2000,4000],
 L_selection_rule='Validation baseline only. Evaluate ascending candidates until first L>=300 with Recall>=99.9%, else4000. Include15,100,max; fill to8 with remaining candidates<=max maximizing minimum logarithmic spacing, smaller L breaks ties. Freeze before training and full grid/test. Baseline64 integer L from10..max uses same log-gap filling.',
 training='Fresh100M-only weights. Matched mixed-L LR/XGB/stateMLP/qvMLP use identical sampled cache-miss rows,1200 train queries, proxy contribution labels. Separate L100 MLP uses all L100 cache-miss rows. ET predicts current absolute prefix GT recall on unpruned mixed-L paths.',
 neural_training='160 inner epochs, LR .001/40,.0003/40,.0001/40,.00003/40; choose internal200-query best loss then refit all1200 for selected epoch. Same hidden128,64; state17, qv529. One seed; denormal flushing enabled.',
 qv='Lossless ZSTD level1 independent32-vector uint8 blocks; entire payload+uint64 offsets resident. Exact uint8->float128 features [q,v,abs(q-v),q*v,state17]. Every score decodes a block; timed CPU includes decode. No extra online vector file reads. Resource implementation differs from1M full float table.',
 node_ID_schema='uint32 prune_nodes.u32 sidecar matches every20-float prune row; row1 placeholder0. Native score records: q128 float, node uint32 bits, state17 float, label/expected floats. Never numeric float node IDs. Search returns uint64 IDs.',
 P0_A='Fixed L100 beam4; Vanilla, rank>72 AND completed>100 heuristic, Ours. IndependentMLP/LR versions,9 balanced cyclic repeats each. Gate validation retain-FNR1%; no final Recall guarantee.',
 P0_B='MLP_L100 and mixed-L LR versions;5 families each.10 unique total families across studies,64 configs each;8 Gate risk levels,8 ET thresholds,8 predeclared joint pairs. Identical controls shared.',
 P0_C='Baseline, mixed-L LR/XGB/stateMLP/qvMLP; matched same training rows, each64configs. Same heldout L100 candidate diagnostics and native inference cost.',
 evaluation='Freeze validation-calibrated grid, lock choices by recall target/objective before test. Full fixed640 test grid descriptive; repeat final objective frontiers, common exact integer-hit winners, feasible-quality winners and validation-locked choices to3 until stable. P0A9 repeats; no cost interpolation.',
 physical_io='O_DIRECT reader requests/bytes plus /proc/self/io read_bytes measured in timed window. Extra unexpected reads fail verification; WSL2 guest block traffic, not NAND counters.',
 limits=['Finite8 joint threshold pairs; no policy-optimality claim','ET trained on unpruned paths; joint trajectory shift possible','Proxy contribution label is not counterfactual Recall harm','One training seed; timing repeats omit training variability','High quality may exceed finite L range or graph capability; report unavailable rather than extrapolate','100M uint8 graph and resident compressed vector implementation differ from1M float implementation; cross-dataset CPU cost not pure scale comparison'])
path=R/'PROTOCOL_BASE.json'
if path.exists():assert json.loads(path.read_text())==P
else:path.write_text(json.dumps(P,indent=2,ensure_ascii=False)+'\n')
print('PROTOCOL FROZEN')
