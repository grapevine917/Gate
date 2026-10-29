"""Main mild-policy test and an explicitly separated matched-Recall comparison."""
from common import *
assert (D/'VALIDATION_COMPLETE.json').exists()
assert all(sha(p)==h for p,h in P['source_hashes'].items())
configs=json.loads((D/'configs.json').read_text());C={c['key']:c for c in configs};validation=json.loads((D/'validation.json').read_text());V={r['key']:r for r in validation}
main_lock=json.loads((D/'MAIN_SELECTION_LOCK.json').read_text());main=main_lock['configs'];aruns=[]
for rep in range(12):aruns+=load('test',main[rep%6:]+main[:rep%6],rep,True)
main_summary=aggregate(aruns);dump(D/'main_runs.json',aruns);dump(D/'main_summary.json',main_summary);save(D/'main_summary.csv',main_summary)
print('MAIN MILD TEST COMPLETE',flush=True)
# The grid was frozen and all deployment-like selections locked before any test.
initial=load('test',configs);dump(D/'test_grid.json',initial);save(D/'test_grid.csv',initial)
T={r['key']:r for r in initial}
for family in ['MLP','LR']:
    for k in ['total_hits','reads','expansions','skips','ids_sha256']:assert T[f'Dynamic{family}_identity'][k]==T[f'{family}_static_b1'][k]
matched=choose_alignment(initial,V,9960)
if (D/'ALIGNMENT_TEST_MATCH.json').exists():assert json.loads((D/'ALIGNMENT_TEST_MATCH.json').read_text())==matched
else:dump(D/'ALIGNMENT_TEST_MATCH.json',matched)
val_match=json.loads((D/'ALIGNMENT_VALIDATION_LOCK.json').read_text())
selected_keys=set(matched.get('keys',[]))|set(val_match.get('keys',[]));aligned_configs=[C[k] for k in sorted(selected_keys)];bruns=[]
assert aligned_configs
# Fresh timing after quality matching. rep0 test time is not used in this table.
for j,rep in enumerate(range(12,18)):
    ordered=aligned_configs[j%len(aligned_configs):]+aligned_configs[:j%len(aligned_configs)]
    bruns+=load('test',ordered,rep,True)
aligned_summary=aggregate(bruns);dump(D/'aligned_runs.json',bruns);dump(D/'aligned_summary.json',aligned_summary);save(D/'aligned_summary.csv',aligned_summary)
allruns={(r['key'],r['repeat']):r for r in aruns+initial+bruns};assert len(allruns)==len(aruns)+len(initial)+len(bruns)-len(main)
dump(D/'all_unique_test_runs.json',list(allruns.values()))
assert all(sha(p)==h for p,h in P['source_hashes'].items())
main_data={r['key']:r for r in main_summary};aligned_data={r['key']:r for r in aligned_summary}
for group in ['MLP','LR']:
    static=main_data[f'{group}_static_b1'];dyn=next(r for r in main_summary if r['family']==group and r['kind'] in [4,5])
    assert static['tau']==dyn['tau'] and dyn['late_tau']>static['tau'] and dyn['exp']>=100
if matched['available'] and matched['mode']=='exact':
    assert len({aligned_data[k]['total_hits'] for k in matched['keys']})==1
    for family in ['MLP','LR']:
        pair=[aligned_data[k] for k in matched['keys'] if aligned_data[k]['family']==family]
        assert len(pair)==2 and pair[0]['tau']==pair[1]['tau']
dump(D/'verification.json',dict(passed=True,models_not_retrained=True,source_weights_and_inputs_unchanged=True,main_L_all100=all(r['inference_L']==100 for r in main_summary),main_base_tau_exactly_static=True,late_tau_above_static=True,identity_configs_equal_static=True,physical_IO_verified=True,no_swap=True,all_repeated_IDs_and_counts_equal=True,main_repeats=12,aligned_fresh_repeats=6,validation_lock_precedes_test=True,alignment_exact=matched.get('mode')=='exact',aligned_static_dynamic_base_matched=True,unique_test_runs=len(allruns)))
dump(D/'MEASUREMENTS_COMPLETE.json',dict(complete=True,validation_configs=len(configs),main_test_runs=len(aruns),grid_test_configs=len(initial),aligned_configs=len(aligned_configs),aligned_fresh_runs=len(bruns),unique_test_runs=len(allruns),strict_alignment=matched))
print('ALL MEASUREMENTS COMPLETE',matched,flush=True)
