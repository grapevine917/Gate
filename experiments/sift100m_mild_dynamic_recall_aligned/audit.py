"""Independent final checks of source provenance, selected IDs and measurements."""
from common import *
assert json.loads((D/'COMPLETE.json').read_text())['complete']
rows=json.loads((D/'all_unique_test_runs.json').read_text());main=json.loads((D/'main_summary.json').read_text());aligned={r['key']:r for r in json.loads((D/'aligned_summary.json').read_text())}
assert len({(r['key'],r['repeat']) for r in rows})==len(rows)
assert len(main)==6 and {r['kind'] for r in main}==set(range(6)) and all(r['repeats']==12 and r['inference_L']==100 for r in main)
assert all(sha(p)==h for p,h in P['source_hashes'].items())
old=json.loads((W/'TRAIN_COMPLETE.json').read_text());assert all(sha(W/'models'/f'{k}.bin')==h for k,h in old['models'].items())
controls={}
for new,oldkey in [('DiskANN_L100','DiskANN_L100_p0'),('Heuristic_r72_e100','Heuristic_L100_p3')]:
    controls[new]=sha(D/'runs/test'/f'{new}_r0.ids')==sha(W/'runs/test'/f'{oldkey}_r0.ids')
assert all(controls.values())
matches={}
for fn in ['ALIGNMENT_VALIDATION_LOCK.json','ALIGNMENT_TEST_MATCH.json']:
    match=json.loads((D/fn).read_text());rs=[aligned[k] for k in match['keys']]
    assert len(rs)==6 and {r['kind'] for r in rs}==set(range(6)) and all(r['repeats']==6 for r in rs)
    for family in ['MLP','LR']:
        ss=next(r for r in rs if r['family']==family and r['kind'] in [2,3]);dd=next(r for r in rs if r['family']==family and r['kind'] in [4,5]);assert ss['tau']==dd['tau'] and dd['extra_skips']>0
    hits=[]
    for r in rs:
        for rep in range(12,18):
            p=prefix('test',r,rep);ids=np.fromfile(p.with_suffix('.ids'),'<u8').reshape(1000,10)
            h=sum(len(set(a)&set(b)) for a,b in zip(ids,gt['test']));assert h==r['total_hits']
            m=json.loads(p.with_suffix('.meta.json').read_text());assert m['linux_read_bytes']==m['reader_bytes']==m['reader_requests']*4096
        hits.append(r['total_hits'])
    if fn=='ALIGNMENT_TEST_MATCH.json' and match['mode']=='exact':assert len(set(hits))==1
    matches[fn]=dict(mode=match['mode'],test_hits=hits,test_spread_pp=(max(hits)-min(hits))/100)
audit=dict(passed=True,unique_test_runs=len(rows),validation_configs=168,smoke_configs=7,main_repeats=12,aligned_repeats=6,source_hashes_unchanged=True,original_100M_models_unchanged=True,historical_controls_identical=controls,alignment=matches,report_sha256=sha(D/'REPORT.md'))
dump(D/'FINAL_AUDIT.json',audit);print(json.dumps(audit,indent=2))
