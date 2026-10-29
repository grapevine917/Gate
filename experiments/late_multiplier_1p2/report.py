"""Full metrics and independent audit for the fixed 1.2x experiment."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
D=Path(__file__).resolve().parent;ROOT=D.parents[1]
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,rows):
    with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
runlock=json.loads((D/'RUN_LOCK.json').read_text());assert sha(D/'run.py')==runlock['coordinator'];assert all(sha(p)==h for p,h in runlock['protocols'].items())
names={0:'DiskANN',1:'Heuristic',2:'Static MLP',3:'Static LR',4:'Dynamic MLP ×1.2',5:'Dynamic LR ×1.2'}
def aggregate(rows,keys):
    out=[]
    for key in keys:
        rs=[r for r in rows if r['key']==key];r=rs[0].copy();r.pop('repeat');r['repeats']=len(rs)
        for k in ['total_hits','physical_bytes','reads','expansions','skips','extra_skips','eligible','model_calls','io_rounds','ids_sha256']:assert len({a[k] for a in rs})==1,(key,k)
        for k in ['p50_ms','p95_ms','p99_ms','mean_ms','qps','io_ms']:r[k]=float(np.median([a[k] for a in rs]))
        r['p95_min_ms']=min(a['p95_ms'] for a in rs);r['p95_max_ms']=max(a['p95_ms'] for a in rs);out.append(r)
    return out
def table(rows):
    fields=[('Group',lambda r:names[r['kind']]),('L',lambda r:str(r['inference_L'])),('Recall %',lambda r:f"{r['recall']:.2f}"),('P50 ms',lambda r:f"{r['p50_ms']:.4f}"),('P95 ms',lambda r:f"{r['p95_ms']:.4f}"),('P99 ms',lambda r:f"{r['p99_ms']:.4f}"),('Mean ms',lambda r:f"{r['mean_ms']:.4f}"),('QPS',lambda r:f"{r['qps']:.1f}"),('KiB/query',lambda r:f"{r['physical_bytes']/1024:.2f}"),('Reads/query',lambda r:f"{r['reads']:.3f}"),('Exp/query',lambda r:f"{r['expansions']:.3f}"),('Skips/query',lambda r:f"{r['skips']:.3f}"),('Calls/query',lambda r:f"{r['model_calls']:.3f}"),('IO ms',lambda r:f"{r['io_ms']:.4f}"),('IO rounds/query',lambda r:f"{r['io_rounds']:.3f}"),('Eligible/query',lambda r:f"{r['eligible']:.3f}"),('Extra skips/query',lambda r:f"{r['extra_skips']:.3f}")]
    return '| '+' | '.join(n for n,_ in fields)+' |\n| '+' | '.join(['---']*len(fields))+' |\n'+'\n'.join('| '+' | '.join(f(r) for _,f in fields)+' |' for r in rows)+'\n'
report=['# SIFT1M / SIFT100M：固定晚期阈值为静态阈值的 1.2 倍','',
'两套纯 L=100 训练的模型均复用，不重新训练。主实验固定推理 L=100、K=10、beam=4、4 查询线程、10000 缓存节点；每数据集验证/测试各1000查询。只改变晚期阈值倍率，不重新选择早期阈值及 rank/扩展门槛。晚期阈值=1.2×静态基础阈值，不是在上一轮晚期阈值上再乘1.2。',
'', '主实验六组12次循环交错复测；另复测上一轮Recall对照的六组参数6次，只把动态组倍率改为1.2。后者继承之前测试后描述性匹配的参数，不重新匹配、不插值，实际Recall残差如实报告。相同配置/重复在两表之间共享。两数据集依次运行，避免SSD竞争。延迟及QPS为各轮指标中位数。','']
effects=[];audits={};all_summaries=[];tail=[]
for ds,label in [('sift1m','SIFT1M'),('sift100m','SIFT100M')]:
    out=D/ds;P=json.loads((out/'PROTOCOL.json').read_text());assert json.loads((out/'MEASUREMENTS_COMPLETE.json').read_text())['complete']
    assert all(sha(p)==h for p,h in P['source_hashes'].items())
    runs=json.loads((out/'all_test_runs.json').read_text());assert len({(r['key'],r['repeat']) for r in runs})==len(runs)
    old=ROOT/'experiments'/f'{ds}_mild_dynamic_recall_aligned';oldT={r['key']:r for r in json.loads((old/'test_grid.json').read_text())}
    W=ROOT/'experiments'/('sift100m_p0' if ds=='sift100m' else 'sift1m_wide_p0')
    with (W/'data/test.gt').open('rb') as f:n,k=np.fromfile(f,'<u4',2);gt=np.fromfile(f,'<u4').reshape(int(n),int(k))[:,:10]
    checked=set()
    for r in runs:
        p=out/'runs/test'/f"{r['key']}_r{r['repeat']}";m=json.loads(p.with_suffix('.meta.json').read_text())
        assert m['linux_read_bytes']==m['reader_bytes']==m['reader_requests']*4096 and m['swap_before_kib']==m['swap_after_kib']==0
        assert sha(p.with_suffix('.ids'))==r['ids_sha256']
        if r['kind']<=3:assert r['ids_sha256']==oldT[r['key']]['ids_sha256']
        if r['key'] not in checked:
            ids=np.fromfile(p.with_suffix('.ids'),'<u8').reshape(1000,10);hits=sum(len(set(a)&set(b)) for a,b in zip(ids,gt));assert hits==r['total_hits'];checked.add(r['key'])
    validation=json.loads((out/'validation.json').read_text());save(out/'validation.csv',validation)
    report+=['',f'## {label}','', '主实验阈值与触发条件：','']
    for c in P['configs']:
        if c['kind'] in [4,5] and c['key'] in P['groups']['main']:report.append(f"- {c['family']}：τ={c['tau']:.10g}，late τ={c['late_tau']:.10g}；rank>{c['rank']} 且 completed>{c['exp']}。")
    for study in ['main','recall_reference']:
        rr=json.loads((out/(study+'_runs.json')).read_text());rs=aggregate(rr,P['groups'][study]);save(out/(study+'_summary.csv'),rs);dump(out/(study+'_summary.json'),rs)
        expected=12 if study=='main' else 6;assert len(rs)==6 and all(r['repeats']==expected for r in rs)
        lo=min(r['total_hits'] for r in rs);hi=max(r['total_hits'] for r in rs)
        report+=['', '### 固定推理 L=100 主实验' if study=='main' else '### 原 Recall 对照参数，改为 ×1.2 后复测','']
        if study=='recall_reference':report.append(f"实际测试Recall区间 {lo/100:.2f}%–{hi/100:.2f}%，差距 {(hi-lo)/100:.2f} 个百分点；"+('六组严格对齐。' if lo==hi else '未严格对齐，不能作为严格同Recall结论。'))
        report+=['',table(rs)]
        for family in ['MLP','LR']:
            s=next(r for r in rs if r['family']==family and r['kind'] in [2,3]);d=next(r for r in rs if r['family']==family and r['kind'] in [4,5]);assert s['tau']==d['tau'] and d['late_tau']==1.2*s['tau']
            e=dict(dataset=ds,study=study,family=family,recall_delta_pp=d['recall']-s['recall'],p95_change_percent=(d['p95_ms']/s['p95_ms']-1)*100,qps_change_percent=(d['qps']/s['qps']-1)*100,bytes_change_percent=(d['physical_bytes']/s['physical_bytes']-1)*100,extra_skips_per_query=d['extra_skips'])
            effects.append(e);report.append(f"{family} 动态相对静态：Recall {e['recall_delta_pp']:+.2f} pp，P95 {e['p95_change_percent']:+.2f}%，QPS {e['qps_change_percent']:+.2f}%，读取字节 {e['bytes_change_percent']:+.2f}%；动态P95范围 {d['p95_min_ms']:.4f}–{d['p95_max_ms']:.4f} ms。")
        all_summaries.extend([dict(dataset=ds,study=study,**r) for r in rs])
        if study=='main':
            for family in ['MLP','LR']:
                pair=[r for r in rs if r['family']==family];s=next(r for r in pair if r['kind'] in [2,3]);d=next(r for r in pair if r['kind'] in [4,5]);aa={}
                for r in pair:aa[r['key']]=[np.genfromtxt(out/'runs/test'/f"{r['key']}_r{rep}.csv",delimiter=',',names=True) for rep in range(12)]
                med={key:np.median([a['latency_us'] for a in arrays],axis=0) for key,arrays in aa.items()};ix=np.argsort(med[s['key']])[-50:]
                for i in ix:tail.append(dict(dataset=ds,family=family,query=int(i),static_median_ms=float(med[s['key']][i]/1000),dynamic_median_ms=float(med[d['key']][i]/1000),static_hits=int(aa[s['key']][0]['hits'][i]),dynamic_hits=int(aa[d['key']][0]['hits'][i]),extra_skips=int(aa[d['key']][0]['progress_extra_skips'][i])))
                tt=[r for r in tail if r['dataset']==ds and r['family']==family];a=np.mean([r['static_median_ms'] for r in tt]);b=np.mean([r['dynamic_median_ms'] for r in tt]);report.append(f"固定静态{family}最慢50个查询：平均逐查询中位延迟 {a:.4f}→{b:.4f} ms（{(b/a-1)*100:+.2f}%），总命中 {sum(r['static_hits'] for r in tt)}→{sum(r['dynamic_hits'] for r in tt)}。仅作配对尾部诊断。")
    rejected=json.loads((out/'IO_REJECTIONS.json').read_text()) if (out/'IO_REJECTIONS.json').exists() else []
    audits[ds]=dict(passed=True,source_hashes_unchanged=True,all_controls_and_static_IDs_match_previous=True,physical_IO_verified=True,no_swap=True,repeated_counts_equal=True,unique_test_runs=len(runs),validation_configs=len(validation),IO_rejected_attempts=len(rejected))
    dump(out/'COMPLETE.json',audits[ds]);dump(out/'STATUS.json',dict(phase='complete',complete=True))
report+=['','## 核验与解释边界','',
'固定×1.2参数，无验证/测试后调参；参数冻结后协议与模型文件哈希保持不变。返回ID重算Recall；各次reader字节、Linux read_bytes和读取次数×4096严格相等且无交换；重复间ID和离散计数一致；本轮原生、启发式和静态模型输出与历史相同。IO异常尝试归档重试，不混入统计。',
'晚期触发后仍按模型分数判断，×1.2不等于启发式触发后全部剪枝。Extra skips表示当前动态轨迹中基础阈值会保留的额外剪枝，不是独立节点因果收益。小幅计时差异不能直接视为稳定加速，重复计时不包含重训练不确定性；查询集此前已分析，结果属于探索性验证。SIFT100M使用uint8索引、SIFT1M使用float索引，跨数据集差异不能单独归因于数据量。',
'', '主实验各数据集的r/e沿用各自上一轮验证选择，跨数据集并非完全相同触发策略。Recall-reference表沿用上一轮描述性匹配参数，其早期阈值/触发条件可以区别于主实验；没有声称本次自动保持Recall对齐。两轮历史动态计时不直接用于计算本轮加速。',
'', '复现：项目根目录运行 `.venv/bin/python experiments/late_multiplier_1p2/run.py`；已通过检查的原始测量会复用。完整参数及源文件哈希见各数据集 PROTOCOL.json。','']
save(D/'all_summaries.csv',all_summaries);save(D/'effects.csv',effects);save(D/'paired_tail.csv',tail)
(D/'REPORT.md').write_text('\n'.join(report));dump(D/'COMPLETE.json',dict(complete=True,datasets=audits,effects=effects,report_sha256=sha(D/'REPORT.md')))
print('\n'.join(report));print('ALL COMPLETE',flush=True)
