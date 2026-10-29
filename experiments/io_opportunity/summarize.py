"""Aggregate jointly replayed opportunities; never sum singleton savings."""
from pathlib import Path
import csv,json,collections,hashlib
ROOT=Path(__file__).resolve().parent
out={}
for dataset in ['sift','deep']:
    rows=list(csv.DictReader(open(ROOT/f'{dataset}_L100_queries.csv')))
    candidates=list(csv.DictReader(open(ROOT/f'{dataset}_L100_candidates.csv')))
    groups=collections.defaultdict(list)
    for r in rows:groups[r['reference'],r['policy']].append(r)
    # Select only among actually replayed joint policies, with per-query fallback.
    for ref in ['original','pruning']:
        byq=collections.defaultdict(list)
        for (rr,policy),rs in list(groups.items()):
            if rr==ref:
                for r in rs:byq[r['query']].append(r)
        for strict in [False,True]:
            selected=[]
            for q,rs in byq.items():
                valid=[r for r in rs if int(r['recall'])>=int(r['baseline_recall']) and (not strict or int(r['exact']))]
                selected.append(min(valid,key=lambda r:int(r['reads'])))
            groups[ref,'best_replayed_exact' if strict else 'best_replayed_recall']=selected
    summary={}
    for (ref,policy),rs in groups.items():
        assert len(rs)==1000 and len({r['query'] for r in rs})==1000
        baseline=sum(int(r['baseline_reads']) for r in rs)
        feasible=[r for r in rs if int(r['recall'])>=int(r['baseline_recall']) and int(r['reads'])<=int(r['baseline_reads'])]
        # Infeasible joint policies fall back to their reference, with zero savings.
        omitted=sum(int(r['omitted_reads']) for r in feasible)
        new=sum(int(r['new_reads']) for r in feasible)
        saved=sum(int(r['baseline_reads'])-int(r['reads']) for r in feasible)
        assert saved==omitted-new
        summary[f'{ref}/{policy}']={
            'queries':len(rs),'quality_feasible_queries':len(feasible),
            'quality_violations':sum(int(r['recall'])<int(r['baseline_recall']) for r in rs),
            'io_increase_queries':sum(int(r['reads'])>int(r['baseline_reads']) for r in rs),
            'baseline_reads':baseline,'replayed_reads':sum(int(r['reads']) for r in rs),
            'omitted_reference_reads_feasible':omitted,'replacement_reads_feasible':new,
            'avoidable_io_ratio_gross':omitted/baseline,
            'net_io_saving_ratio':saved/baseline,
            'reference_recall_at_10':sum(int(r['baseline_recall']) for r in rs)/len(rs)/10,
            'replayed_recall_at_10':sum(int(r['recall']) for r in rs)/len(rs)/10,
            'exact_endpoint_queries':sum(int(r['exact']) for r in rs),
            'benefited_queries':sum(int(r['reads'])<int(r['baseline_reads']) for r in feasible),
            'mean_baseline_expansions':sum(int(r['baseline_expansions']) for r in rs)/len(rs),
            'mean_baseline_cache_hits':sum(int(r['baseline_hits']) for r in rs)/len(rs),
        }
    table=collections.defaultdict(lambda:collections.Counter())
    for r in candidates:
        key='cache_miss' if int(r['cache_miss']) else 'cache_hit'
        c=table[key];c['visits']+=1
        c['zero_admission']+=int(r['admitted'])==0
        c['singleton_quality_safe']+=int(r['singleton_recall_ok'])
        c['singleton_exact']+=int(r['singleton_exact'])
        c['singleton_safe_positive_io_value']+=int(r['singleton_recall_ok']) and int(r['singleton_delta'])>0
        c['singleton_safe_zero_io_value']+=int(r['singleton_recall_ok']) and int(r['singleton_delta'])==0
        c['singleton_safe_negative_io_value']+=int(r['singleton_recall_ok']) and int(r['singleton_delta'])<0
    # Three exhaustive primary classes; high-benefit is a diagnostic subclass.
    utility=collections.Counter()
    per_query=collections.defaultdict(collections.Counter)
    classified=[]
    for r in candidates:
        if not int(r['cache_miss']):continue
        safe=bool(int(r['singleton_recall_ok']))
        delta=int(r['singleton_delta'])  # reference requests - skipped requests
        primary='quality_critical' if not safe else ('redundant' if delta>0 else 'replacement_read')
        high=safe and delta<0
        utility[primary]+=1;utility['high_benefit_proxy']+=high
        per_query[r['query']][primary]+=1;per_query[r['query']]['high_benefit_proxy']+=high
        classified.append({'query':r['query'],'node':r['node'],'primary_type':primary,
                           'high_benefit_proxy':int(high),'submitted_request_delta':delta,
                           'strict_result_preserving':r['singleton_exact']})
    total=table['cache_miss']['visits']
    assert sum(utility[k] for k in ['redundant','quality_critical','replacement_read'])==total
    assert utility['high_benefit_proxy']<=utility['replacement_read']
    with open(ROOT/f'{dataset}_L100_miss_utility.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=list(classified[0]));writer.writeheader();writer.writerows(classified)
    with open(ROOT/f'{dataset}_L100_miss_utility_per_query.csv','w') as f:
        keys=['redundant','quality_critical','replacement_read','high_benefit_proxy']
        writer=csv.DictWriter(f,fieldnames=['query','cache_miss_expansions']+keys);writer.writeheader()
        for q,c in sorted(per_query.items(),key=lambda x:int(x[0])):
            writer.writerow({'query':q,'cache_miss_expansions':sum(c[k] for k in keys[:3]),**{k:c[k] for k in keys}})
    distribution={'reference':'original','unit':'query-node cache-miss expansion',
                  'denominator':total,'counts':dict(utility),
                  'fractions':{k:utility[k]/total for k in ['redundant','quality_critical','replacement_read','high_benefit_proxy']},
                  'high_benefit_is_subset_of':'replacement_read',
                  'high_benefit_evidence':'Skipping raises submitted requests without lowering Recall; earlier convergence or priority scheduling not tested.'}
    metrics={}
    for ref in ['original','pruning']:
        rr=summary[f'{ref}/best_replayed_recall'];strict=summary[f'{ref}/best_replayed_exact']
        metrics[ref]={'reference_submitted_ssd_read_requests':rr['baseline_reads'],
                      'avoidable_io_ratio':rr['avoidable_io_ratio_gross'],
                      'net_io_saving_requests':rr['baseline_reads']*rr['net_io_saving_ratio'],
                      'net_io_saving_ratio':rr['net_io_saving_ratio'],
                      'strict_result_preserving_saving_ratio':strict['net_io_saving_ratio'],
                      'strict_result_preserving_saving_requests':strict['baseline_reads']*strict['net_io_saving_ratio']}
    old_path=ROOT.parent/('sift1m/feature_ablation/online/r0_baseline.csv' if dataset=='sift' else 'deep1m/online/r0_baseline_L100.csv')
    old={r['query_row']:r for r in csv.DictReader(open(old_path))}
    for r in groups['original','baseline']:
        assert int(r['baseline_reads'])==int(old[r['query']]['ssd_reads'])
        assert int(r['baseline_expansions'])==int(old[r['query']]['expansions'])
    out[dataset]={'io_unit':'Submitted SSD Read Requests','cache_miss_expansion_utility_distribution':distribution,'headline_metrics':metrics,'joint':summary,'singleton_visit_value_by_cache':dict(table),'prior_baseline_reads_and_expansions_parity':True}

    assert sum(v['visits'] for v in table.values())==sum(int(r['baseline_expansions']) for r in groups['original','baseline'])
    assert table['cache_miss']['visits']==summary['original/baseline']['baseline_reads']
(ROOT/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
lines=['# DiskANN + Pruning：访问价值与可避免读取空间','',
'固定现有索引、各 1,000 条测试查询，L=100、k=10、beam=4、BFS 缓存 10,000 节点。展开预算不设硬上限；联合跳过最多使用参考展开集合内的节点。所有质量约束均逐查询执行。完整口径见 [PROTOCOL.md](PROTOCOL.md)。','',
'以下选取的是已经实际重放的可行组合；测试反馈用于事后选掩码，收益不代表可部署模型的泛化能力或算法无关上界。Avoidable I/O Ratio 按省掉的参考读取计算；净节省率进一步扣除新增读取。','',
'| 数据集 | 参考搜索 | 质量约束 | Submitted SSD Read Requests/查询 | Avoidable I/O Ratio | Net I/O Saving | 受益查询 |','|---|---|---|---:|---:|---:|---:|']
for d in ['sift','deep']:
 for ref in ['original','pruning']:
  for policy,label in [('best_replayed_exact','返回 ID 顺序及距离相同'),('best_replayed_recall','逐查询 Recall@10 不下降')]:
   r=out[d]['joint'][f'{ref}/{policy}']
   lines.append(f"| {d.upper()}1M | {ref} | {label} | {r['baseline_reads']/1000:.3f} | {r['avoidable_io_ratio_gross']:.2%} | {r['net_io_saving_ratio']:.2%} | {r['benefited_queries']}/1000 |")
lines+=['','I/O 计量单位统一为 **Submitted SSD Read Requests**：O_DIRECT Reader 提交的索引读取请求，不是 Physical SSD I/O。历史 CSV 的 `reads` / `ssd_reads` 字段保留兼容，但含义同此口径。','',
'Cache Miss Expansion Utility Distribution（原始搜索的查询—节点展开事件）：','',
'| 数据集 | 冗余节点 | 质量关键节点 | 替代读取节点 | 高收益诊断子类 |',
'|---|---:|---:|---:|---:|']
for d in ['sift','deep']:
 u=out[d]['cache_miss_expansion_utility_distribution'];c=u['counts'];f=u['fractions']
 lines.append('| '+d.upper()+'1M | '+' | '.join(f"{c[k]} ({f[k]:.2%})" for k in ['redundant','quality_critical','replacement_read','high_benefit_proxy'])+' |')
lines+=['','冗余：跳过后 Recall 不下降且请求数减少；质量关键：Recall 下降；替代读取：Recall 不下降且请求数不减少。这三类互斥且覆盖全部 Cache Miss 展开。高收益诊断子类是替代读取中请求数严格增加的节点，说明保留它可能避免后续请求；它不是第四个可相加类别，且尚未验证优先调度或提前收敛。','',
'该分布按查询—节点事件加权，另保存逐查询分布；目前只对 original 参考保存了逐节点反事实明细，pruning 的剩余空间来自联合重放，不能据此声称已获得 pruning 后的节点效用分布。','',
'Strict Result-preserving Saving 定义为返回 ID 顺序及距离完全一致时的净请求节省率：','']
for d in ['sift','deep']:
 m=out[d]['headline_metrics']
 lines.append(f"- {d.upper()}1M：原始 {m['original']['strict_result_preserving_saving_ratio']:.2%}；pruning 后 {m['pruning']['strict_result_preserving_saving_ratio']:.2%}。")
lines+=['','原始与已有 pruning 的参考 Recall@10（GT 前十 ID，不扩展距离并列）：','']
for d in ['sift','deep']:
 a=out[d]['joint']['original/baseline'];b=out[d]['joint']['pruning/baseline']
 lines.append(f"- {d.upper()}1M：原始 {a['reference_recall_at_10']:.2%}，pruning {b['reference_recall_at_10']:.2%}。Pruning 剩余空间相对于自身质量成立。")
lines+=['','单节点访问价值与缓存状态（原始参考）：','',
'| 数据集 | 缓存状态 | 展开次数 | 单节点跳过质量安全 | 安全且读取减少 | 安全但读取增加 |',
'|---|---|---:|---:|---:|---:|']
for d in ['sift','deep']:
 for key,c in out[d]['singleton_visit_value_by_cache'].items():
  lines.append(f"| {d.upper()}1M | {key} | {c['visits']} | {c['singleton_quality_safe']} | {c['singleton_safe_positive_io_value']} | {c['singleton_safe_negative_io_value']} |")
lines+=['','这些单节点统计不能相加得到联合收益；候选被跳过后，队列、模型决策和后续展开都会改变。联合结果以完整重放为准。','',
'不加质量回退的 zero-admission 联合掩码诊断：','']
for d in ['sift','deep']:
 for ref in ['original','pruning']:
  r=out[d]['joint'][f'{ref}/zero_admission_joint']
  lines.append(f"- {d.upper()}1M / {ref}：Recall 下降 {r['quality_violations']} 条，读取增加 {r['io_increase_queries']} 条，返回完全一致 {r['exact_endpoint_queries']}/1000。")
lines+=['','验证：两数据集原始逐查询读取数、展开数与已有 r0 baseline 一致；每次原生重放均验证未缓存展开数等于 n_ios；汇总验证联合省略请求减新增请求等于净节省。','',
'产物：[summary.json](summary.json)、单节点 candidates.csv、联合 queries.csv、[provenance.json](provenance.json)。[run.sh](run.sh) 可重跑；原生 runner 为 `DiskANN/apps/search_io_opportunity.cpp`。','',
'该轮没有测量物理 SSD 计数，也没有将事后反事实计算开销计入部署收益；未证明最优联合掩码。进一步落地须在独立验证集学习可用特征，并在独立测试集评估选择开销、质量与真实净 I/O。']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
