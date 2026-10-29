#!/usr/bin/env python3
"""Plot/report joint ET+pruning against freshly measured component baselines."""
import json
import plot_termination_curves as fig
from calibrate_et_pruning import OUT
ONLINE=OUT/'online'
METHODS={'DiskANN':(.18,.25,.32),'Budget Early Termination':(.86,.40,.10),
    'Learned ET All17':(.49,.24,.67),'Learned Top3 Expansion Pruning':(.08,.56,.36),
    'ET + Expansion Pruning':(.81,.13,.18)}

def main():
    assert json.loads((ONLINE/'COMPLETE.json').read_text())['complete']
    rows=json.loads((ONLINE/'summary.json').read_text());base=rows[0]
    selected=json.loads((OUT/'selected_joint_configs.json').read_text())
    provenance=json.loads((OUT/'calibration_provenance.json').read_text())
    fig.COLORS=METHODS
    notes=[
        'Joint model: All17 ET at beam boundaries + Top3 expansion pruning per candidate; both state construction and inferences included.',
        'ET recalibrated on pruned validation paths. Joint configurations selected by real validation Recall / read budgets; no test tuning.',
        'Same test queries, randomized joint timing. Warm-host latency bars: trial min/max. SSD reads are index requests, not device counters.']
    for ext in ('png','svg','pdf'):
        fig.render(ONLINE/f'recall_cost_curves.{ext}',rows,
                   title='ET + expansion pruning: joint live-search comparison',notes=notes)
    reductions=[];frontier=[]
    for r in rows:
        reductions.append(dict(key=r['key'],method=r['method'],recall_loss_pp=base['recall_pct']-r['recall_pct'],
            reads_reduction_pct=100*(1-r['ssd_reads_per_query']/base['ssd_reads_per_query']),
            expansions_reduction_pct=100*(1-r['expansions_per_query']/base['expansions_per_query']),
            latency_reduction_pct=100*(1-r['latency_us']/base['latency_us'])))
    for method in METHODS:
        group=[r for r in rows if r['method']==method]
        for cost in ('ssd_reads_per_query','expansions_per_query','latency_us'):
            for r in group:
                if not any(v['recall_pct']>=r['recall_pct'] and v[cost]<=r[cost] and
                           (v['recall_pct']>r['recall_pct'] or v[cost]<r[cost]) for v in group):
                    frontier.append(dict(method=method,cost=cost,key=r['key'],recall_pct=r['recall_pct'],value=r[cost]))
    (ONLINE/'reductions_vs_baseline.json').write_text(json.dumps(reductions,indent=2)+'\n')
    (ONLINE/'observed_frontiers.json').write_text(json.dumps(frontier,indent=2)+'\n')
    # Describe domination at observed points; never use this to re-select a test deployment configuration.
    comparisons=[]
    joint=[r for r in rows if r['method']=='ET + Expansion Pruning']
    et=[r for r in rows if r['method']=='Learned ET All17']
    for r in joint:
        for cost in ('ssd_reads_per_query','expansions_per_query','latency_us'):
            dominated_by=[v['key'] for v in et if v['recall_pct']>=r['recall_pct'] and v[cost]<=r[cost] and
                          (v['recall_pct']>r['recall_pct'] or v[cost]<r[cost])]
            dominates=[v['key'] for v in et if r['recall_pct']>=v['recall_pct'] and r[cost]<=v[cost] and
                       (r['recall_pct']>v['recall_pct'] or r[cost]<v[cost])]
            comparisons.append(dict(joint_key=r['key'],cost=cost,dominated_by_standalone_et=dominated_by,
                                    dominates_standalone_et=dominates))
    (ONLINE/'observed_joint_vs_et.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    joint_mid=next(r for r in rows if r['key']=='joint_fnr_0.5pct_et_query_0.25pct')
    et_mid=next(r for r in rows if r['key']=='et_unsafe_query_0.5pct')
    joint_hi=next(r for r in rows if r['key']=='joint_fnr_0.25pct_et_query_0.1pct')
    prune_hi=next(r for r in rows if r['key']=='prune_fnr_0.5pct')
    lines=['# ET + Expansion Pruning','',
        '主组合为冻结的All17 learned ET + 冻结的Top3 expansion-pruning MLP。无需重训；ET在beam边界检查，p_safe>=tau_ET则结束整次搜索；未停止时逐候选使用p_retain<tau_P跳过展开，保留原beam slot。',
        '固定SIFT1M、L=100、K=10、beam=4、threads=4、cache=10000，仍是原1000条test query。各配置全部重新交错随机计时3次，包含两个模型的真实C++推理与state构造，不拼接旧latency。',
        '', '## Validation calibration and joint selection','',
        'Pruning沿用validation FNR targets=0.1%、0.25%、0.5%、1%、2%的五个已定tau_P；这些是原baseline候选标签上的名义FNR，不声称是组合轨迹上的实际FNR。',
        '每个tau_P先在validation完整运行pruning-only，并在ET禁用（tau_ET=1.1）的同一路径上采集ET原生概率。新的safe标签表示已评分节点包含该档完整pruning-only最终top10；这样可校准ET相对pruning-only的额外风险，而不是误用完整baseline的旧状态/阈值。',
        'Conditional ET按误停query targets=0%、0.1%、0.25%、0.5%、1%、2%、5%、10%重新校准，共40个联合配置。安全标签沿同一pruned path单调；原生float32 nextafter阈值排除首个超预算分数及ties。',
        '条件ET安全只保留pruning-only端点，无法撤销pruning损失。两类FNR/误停率分母不同，也不相加；最终以实际Recall和成本评估组合。',
        f'40个组合全部真实运行validation。Validation baseline Recall={provenance["validation_baseline"]["recall_pct"]:.2f}%；按Recall损失预算 '+', '.join(str(x) for x in provenance['recall_loss_budgets_pp'])+' 个百分点，选择最低SSD Reads/query，依次以较少Expansions、较高Recall、较保守风险参数打破并列。Latency不参与选择。',
        f'共锁定{len(selected)}个不同组合后才开始test；零Recall损失预算下没有可行联合点，不虚构配置。Validation预算不保证test满足相同Recall。',
        '', '| selected validation Recall-loss budgets pp | nominal pruning FNR target | conditional ET query target | tau_P | tau_ET | validation Recall | validation Reads |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in selected:
        lines.append(f'| {r["selected_for_recall_budgets_pp"]} | {r["nominal_prune_fnr_pct"]:g}% | {r["conditional_et_query_target_pct"]:g}% | {r["prune_tau"]:.9g} | {r["et_tau"]:.9g} | {r["recall_pct"]:.3f}% | {r["ssd_reads_per_query"]:.3f} |')
    lines+=['','## Live test','',
        f'所有{len(rows)}配置交错重复3次，共{len(rows)*3}次计时；报告median。无模型DiskANN、Budget ET（13档）、All17 ET（8档）、Top3 pruning（5档）均重新测量。',
        '', '| Method | Configuration | Recall@10 | SSD Reads/query | Expansions/query | Latency ms | QPS | ET stops/query | Candidate skips/query | Additional unsafe ET queries |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['method']=='ET + Expansion Pruning':
            config=f'P nominal FNR {r["nominal_prune_fnr_pct"]:g}%, conditional ET query {r["conditional_et_query_target_pct"]:g}%'
        elif r['method']=='Learned ET All17':config=f'unsafe-query target {r["parameter"]:g}%'
        elif r['method']=='Learned Top3 Expansion Pruning':config=f'retain FNR target {r["parameter"]:g}%'
        elif r['budget'] is not None:config=f'{r["budget"]} reads'
        else:config='no models'
        bad='—' if r.get('test_conditional_et_unsafe_queries') is None else str(r['test_conditional_et_unsafe_queries'])
        lines.append(f'| {r["method"]} | {config} | {r["recall_pct"]:.3f}% | {r["ssd_reads_per_query"]:.3f} | {r["expansions_per_query"]:.3f} | {r["latency_us"]/1000:.3f} | {r["qps"]:.1f} | {r["et_stopped_query_rate"]:.3f} | {r["skips_per_query"]:.3f} | {bad} |')
    lines+=['','## Curves','','![Recall / costs](online/recall_cost_curves.png)','',
        '图中展示固定validation选定组合的test曲线；仅连接实测点，无插值拟合或test调参。下排放大Recall>=99%。每个成本维度的实测非支配点及joint-vs-ET支配关系另存JSON，仅供结果描述。',
        '', '## Interpretation', '',
        '本轮组合没有明显改善单独All17 ET的整体Recall–成本权衡，多个联合点在高召回区域被单独组件的实测点支配。不能将两类gate的独立节省简单相加。',
        f'联合P nominal FNR=0.5% / conditional ET query target=0.25%：Recall={joint_mid["recall_pct"]:.2f}%、Reads={joint_mid["ssd_reads_per_query"]:.3f}、Expansions={joint_mid["expansions_per_query"]:.3f}、Latency={joint_mid["latency_us"]/1000:.3f}ms；单独ET query target=0.5%：Recall={et_mid["recall_pct"]:.2f}%、Reads={et_mid["ssd_reads_per_query"]:.3f}、Expansions={et_mid["expansions_per_query"]:.3f}、Latency={et_mid["latency_us"]/1000:.3f}ms。',
        f'相同Recall={joint_hi["recall_pct"]:.2f}%时，联合P=0.25%/conditional ET=0.1%使用Reads={joint_hi["ssd_reads_per_query"]:.3f}，单独pruning FNR=0.5%为Reads={prune_hi["ssd_reads_per_query"]:.3f}；联合并未进一步降低读取。',
        '中低召回段有联合点以较低Recall换取少量额外读取节省，但当前离散测量不能证明相同Recall下的统一优势。两个模型的收益可能重叠，额外candidate推理也会增加成本；本实验未做同tau_ET的完整因果消融，不能把所有差值归因于pruning或其推理开销。',
        '单独ET沿用原baseline validation阈值，联合ET按pruned validation轨迹重校准。结论针对这两个冻结模型和本次校准/选择流程，尚不排除在pruned状态上重新训练ET后有更好组合。',
        '', '## Verification','',
        '两模型同时禁用时全部test逐项复现baseline。关闭pruning时复现standalone ET；关闭ET时各档复现standalone pruning的ID、距离与展开/读取/gate计数。两个模型各自的Python/C++概率已审计。',
        '40次validation joint及全部选定test joint逐query核验第一次停止位置、已完成展开数、ET推理次数和停止计数；safe ET停止逐项保留pruning-only ID与距离。额外unsafe停止恰好对应pruning-only端点变化。',
        '所有计时repeat结果ID一致，Recall独立重算，expansions=gate_evaluated-gate_skipped核验，Budget ET overshoot<=3核验。Standalone controls保持前序搜索行为。',
        'SSD Reads为索引请求数，不是设备物理读取；Latency为当前主机预热状态。测试query此前用于探索性实验，未参与本次阈值校准/组合选择。',
        '', '## Reproduce','', '在learned-gate目录运行：','',
        '```bash','cmake --build DiskANN/build --target search_disk_index -j 4',
        '.venv/bin/python experiments/sift1m/calibrate_et_pruning.py',
        '.venv/bin/python experiments/sift1m/run_online_et_pruning.py',
        '.venv/bin/python experiments/sift1m/plot_et_pruning.py','```']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(complete=True,frozen_models=True,
        validation_grid=40,selected_joint_points=len(selected),measured_test_runs=len(rows)*3,
        heldout_test_queries=1000,plots=True),indent=2)+'\n')
    print('ET + pruning report and PNG / SVG / PDF curves exported.',flush=True)

if __name__=='__main__':main()
