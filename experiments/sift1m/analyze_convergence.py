#!/usr/bin/env python3
"""Baseline convergence stratification and exact per-query recall-matched ET/pruning."""
import json
import numpy as np
import analyze_lid_buckets as lib
import plot_online_curves as fig
E=lib.E
OUT=E/'convergence_buckets'
COLORS={'fast':(.12,.43,.75),'mid':(.9,.55,.1),'slow':(.7,.2,.3)}

def scatter(path,points):
    W,H=1100,760
    s=fig.image_surface(0,W,H) if path.suffix=='.png' else (fig.svg_surface if path.suffix=='.svg' else fig.pdf_surface)(str(path).encode(),W,H)
    c=fig.create(s);fig.rgb(c,1,1,1);fig.paint(c)
    fig.text(c,35,38,'LID vs recall-matched ET minus pruning I/O',24,True)
    fig.text(c,35,65,'Each point is one test query; exact same Recall@10 = its own baseline Recall@10',15)
    left,top,pw,ph=100,145,920,470
    xs=np.array([p['lid'] for p in points]);ys=np.array([p['delta_cost'] for p in points])
    xmin,xmax=0,float(np.ceil(xs.max()/5)*5);ymin,ymax=float(np.floor(min(ys.min(),-1)/10)*10),float(np.ceil(max(ys.max(),1)/10)*10)
    xp=lambda x:left+(x-xmin)/(xmax-xmin)*pw
    yp=lambda y:top+ph-(y-ymin)/(ymax-ymin)*ph
    for y in np.linspace(ymin,ymax,7):
        fig.segment(c,left,yp(y),left+pw,yp(y),(.88,.9,.92));fig.text(c,left-58,yp(y)+4,f'{y:.0f}',13)
    for x in np.linspace(xmin,xmax,6):
        fig.segment(c,xp(x),top,xp(x),top+ph,(.92,.93,.94));fig.text(c,xp(x)-10,top+ph+25,f'{x:.0f}',13)
    fig.segment(c,left,yp(0),left+pw,yp(0),(.2,.25,.3),2)
    for p in points:
        fig.rgba(c,*COLORS[p['speed_bucket']],.38);fig.arc(c,xp(p['lid']),yp(p['delta_cost']),3.1,0,2*np.pi);fig.fill(c)
    for i,(name,col) in enumerate(COLORS.items()):
        fig.segment(c,100+i*240,105,130+i*240,105,col,4);fig.text(c,140+i*240,110,name+' convergence',15)
    fig.text(c,500,top+ph+55,'LID(q), k=100',17)
    fig.text(c,100,135,'Delta Cost = ET reads - pruning reads',15)
    fig.text(c,35,700,f'{len(points)} matched test queries | Negative: ET cheaper; positive: pruning cheaper | SSD read requests/query',14)
    fig.text(c,35,728,'Minimum measured cost per method at exact target recall; descriptive oracle selection, no interpolation.',14)
    if path.suffix=='.png':assert fig.png(s,str(path).encode())==0
    fig.finish(s);fig.destroy(c);fig.surface_destroy(s)

def main():
    OUT.mkdir(exist_ok=True)
    baseline=lib.mat(E/'baseline_100_idx_uint32.bin','<u4')[:,:10]
    convergence=np.zeros(len(baseline),dtype='int64');total=np.zeros_like(convergence)
    for split in ('train','validation','test'):
        m=np.load(E/f'training_data_l100/{split}/metadata.npy',mmap_mode='r')
        final=(m['node_id'][:,None]==baseline[m['query_id']]).any(axis=1)
        np.maximum.at(convergence,m['query_id'][final],m['step_id'][final].astype('int64')+1)
        np.maximum.at(total,m['query_id'],m['step_id'].astype('int64')+1)
        for qid in np.load(E/f'training_data_l100/{split}/query_ids.npy'):
            np.testing.assert_array_equal(np.sort(m['node_id'][(m['query_id']==qid)&final]),np.sort(baseline[qid]))
    assert np.all(convergence>0) and np.all(convergence<=total)
    a=np.fromfile(E/'learned_termination/baseline_boundaries.f32',dtype='<f4').reshape(-1,21)
    safe=a[:,19]==1
    assert np.all(a[safe,9]>=convergence[a[safe,0].astype('int64')])
    lid=np.load(E/'lid_buckets/lid_k100.npy')
    testids=np.load(E/'training_data_l100/test/query_ids.npy')
    speed=np.empty(len(lid),dtype='<U4');buckets={}
    for b,ids in zip(COLORS,np.array_split(np.lexsort((np.arange(len(lid)),convergence)),3)):
        speed[ids]=b;folder=OUT/f'Q_{b}';folder.mkdir(exist_ok=True)
        ids=np.sort(ids);np.save(folder/'query_ids.npy',ids)
        np.save(folder/'test_query_ids.npy',testids[np.isin(testids,ids)])
        buckets[b]=dict(count=len(ids),test_count=int(np.isin(testids,ids).sum()),convergence_min=int(convergence[ids].min()),convergence_max=int(convergence[ids].max()))
    lib.table(OUT/'queries.csv',[dict(query_id=i,convergence_expansions=int(convergence[i]),total_expansions=int(total[i]),convergence_fraction=float(convergence[i]/total[i]),speed_bucket=speed[i],lid=float(lid[i])) for i in range(len(lid))])
    source=json.loads((E/'lid_buckets/summary.json').read_text())
    import csv
    with (E/'lid_buckets/test_per_query.csv').open() as f:data=list(csv.DictReader(f))
    configs={r['key']:r for r in source}
    arrays={}
    for key in configs:
        if configs[key]['method'] not in ('DiskANN','Learned ET All17','Learned Top3 Expansion Pruning'):continue
        group=[r for r in data if r['key']==key]
        np.testing.assert_array_equal([int(r['query_id']) for r in group],testids)
        arrays[key]=(np.array([float(r['recall_pct']) for r in group]),np.array([int(r['ssd_reads']) for r in group]))
    rows=[]
    for b in COLORS:
        mask=speed[testids]==b
        for key,(rec,cost) in arrays.items():
            rows.append(dict(bucket=b,method=configs[key]['method'],key=key,queries=int(mask.sum()),recall_pct=float(rec[mask].mean()),ssd_reads_per_query=float(cost[mask].mean())))
    lib.table(OUT/'summary.csv',rows);lib.dump(OUT/'summary.json',rows)
    old=lib.METHODS
    lib.METHODS={k:v for k,v in old.items() if v[0]!='ET+pruning'}
    def curves(path):
        W,H=1620,650
        s=fig.image_surface(0,W,H) if path.suffix=='.png' else (fig.svg_surface if path.suffix=='.svg' else fig.pdf_surface)(str(path).encode(),W,H)
        c=fig.create(s);fig.rgb(c,1,1,1);fig.paint(c)
        fig.text(c,35,38,'SIFT1M: ET vs pruning by baseline convergence speed',24,True)
        fig.text(c,35,65,'Convergence = expansions until all baseline final top10 are scored | Full-query thirds | Original 1,000 test queries',14)
        for i,(name,color) in enumerate(lib.METHODS.values()):
            fig.segment(c,45+i*310,100,75+i*310,100,color,3);fig.text(c,85+i*310,105,name,15)
        lo=np.floor(min(r['recall_pct'] for r in rows)*2)/2-.2;xmax=np.ceil(max(r['ssd_reads_per_query'] for r in rows)/10)*10
        for i,b in enumerate(COLORS):
            left,top,pw,ph=75+i*535,180,420,320
            xp=lambda v:left+v/xmax*pw
            yp=lambda v:top+ph-(v-lo)/(100-lo)*ph
            fig.text(c,left,145,f'Q_{b}: {buckets[b]["test_count"]} test queries',19,True)
            for v in np.linspace(lo,100,6):
                fig.segment(c,left,yp(v),left+pw,yp(v),(.88,.9,.92));fig.text(c,left-51,yp(v)+4,f'{v:.2f}',12)
            for v in np.linspace(0,xmax,5):
                fig.segment(c,xp(v),top,xp(v),top+ph,(.93,.94,.95));fig.text(c,xp(v)-10,top+ph+24,f'{v:.0f}',12)
            for method,(name,col) in lib.METHODS.items():
                points=sorted([r for r in rows if r['bucket']==b and r['method']==method],key=lambda r:r['ssd_reads_per_query'])
                for p,z in zip(points,points[1:]):fig.segment(c,xp(p['ssd_reads_per_query']),yp(p['recall_pct']),xp(z['ssd_reads_per_query']),yp(z['recall_pct']),col,2)
                for p in points:
                    fig.rgb(c,*col);fig.arc(c,xp(p['ssd_reads_per_query']),yp(p['recall_pct']),4,0,2*np.pi);fig.fill(c)
            fig.text(c,left,170,'Recall@10 (%)',13);fig.text(c,left+70,555,'SSD read requests / query',15)
        fig.text(c,35,610,'Frozen validation thresholds; speed buckets use baseline trajectories only. Ties split by query ID.',14)
        if path.suffix=='.png':assert fig.png(s,str(path).encode())==0
        fig.finish(s);fig.destroy(c);fig.surface_destroy(s)
    points=[];excluded=[];target=arrays['baseline'][0]
    for j,qid in enumerate(testids):
        best={}
        for method in ('Learned ET All17','Learned Top3 Expansion Pruning'):
            choices=[(int(cost[j]),key) for key,(rec,cost) in arrays.items() if configs[key]['method']==method and abs(rec[j]-target[j])<1e-8]
            if choices:best[method]=min(choices)
        if len(best)!=2:
            excluded.append(dict(query_id=int(qid),target_recall_pct=float(target[j]),missing_methods=[m for m in ('Learned ET All17','Learned Top3 Expansion Pruning') if m not in best]));continue
        et,pr=best['Learned ET All17'],best['Learned Top3 Expansion Pruning']
        points.append(dict(query_id=int(qid),lid=float(lid[qid]),speed_bucket=speed[qid],convergence_expansions=int(convergence[qid]),target_recall_pct=float(target[j]),cost_et=et[0],cost_pruning=pr[0],delta_cost=et[0]-pr[0],et_key=et[1],pruning_key=pr[1]))
    lib.table(OUT/'recall_matched_per_query.csv',points);lib.dump(OUT/'unmatched_queries.json',excluded)
    for ext in ('png','svg','pdf'):
        curves(OUT/f'recall_io_by_convergence.{ext}');scatter(OUT/f'lid_delta_cost.{ext}',points)
    stats={b:dict(matched=sum(p['speed_bucket']==b for p in points),mean_delta_cost=float(np.mean([p['delta_cost'] for p in points if p['speed_bucket']==b]))) for b in COLORS}
    lib.dump(OUT/'manifest.json',dict(buckets=buckets,matched=len(points),unmatched=len(excluded),match_rule='exact per-query baseline Recall@10; minimum measured reads independently per method',cost='SSD index read requests',speed='first expansion prefix containing all baseline final top10',bucket_ties='query ID',matched_stats=stats,lid_delta_correlation=float(np.corrcoef([p['lid'] for p in points],[p['delta_cost'] for p in points])[0,1])))
    (OUT/'REPORT.md').write_text('# 按收敛速度比较 ET 与 pruning\n\n全部 10000 query 按基线最终 top10 均完成评分时的展开前缀长度三等分为 fast / mid / slow，ties 按 query ID。逐 query 校验最终 top10 在轨迹中的成员，safe beam boundary 完成数不早于收敛点。原始数据和原实验均保留。\n\n'+ '\n'.join(f'- {b}: {v}' for b,v in buckets.items())+'\n\n![分桶曲线](recall_io_by_convergence.png)\n\n曲线仅使用原 1000 条 test query 和已冻结 validation 阈值。ET 为 All17，pruning 为 Top3。\n\n![二维分析](lid_delta_cost.png)\n\n二维图每个点是一条 test query，目标 Recall 为该 query 自身的 DiskANN 基线 Recall@10；ET 与 pruning 各自在实测配置中选取 **Recall 完全等于目标** 的最低 I/O。ΔCost = Cost_ET − Cost_Pruning，负数表示 ET 更省，正数表示 pruning 更省。不同 query 的目标 Recall 可以不同，图不表示所有点均为统一 100% Recall。不插值，不外推，不以平均 Recall 相等冒充逐 query Recall 相等。\n\n'+f'成功对齐 {len(points)}/1000，排除 {len(excluded)} 条（见 unmatched_queries.json）。最低成本选择使用 query 的评测结果，仅用于离线描述，不是部署策略，不能用于声称无需 oracle 即可获得相同收益。配置网格不同也会影响结果。\n\n'+ '\n'.join(f'- {b}: matched={v["matched"]}, mean ΔCost={v["mean_delta_cost"]:.3f} reads/query' for b,v in stats.items())+'\n\n`queries.csv` 含全部 query 的收敛点、总展开数、收敛比例和 LID；`summary.csv` 为分桶曲线数据；`recall_matched_per_query.csv` 记录二维图的目标 Recall、两种配置和成本。\n\n复现：`.venv/bin/python experiments/sift1m/analyze_convergence.py`\n')
    print(json.dumps(dict(buckets=buckets,matched=len(points),excluded=len(excluded),stats=stats),indent=2))

if __name__=='__main__':main()
