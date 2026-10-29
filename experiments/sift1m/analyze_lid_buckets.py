#!/usr/bin/env python3
"""Preserve SIFT1M, compute query LID from GT100 neighbors, stratify measured runs."""
import csv
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import plot_online_curves as fig

E = Path(__file__).resolve().parent
ROOT = E.parents[1]
DATA = ROOT / 'data/sift'
OUT = E / 'lid_buckets'
ONLINE = E / 'et_pruning/online'
METHODS = {'Learned ET All17': ('ET', (.49,.24,.67)),
           'Learned Top3 Expansion Pruning': ('pruning', (.08,.56,.36)),
           'ET + Expansion Pruning': ('ET+pruning', (.81,.13,.18)),
           'DiskANN': ('DiskANN', (.18,.25,.32))}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()

def mat(path, dtype):
    with path.open('rb') as f:
        n,d = np.fromfile(f, '<u4', 2)
        return np.fromfile(f, dtype, int(n)*int(d)).reshape(int(n),int(d))

def write_mat(path, a):
    with path.open('wb') as f:
        np.asarray(a.shape, dtype='<u4').tofile(f)
        a.tofile(f)

def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def table(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def render(path, rows, buckets):
    W,H = 1620,650
    surface = fig.image_surface(0,W,H) if path.suffix == '.png' else (fig.svg_surface if path.suffix == '.svg' else fig.pdf_surface)(str(path).encode(),W,H)
    c = fig.create(surface); fig.rgb(c,1,1,1); fig.paint(c)
    fig.text(c,35,38,'SIFT1M: Recall@10 vs I/O by query LID',25,True)
    fig.text(c,35,65,'MLE LID k=100, Euclidean distances | Full-query tertiles | Original held-out test subset | L=100, beam=4',14)
    for i,(name,color) in enumerate(METHODS.values()):
        x = 45 + i*250
        fig.segment(c,x,95,x+30,95,color,3); fig.text(c,x+40,100,name,15)
    lo = np.floor(min(r['recall_pct'] for r in rows)*2)/2 - .2
    xmax = np.ceil(max(r['ssd_reads_per_query'] for r in rows)/10)*10
    for i,b in enumerate(('low','mid','high')):
        group = [r for r in rows if r['bucket']==b]
        left,top,pw,ph = 75+i*535,175,420,330
        xp = lambda v: left + v/xmax*pw
        yp = lambda v: top+ph-(v-lo)/(100-lo)*ph
        fig.text(c,left,142,f'Q_{b}: {buckets[b]["test_count"]} test queries',19,True)
        for v in np.linspace(lo,100,6):
            fig.segment(c,left,yp(v),left+pw,yp(v),(.88,.9,.92)); fig.text(c,left-51,yp(v)+4,f'{v:.2f}',12)
        for v in np.linspace(0,xmax,5):
            fig.segment(c,xp(v),top,xp(v),top+ph,(.93,.94,.95)); fig.text(c,xp(v)-10,top+ph+24,f'{v:.0f}',12)
        for method,(name,color) in METHODS.items():
            points = sorted([r for r in group if r['method']==method],key=lambda r:r['ssd_reads_per_query'])
            for a,z in zip(points,points[1:]): fig.segment(c,xp(a['ssd_reads_per_query']),yp(a['recall_pct']),xp(z['ssd_reads_per_query']),yp(z['recall_pct']),color,2)
            for r in points:
                fig.rgb(c,*color); fig.arc(c,xp(r['ssd_reads_per_query']),yp(r['recall_pct']),4,0,2*np.pi); fig.fill(c)
        fig.text(c,left+75,top+ph+52,'SSD read requests / query',15)
        fig.text(c,left-20,top-10,'Recall@10 (%)',13)
    fig.text(c,35,600,'Fixed validation-selected operating points; no bucket-specific tuning. Lines connect observed configurations.',14)
    fig.text(c,35,624,'I/O counts index read requests. LID describes local geometry; higher LID need not imply harder search for every query.',13)
    if path.suffix=='.png': assert fig.png(surface,str(path).encode())==0
    fig.finish(surface); fig.destroy(c); fig.surface_destroy(surface)

def main():
    OUT.mkdir(exist_ok=True)
    backup = ROOT/'data/sift_original'
    backup.mkdir(exist_ok=True)
    hashes = {}
    for source in sorted(DATA.iterdir()):
        if not source.is_file(): continue
        target = backup/source.name
        if not target.exists(): shutil.copy2(source,target)
        sha = digest(source)
        assert digest(target)==sha, f'Backup differs: {source}'
        hashes[source.name] = sha
    dump(backup/'SHA256.json',hashes)
    q = mat(DATA/'sift_query.fbin','<f4')
    with (DATA/'sift_base.fbin').open('rb') as f: n,d = np.fromfile(f,'<u4',2)
    base = np.memmap(DATA/'sift_base.fbin',dtype='<f4',mode='r',offset=8,shape=(int(n),int(d)))
    gt = mat(DATA/'gt100.bin','<u4')
    assert gt.shape==(len(q),100)
    lid = np.empty(len(q)); radii = np.empty_like(gt,dtype='float64')
    for start in range(0,len(q),100):
        end = min(start+100,len(q))
        delta = base[gt[start:end]].astype('float64')-q[start:end,None,:]
        radii[start:end] = np.sort(np.sqrt(np.einsum('ijk,ijk->ij',delta,delta)),axis=1)
    assert np.all(radii>0)
    lid[:] = -100/np.log(radii/radii[:,-1:]).sum(axis=1)
    assert np.all(np.isfinite(lid)) and np.all(lid>0)
    np.save(OUT/'lid_k100.npy',lid); np.save(OUT/'neighbor_radii_k100.npy',radii)
    labels = np.empty(len(q),dtype='<U4'); buckets = {}
    testids = np.load(E/'training_data_l100/test/query_ids.npy')
    np.testing.assert_array_equal(mat(E/'online_m3/test_query.fbin','<f4'),q[testids])
    for b,ids in zip(('low','mid','high'),np.array_split(np.lexsort((np.arange(len(q)),lid)),3)):
        labels[ids] = b; folder = OUT/f'Q_{b}'; folder.mkdir(exist_ok=True)
        ids = np.sort(ids); np.save(folder/'query_ids.npy',ids)
        write_mat(folder/'query.fbin',q[ids]); write_mat(folder/'groundtruth_ids.bin',gt[ids])
        testmask = labels[testids]==b if b=='high' else np.isin(testids,ids)
        np.save(folder/'test_query_ids.npy',testids[testmask])
        buckets[b] = dict(count=len(ids),test_count=int(testmask.sum()),lid_min=float(lid[ids].min()),lid_max=float(lid[ids].max()),lid_mean=float(lid[ids].mean()))
    assert sum(v['test_count'] for v in buckets.values())==len(testids)
    table(OUT/'queries.csv',[dict(query_id=i,lid=float(lid[i]),bucket=labels[i],is_test=int(i in set(testids))) for i in range(len(q))])
    assert json.loads((ONLINE/'COMPLETE.json').read_text())['complete']
    rows=[]; perquery=[]
    for config in json.loads((ONLINE/'configs.json').read_text()):
        if config['method'] not in METHODS: continue
        reference = None
        for repeat in range(3):
            prefix = ONLINE/f'r{repeat}_{config["key"]}'
            result = mat(Path(str(prefix)+'_100_idx_uint32.bin'),'<u4')[:,:10]
            stats = np.genfromtxt(Path(str(prefix)+'.csv'),delimiter=',',names=True)
            np.testing.assert_array_equal(stats['query_row'],np.arange(len(testids)))
            recalls = (result[:,:,None]==gt[testids,None,:10]).any(axis=2).mean(axis=1)*100
            if reference is None: reference=(result,stats['ssd_reads'].copy(),recalls)
            else:
                np.testing.assert_array_equal(result,reference[0]); np.testing.assert_array_equal(stats['ssd_reads'],reference[1])
        for b in buckets:
            mask = labels[testids]==b
            rows.append(dict(bucket=b,method=config['method'],key=config['key'],queries=int(mask.sum()),recall_pct=float(recalls[mask].mean()),ssd_reads_per_query=float(stats['ssd_reads'][mask].mean()),et_tau=config['et_tau'],prune_tau=config['prune_tau']))
        for j,qid in enumerate(testids):
            perquery.append(dict(query_id=int(qid),bucket=labels[qid],lid=float(lid[qid]),key=config['key'],recall_pct=float(recalls[j]),ssd_reads=int(stats['ssd_reads'][j])))
        original = next(r for r in json.loads((ONLINE/'summary.json').read_text()) if r['key']==config['key'])
        assert abs(recalls.mean()-original['recall_pct'])<1e-8
        assert abs(stats['ssd_reads'].mean()-original['ssd_reads_per_query'])<1e-8
    table(OUT/'summary.csv',rows); dump(OUT/'summary.json',rows); table(OUT/'test_per_query.csv',perquery)
    for ext in ('png','svg','pdf'): render(OUT/f'recall_io_by_lid.{ext}',rows,buckets)
    dump(OUT/'manifest.json',dict(k=100,lid_formula='-k / sum(log(r_i/r_k)); r_i are Euclidean distances to supplied GT100 base neighbors',partition='all 10000 queries sorted by (LID, query_id), equal-count thirds',buckets=buckets,backup=str(backup),original_sha256=hashes,curve_source=str(ONLINE),curve_queries='original 1000 held-out test queries only',repeat_ids_and_io_identical=True,aggregate_matches_original=True))
    (OUT/'REPORT.md').write_text('# SIFT1M LID 分桶\n\n原始数据完整副本：`data/sift_original/`，SHA256 校验通过；原始实验输出保留。\n\n使用 GT100 的真实邻居向量重算欧氏距离，LID(q) = -100 / Σ log(r_i/r_100)。不能直接将平方距离代入，否则 LID 会缩小一半。GT100 沿用项目现有 ground truth，其既有精确性抽查见 `../groundtruth_validation.json`。\n\n全部 10000 个 query 按 (LID, query_id) 排序三等分；边界 ties 按 query_id 分配。LID 是局部几何难度代理，不保证逐 query 搜索难度单调。\n\n'+ '\n'.join(f'- Q_{b}: {v["count"]} 条，LID [{v["lid_min"]:.4f}, {v["lid_max"]:.4f}]，test {v["test_count"]} 条。' for b,v in buckets.items())+'\n\n曲线仅使用现有 1000 条 test query，按桶独立计算 Recall@10 与平均 SSD read requests/query。ET=All17 learned ET，pruning=Top3 expansion pruning；ET+pruning 使用原 validation 选定的组合。保持模型和阈值，未按桶调参。三次原始运行的结果 ID 和 I/O 均一致，桶加权均值复现原整体结果。\n\n![Recall–I/O](recall_io_by_lid.png)\n\n`queries.csv` 保存所有 query 的 LID 和桶；`Q_*/` 保存 query、GT IDs、原始 ID 和 test ID；`test_per_query.csv` 保存逐 query 的 Recall/I/O；`summary.csv` 保存曲线数据。`groundtruth_ids.bin` 是带形状头的 ID 矩阵，不含距离。\n\n复现：`.venv/bin/python experiments/sift1m/analyze_lid_buckets.py`\n')
    print(json.dumps(buckets,indent=2),flush=True)

if __name__=='__main__': main()
