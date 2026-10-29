"""Three standalone scientific figures. Matplotlib if available; vector SVG fallback offline."""
from pathlib import Path
import json,math,html,xml.etree.ElementTree as ET
R=Path(__file__).resolve().parent;F=R/'figures';F.mkdir(exist_ok=True)
D=json.loads((R/'frontiers.json').read_text())
FAMILIES=['DiskANN', 'Heuristic', 'MLP', 'LR', 'DynamicMLP', 'DynamicLR', 'ET', 'ET_MLP', 'ET_LR']
LABELS=['DiskANN', 'Heuristic', 'MLP', 'LR', '动态 MLP', '动态 LR', 'ET', 'ET + MLP', 'ET + LR']
COLORS=['#111111','#E69F00','#0072B2','#009E73','#0072B2','#009E73','#777777','#CC79A7','#D55E00']
DASH=['','9 4','','','7 3','7 3','3 3','9 3 2 3','9 3 2 3'];MARKERS=['o','s','^','D','v','+','x','s','D']
SPECS=[('p95_ms','recall_p95','P95 query latency (ms)',1.),('qps','recall_qps','Throughput (queries / second)',1.),('physical_bytes','recall_physical_bytes','Measured block read (KiB / query)',1/1024),('reads','mechanism_recall_requests','Direct read requests / query',1.)]
try:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 backend='matplotlib'
except ImportError:backend='standalone SVG (offline Matplotlib unavailable)'
def nice(v):
 if v<=0:return 1
 b=10**math.floor(math.log10(v));t=v/b
 return next(n*b for n in [1,2,2.5,5,10] if n>=t)
def marker(x,y,color,kind,r=4):
 stroke=f'stroke="{color}" stroke-width="1.8"';fill='white'
 if kind=='o':return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" {stroke}/>'
 if kind=='s':return f'<rect x="{x-r}" y="{y-r}" width="{2*r}" height="{2*r}" fill="{fill}" {stroke}/>'
 if kind=='D':pts=[(x,y-r*1.3),(x+r,y),(x,y+r*1.3),(x-r,y)]
 elif kind=='^':pts=[(x,y-r*1.3),(x+r*1.2,y+r),(x-r*1.2,y+r)]
 elif kind=='v':pts=[(x,y+r*1.3),(x+r*1.2,y-r),(x-r*1.2,y-r)]
 elif kind=='+':return f'<path d="M{x-r*1.3} {y}H{x+r*1.3}M{x} {y-r*1.3}V{y+r*1.3}" {stroke}/>'
 else:return f'<path d="M{x-r} {y-r}L{x+r} {y+r}M{x-r} {y+r}L{x+r} {y-r}" {stroke}/>'
 return f'<polygon points="'+ ' '.join(f'{a},{b}' for a,b in pts)+f'" fill="{fill}" {stroke}/>'
def svg(metric,name,label,factor):
 out=['<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="830" viewBox="0 0 1320 830"><rect width="1320" height="830" fill="white"/><g font-family="Arial, sans-serif" fill="#222">',f'<text x="660" y="37" text-anchor="middle" font-size="25">Recall@10 vs {html.escape(label)}</text>','<text x="660" y="65" text-anchor="middle" font-size="14">SIFT1M | 1,000 evaluation queries | beam 4 | 4 query threads | equal 64-configuration tuning budget</text>']
 for panel,(xmin,xmax,title) in enumerate([(90.,100.,'Low to high recall'),(99.,100.,'High-recall detail')]):
  x0=105+panel*650;y0=115;w=500;h=440
  vals=[r[metric]*factor for fam in FAMILIES for r in D[metric][fam] if xmin<=r['recall']<=xmax]
  if not vals:vals=[0,1]
  lo=0 if panel==0 else min(vals)*.90;hi=max(vals)*1.06;step=nice((hi-lo)/5);ymin=math.floor(lo/step)*step;ymax=math.ceil(hi/step)*step
  if ymax==ymin:ymax=ymin+step
  xx=lambda v:x0+(v-xmin)/(xmax-xmin)*w;yy=lambda v:y0+h-(v-ymin)/(ymax-ymin)*h
  out.append(f'<defs><clipPath id="clip{panel}"><rect x="{x0}" y="{y0}" width="{w}" height="{h}"/></clipPath></defs><text x="{x0+w/2}" y="96" font-size="17" text-anchor="middle">{title}</text>')
  ticks=[90,92,95,98,100] if panel==0 else [99,99.2,99.5,99.8,100]
  for t in ticks:
   x=xx(t);out.append(f'<path d="M{x} {y0}V{y0+h}" stroke="#e6e6e6"/><text x="{x}" y="{y0+h+27}" font-size="14" text-anchor="middle">{t:g}</text>')
  for j in range(int(round((ymax-ymin)/step))+1):
   v=ymin+j*step;y=yy(v);out.append(f'<path d="M{x0} {y}H{x0+w}" stroke="#e6e6e6"/><text x="{x0-12}" y="{y+5}" font-size="14" text-anchor="end">{v:g}</text>')
  out.append(f'<path d="M{x0} {y0}V{y0+h}H{x0+w}" fill="none" stroke="#444" stroke-width="1.2"/><g clip-path="url(#clip{panel})">')
  for i,fam in enumerate(FAMILIES):
   rows=D[metric][fam];points=' '.join(f'{xx(r["recall"]):.3f},{yy(r[metric]*factor):.3f}' for r in rows);dash=f' stroke-dasharray="{DASH[i]}"' if DASH[i] else ''
   out.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[i]}" stroke-width="{2.8 if i==0 else 2}"{dash}/>')
   for r in rows:
    tooltip=f"{LABELS[i]} | L={r['L']} | tau={r['tau']:.5g} | ET={r['et_tau']:.5g} | Recall={r['recall']:.3f}% | {label}={r[metric]*factor:.5g} | repeats={r.get('timing_repeats',1)}"
    out.append('<g><title>'+html.escape(tooltip)+'</title>'+marker(xx(r['recall']),yy(r[metric]*factor),COLORS[i],MARKERS[i],3.7)+'</g>')
  out.append(f'</g><text x="{x0+w/2}" y="612" font-size="17" text-anchor="middle">Recall@10 (%)</text><text transform="translate({x0-77},{y0+h/2}) rotate(-90)" font-size="16" text-anchor="middle">{html.escape(label)}</text>')
 for i,(text,color) in enumerate(zip(LABELS,COLORS)):
  x=55+(i%4)*325;y=650+(i//4)*32;dash=f' stroke-dasharray="{DASH[i]}"' if DASH[i] else ''
  out.append(f'<path d="M{x} {y}H{x+35}" stroke="{color}" stroke-width="2.2"{dash}/>'+marker(x+17,y,color,MARKERS[i],4)+f'<text x="{x+44}" y="{y+5}" font-size="13">{html.escape(text)}</text>')
 out+=['<text x="660" y="763" text-anchor="middle" font-size="13">Markers: measured configurations. Lines guide the eye; no claim of exact matched-Recall configurations between markers.</text>','<text x="660" y="788" text-anchor="middle" font-size="13">Separate empirical frontiers; at least 3 repeats, shared P0-A policies use 12 repeats.</text>','</g></svg>']
 p=F/f'{name}.svg';p.write_text(''.join(out));ET.parse(p)
for metric,name,label,factor in SPECS:
 if backend=='matplotlib':
  plt.rcParams.update({'font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
  fig,axes=plt.subplots(1,2,figsize=(12,5.8))
  for panel,ax in enumerate(axes):
   for i,fam in enumerate(FAMILIES):
    rows=D[metric][fam];ax.plot([r['recall'] for r in rows],[r[metric]*factor for r in rows],label=LABELS[i],color=COLORS[i],marker=MARKERS[i],markersize=3,linewidth=1.4,linestyle='--' if DASH[i] else '-')
   ax.set_xlim((90,100) if panel==0 else(99,100));ax.set_xticks([90,92,95,98,100] if panel==0 else [99,99.2,99.5,99.8,100]);ax.set_xlabel('Recall@10 (%)');ax.set_ylabel(label);ax.grid(alpha=.2)
  fig.suptitle(f'SIFT1M: Recall@10 vs {label}');handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8);fig.tight_layout(rect=(0,.13,1,.94))
  for ext in ['svg','pdf','png']:fig.savefig(F/f'{name}.{ext}',dpi=220)
  plt.close(fig)
 else:svg(metric,name,label,factor)
(F/'plot_backend.json').write_text(json.dumps(dict(backend=backend,files=[x.name for x in F.glob('recall_*')]),indent=2)+'\n')
print('P0-B LR THREE CORE + ONE MECHANISM FIGURES COMPLETE',backend)
