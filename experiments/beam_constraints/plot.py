"""Standalone SVG curves; Python standard library only."""
from pathlib import Path
import json,html
R=Path(__file__).resolve().parent
rows=json.loads((R/'test.json').read_text());names=['MLP','max_skip_1','max_skip_2','keep_best_1'];colors=['#2563eb','#dc2626','#16a34a','#9333ea']
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1260" height="420" viewBox="0 0 1260 420"><rect width="1260" height="420" fill="white"/><g font-family="sans-serif" fill="#222"><text x="30" y="28" font-size="18">Frozen 17D MLP: beam constraints — SIFT1M, 300 test queries</text>']
for i,(key,label,lo,hi) in enumerate([('reads','SSD requests / query',45,110),('expansions','Expansions / query',55,120),('latency_us','Latency (us / query)',3000,9000)]):
 x0=65+i*415;y0=65;w=320;h=270
 def x(v):return x0+(v-98.8)/1.05*w
 def y(v):return y0+h-(v-lo)/(hi-lo)*h
 svg.append(f'<defs><clipPath id="c{i}"><rect x="{x0}" y="{y0}" width="{w}" height="{h}"/></clipPath></defs>')
 svg.append(f'<text x="{x0}" y="52" font-size="14">{html.escape(label)}</text>')
 for t in [99.,99.2,99.4,99.6,99.8]:
  xx=x(t);svg.append(f'<path d="M{xx} {y0}V{y0+h}" stroke="#ddd"/><text x="{xx}" y="{y0+h+20}" font-size="11" text-anchor="middle">{t:.1f}</text>')
 for j in range(5):
  v=lo+(hi-lo)*j/4;yy=y(v);svg.append(f'<path d="M{x0} {yy}H{x0+w}" stroke="#ddd"/><text x="{x0-8}" y="{yy+4}" font-size="11" text-anchor="end">{v:.0f}</text>')
 svg.append(f'<g clip-path="url(#c{i})">')
 for name,color in zip(names,colors):
  rs=sorted([r for r in rows if r['policy']==name and r['repeat']==0],key=lambda r:r['tau'])
  points=' '.join(f"{x(r['recall']):.2f},{y(r[key]):.2f}" for r in rs)
  svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
  for r in rs:svg.append(f'<circle cx="{x(r["recall"])}" cy="{y(r[key])}" r="3" fill="{color}"/>')
 svg.append(f'</g><text x="{x0+w/2}" y="385" font-size="12" text-anchor="middle">Recall@10 (%)</text>')
for j,(name,color) in enumerate(zip(names,colors)):
 xx=270+j*220;svg.append(f'<rect x="{xx}" y="400" width="18" height="4" fill="{color}"/><text x="{xx+25}" y="407" font-size="12">{name}</text>')
svg.append('</g></svg>');(R/'curves.svg').write_text(''.join(svg))
