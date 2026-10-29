"""Standalone PNG/SVG/PDF scientific charts using installed native Cairo."""
import sys,math
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;OUT=R/'figures'
sys.path.insert(0,str(R.parent/'sift1m'))
from plot_online_curves import image_surface,svg_surface,pdf_surface,create,rgb,paint,text,segment,arc,fill,rectangle,png,finish,destroy,surface_destroy
COLORS=[(.16,.43,.68),(.83,.35,.16),(.18,.58,.39),(.55,.36,.68),(.48,.52,.56)]
def canvas(name,w,h,title,subtitle,draw):
 for ext in ['png','svg','pdf']:
  p=OUT/f'{name}.{ext}';s=image_surface(0,w,h) if ext=='png' else svg_surface(str(p).encode(),w,h) if ext=='svg' else pdf_surface(str(p).encode(),w,h)
  c=create(s);rgb(c,1,1,1);paint(c);text(c,35,38,title,23,True);text(c,35,65,subtitle,13);draw(c)
  if ext=='png':assert png(s,str(p).encode())==0
  finish(s);destroy(c);surface_destroy(s)
def panel(c,box,rows,key,title,series=None):
 x,y,w,h=box;left=x+58;top=y+35;pw=w-85;ph=h-110;bottom=top+ph
 series=series or [('MLP',rows,COLORS[0])]
 vals=[r[key]*100 for _,rs,_ in series for r in rs if r[key] is not None]
 bounds=[r[key+v]*100 for _,rs,_ in series for r in rs for v in ['_ci_low','_ci_high'] if r.get(key+v) is not None]
 lo=max(0,min(vals+bounds)*.94);hi=min(100,max(vals+bounds)*1.06+.05)
 if hi-lo<.1:hi=lo+.1
 xp=lambda i:left+pw*i/max(len(rows)-1,1);yp=lambda v:bottom-(v-lo)/(hi-lo)*ph
 text(c,x+20,y+12,title,17,True)
 for v in np.linspace(lo,hi,5):segment(c,left,yp(v),left+pw,yp(v),(.88,.9,.92));text(c,x+5,yp(v)+4,f'{v:.2f}',11)
 segment(c,left,top,left,bottom,(.3,.3,.3));segment(c,left,bottom,left+pw,bottom,(.3,.3,.3))
 for i,r in enumerate(rows):text(c,xp(i)-24,bottom+23,r['group'].replace('rank ','').replace('–','-'),10)
 text(c,left,bottom+47,'Distance percentile' if '%' in rows[0]['group'] else 'Historical neighbour rank per target',12)
 for name,rs,col in series:
  for i,r in enumerate(rs):
   if r[key] is None:
    text(c,xp(i)-12,bottom-10,'N/A',11)
    continue
   yy=yp(r[key]*100);xx=xp(i)
   if i and rs[i-1][key] is not None:segment(c,xp(i-1),yp(rs[i-1][key]*100),xx,yy,col,2)
   if r.get(key+'_ci_low') is not None:
    yl=yp(r[key+'_ci_low']*100);yh=yp(r[key+'_ci_high']*100);segment(c,xx,yl,xx,yh,col,1.5);segment(c,xx-4,yl,xx+4,yl,col);segment(c,xx-4,yh,xx+4,yh,col)
   rgb(c,*col);arc(c,xx,yy,4,0,2*math.pi);fill(c)
def render_all(a,results,dist):
 OUT.mkdir(exist_ok=True);rows=a['MLP']['distance_bins'];near=a['MLP']['nearest_rank_bins']
 def draw(c):
  for i,(key,title) in enumerate([('agreement','Decision agreement A (%)'),('skip_precision','Directed skip precision (%)'),('coverage','Potential decision coverage (%)'),('agreed_skip_coverage','Agreed Skip overlap / target decisions (%)')]):panel(c,(20+(i%2)*600,105+(i//2)*340,580,315),rows,key,title)
 canvas('similarity_reuse_potential',1240,800,'Query similarity and historical Skip reuse potential','SIFT1M | independent state-only MLP | 1,000,000 history-to-test pairs | 95% target-query cluster bootstrap CI',draw)
 def draw(c):
  for i,(key,title) in enumerate([('agreement','Decision agreement (%)'),('skip_precision','Skip precision (%)'),('coverage','Potential decision coverage (%)')]):panel(c,(10+i*485,110,480,345),near,key,title)
 canvas('per_query_neighbours',1480,490,'Nearest historical queries for each target query','Each test query contributes its own neighbour ranks; no pooled-distance dominance.',draw)
 def draw(c):
  for j,(name,col) in enumerate(zip(['MLP','LR','XGBoost'],COLORS)):segment(c,50+j*220,93,80+j*220,93,col,3);text(c,90+j*220,98,name,14)
  ss=[(name,a[name]['distance_bins'],col) for name,col in zip(['MLP','LR','XGBoost'],COLORS)]
  for i,(key,title) in enumerate([('agreement','Decision agreement (%)'),('skip_precision','Skip precision (%)'),('coverage','Potential decision coverage (%)')]):panel(c,(10+i*485,130,480,345),rows,key,title,ss)
 canvas('model_controls',1480,510,'MLP / LR / XGBoost independent-trajectory controls','Each model uses its own validation-locked threshold and live search states.',draw)
 def draw(c):
  hist,edges=np.histogram(dist,bins=70);x,y,w,h=80,150,500,270
  text(c,50,110,'Measured history-to-test L2 distance distribution',16,True)
  for i,v in enumerate(hist):rgb(c,*COLORS[0]);rectangle(c,x+i*w/len(hist),y+h-v/hist.max()*h,w/len(hist)-1,v/hist.max()*h);fill(c)
  for v in np.linspace(edges[0],edges[-1],5):text(c,x+(v-edges[0])/(edges[-1]-edges[0])*w-15,y+h+25,f'{v:.0f}',12)
  text(c,x+160,y+h+50,'Query L2 distance',14)
  rs=[]
  for r in rows:rs.append({'group':r['group'],'v':r['mean_common_nodes']/100,'u':r['mean_source_skip_common']/100})
  skiprows=[{**r,'v':r['u']} for r in rs]
  panel(c,(640,110,580,360),rs,'v','Mean nodes per Query pair', [('Common decision nodes',rs,COLORS[0]),('Historical Skip overlap',skiprows,COLORS[1])])
  # The generic percent axis multiplies by100: v already divided by100 above.
  text(c,680,477,'Blue: common nodes; orange: historical Skip overlap.',12)
 canvas('distance_and_overlap',1240,510,'Query distance and actual shared decision nodes','Quantile cut points are measured from all sampled pairs, not absolute normalized-distance thresholds.',draw)
 def draw(c):
  names=list(results['profile']);keys=['distance_us','model_us','memory_us','ssd_us','bookkeeping_us'];p=results['profile'];maxv=max(sum(p[n][k] for k in keys) for n in names)
  for i,k in enumerate(keys):rgb(c,*COLORS[i]);rectangle(c,35+i*235,90,15,15);fill(c);text(c,55+i*235,103,k[:-3],13)
  left,top,pw,ph=95,160,1060,300
  for v in np.linspace(0,maxv*1.1,6):yy=top+ph-v/(maxv*1.1)*ph;segment(c,left,yy,left+pw,yy,(.88,.9,.92));text(c,35,yy+4,f'{v:.0f}',12)
  for j,n in enumerate(names):
   base=0;x=left+j*pw/len(names)+35
   for i,k in enumerate(keys):
    v=p[n][k];height=v/(maxv*1.1)*ph;rgb(c,*COLORS[i]);rectangle(c,x,top+ph-(base+v)/(maxv*1.1)*ph,120,height);fill(c);base+=v
   text(c,x,top+ph+25,n,14)
  text(c,95,top+ph+57,'Instrumented microseconds / query. Hardware memory stalls remain attributed to the executing region.',13)
 canvas('cost_breakdown',1240,550,'Five exclusive software cost regions','Separate profiling run; use uninstrumented timing for absolute latency. SSD = synchronous reader time, not physical device time.',draw)
