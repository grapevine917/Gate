"""History/test trace analysis, including zero-overlap pairs in State Coverage.
Feature scale and state cutoffs use history/training only. Query percentile bins
are descriptive empirical distance bins, never deployment thresholds.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
E=load_module('e1_analysis',ROOT/'experiments/e1_similarity/analyze.py')
C=load_module('e2_common',ROOT/'experiments/e2_state_reuse/common.py')
def render(out,rows,states,group_rows):
 F=load_module('e1_figures',ROOT/'experiments/e1_similarity/plot_figures.py');F.OUT=out/'figures';F.OUT.mkdir(exist_ok=True)
 def state_panel(c,box,rs,key,title,xlabel):
  x,y,w,h=box;left=x+65;top=y+35;pw=w-95;bottom=y+h-75;ph=bottom-top
  values=[r[key]*100 for r in rs if r[key] is not None]
  bounds=[r[key+suffix]*100 for r in rs for suffix in ['_ci_low','_ci_high'] if r.get(key+suffix) is not None]
  low=min([0]+values+bounds) if 'coverage' in key else min(values+bounds,default=0)
  high=max(values+bounds,default=0);span=max(high-low,high*.03,1e-8)
  low=max(0,low-span*.08);high=min(100,high+span*.12)
  if high<=low:high=low+1
  xp=lambda i:left+pw*i/max(len(rs)-1,1);yp=lambda v:bottom-(v-low)/(high-low)*ph
  F.text(c,x+20,y+12,title,17,True)
  for v in np.linspace(low,high,5):F.segment(c,left,yp(v),left+pw,yp(v),(.88,.9,.92));F.text(c,x+3,yp(v)+4,f'{v:.3g}',11)
  F.segment(c,left,top,left,bottom,(.3,.3,.3));F.segment(c,left,bottom,left+pw,bottom,(.3,.3,.3))
  for i,r in enumerate(rs):
   F.text(c,xp(i)-19,bottom+23,r['group'].replace('–','-'),10)
   if r[key] is None:F.text(c,xp(i)-10,bottom-10,'N/A',10);continue
   xx=xp(i);yy=yp(r[key]*100);col=F.COLORS[0]
   if i and rs[i-1][key] is not None:F.segment(c,xp(i-1),yp(rs[i-1][key]*100),xx,yy,col,2)
   if r.get(key+'_ci_low') is not None:
    yl=yp(r[key+'_ci_low']*100);yh=yp(r[key+'_ci_high']*100);F.segment(c,xx,yl,xx,yh,col);F.segment(c,xx-3,yl,xx+3,yl,col);F.segment(c,xx-3,yh,xx+3,yh,col)
   F.rgb(c,*col);F.arc(c,xx,yy,3.5,0,2*np.pi);F.fill(c)
  F.text(c,left,bottom+47,xlabel,12)
 def draw(c):
  for i,(key,title) in enumerate([('agreement','Agreement (%)'),('skip_precision','Skip precision (%)'),('coverage','Coverage: all query pairs (%)'),('agreed_skip_coverage','Agreed Skip coverage (%)')]):state_panel(c,(20+(i%2)*600,110+(i//2)*340,580,315),states,key,title,'Normalized state L2 upper bound')
 F.canvas('state_coverage',1240,810,'State similarity and historical decision reuse','X: normalized state L2 upper bound | train-only cutoffs | all pairs include zero-overlap coverage',draw)
 def query_draw(c):
  for i,(key,title) in enumerate([('agreement','Agreement (%)'),('skip_precision','Skip precision (%)'),('coverage','Coverage: all query pairs (%)'),('agreed_skip_coverage','Agreed Skip coverage (%)')]):state_panel(c,(20+(i%2)*600,110+(i//2)*340,580,315),rows,key,title,'Query L2 distance percentile')
 F.canvas('query_similarity',1240,810,'Query similarity and historical decision reuse','Empirical L2 quantiles | independent history and test queries | target-query cluster 95% CI',query_draw)
 grouped=[{**r,**{k+'_ci_low':r.get(k+'_lo') for k in ['agreement','false_skip_rate','skip_precision','delta_p']},**{k+'_ci_high':r.get(k+'_hi') for k in ['agreement','false_skip_rate','skip_precision','delta_p']}} for r in group_rows]
 def joint_draw(c):
  for i,(key,title) in enumerate([('agreement','Agreement (%)'),('false_skip_rate','False Skip Rate vs model (%)'),('skip_precision','Skip precision (%)'),('delta_p','Absolute score gap (pp)')]):state_panel(c,(20+(i%2)*600,110+(i//2)*340,580,315),grouped,key,title,'G1 Q+/S+; G2 Q+/S-; G3 Q-/S+; G4 Q-/S-')
 F.canvas('joint_groups',1240,810,'Query and state similarity groups','Same node only | train-only group cutoffs | False Skip is relative to model decisions, not ground truth',joint_draw)

def main(a):
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 h=dict(np.load(a.history));t=dict(np.load(a.test));q=np.load(a.query_vectors);norm=np.load(a.normalization)
 hi=np.sort(h.get('all_query_ids',np.unique(h['query_id'])));ti=np.sort(t.get('all_query_ids',np.unique(t['query_id'])))
 assert len(np.unique(hi))==len(hi) and len(np.unique(ti))==len(ti)
 assert norm['mean'].shape==(17,) and norm['std'].shape==(17,) and np.all(norm['std']>=0)
 assert not np.intersect1d(hi,ti).size,'History/test IDs overlap'
 assert not set(map(bytes,q[hi]))&set(map(bytes,q[ti])),'History/test vectors overlap'
 for trace in [h,t]:
  ix=np.flatnonzero(trace['model_invoked']);keys=np.c_[trace['query_id'][ix],trace['node_id'][ix]]
  assert len(np.unique(keys,axis=0))==len(keys),'Repeated same-query node decisions require an explicit aggregation policy'
 dist=E.l2(q[hi],q[ti]);metrics=E.metrics(h,t,hi,ti);edges,rows,near,perquery=E.groups(metrics,dist)
 for name,data in [('query_distance_bins',rows),('query_nearest_rank_bins',near),('query_per_target_quantiles',perquery)]:E.csvsave(out/(name+'.csv'),data)
 left,right=C.node_pairs(h,t);tl,tr=C.node_pairs(h)
 scale=norm['std']+1e-8
 ds=np.linalg.norm((h['state_features'][left].astype(float)-t['state_features'][right])/scale,axis=1)
 train_ds=np.linalg.norm((h['state_features'][tl].astype(float)-h['state_features'][tr])/scale,axis=1)
 cutoffs=np.unique(np.quantile(train_ds,[0,.1,.25,.5,.75,.9,.99,1]));thresholds=np.r_[cutoffs,np.inf]
 ri=np.searchsorted(hi,h['query_id'][left]);rj=np.searchsorted(ti,t['query_id'][right]);hs=h['model_decision'][left].astype(bool);ts=t['model_decision'][right].astype(bool)
 qcut=float(np.median(np.linalg.norm(q[h['query_id'][tl]].astype(float)-q[h['query_id'][tr]].astype(float),axis=1)))
 scut=float(np.median(train_ds));data=C.pair_data(h,t,left,right,norm['mean'],norm['std'],q,0,qcut,scut)
 group_rows=[C.summarize(data,data['group']==i,f'G{i}') for i in range(1,5)];E.csvsave(out/'joint_groups.csv',group_rows)
 states=[];joint=[];nshape=(len(hi),len(ti));den=metrics['target_count'][None,:]
 for k,cut in enumerate(thresholds):
  take=ds<=cut;common=np.zeros(nshape,np.int32);skip=common.copy();both=common.copy();equal=common.copy()
  for array,mask in [(common,take),(skip,take&hs),(both,take&hs&ts),(equal,take&(hs==ts))]:np.add.at(array,(ri[mask],rj[mask]),1)
  m=dict(common=common,source_skip=skip,both_skip=both,equal=equal,target_skip=common-equal-skip+2*both,
         coverage=np.divide(skip,den,out=np.zeros(nshape,float),where=den>0),agreed_skip_coverage=np.divide(both,den,out=np.zeros(nshape,float),where=den>0),
         agreement=np.divide(equal,common,out=np.full(nshape,np.nan),where=common>0),skip_precision=np.divide(both,skip,out=np.full(nshape,np.nan),where=skip>0))
  label=f'{cut:.3g}' if np.isfinite(cut) else 'all'
  row=E.aggregate(m,np.ones(nshape,bool),label,dist);row['state_l2_max']=float(cut) if np.isfinite(cut) else None;states.append(row)
  for j in range(5):
   mask=(dist>=edges[j])&((dist<edges[j+1]) if j<4 else dist<=edges[j+1]);r=E.aggregate(m,mask,f'{j+1}:{label}',dist);r['query_bin']=rows[j]['group'];r['state_l2_max']=row['state_l2_max'];joint.append(r)
  if k==len(thresholds)-1:
   assert np.array_equal(skip,metrics['source_skip']) and np.array_equal(common,metrics['common'])
 E.csvsave(out/'state_coverage_cumulative.csv',states);E.csvsave(out/'query_state_coverage.csv',joint)
 # Independent set-based check on observed event pairs plus randomly selected zero-overlap pairs.
 rng=np.random.default_rng(20261019)
 for i,j in np.vstack([np.c_[ri[:50],rj[:50]],np.c_[rng.integers(len(hi),size=50),rng.integers(len(ti),size=50)]]):
  ha=np.flatnonzero((h['query_id']==hi[i])&h['model_invoked']);ta=np.flatnonzero((t['query_id']==ti[j])&t['model_invoked']);hd={int(h['node_id'][x]):x for x in ha};td={int(t['node_id'][x]):x for x in ta}
  n=sum(bool(h['model_decision'][hd[node]]) and np.linalg.norm((h['state_features'][hd[node]].astype(float)-t['state_features'][td[node]])/scale)<=scut for node in hd.keys()&td.keys())
  fast=int(np.count_nonzero((ri==i)&(rj==j)&hs&(ds<=scut)));assert n==fast
 summary=dict(history_queries=len(hi),test_queries=len(ti),query_pairs=len(hi)*len(ti),same_node_events=len(left),train_calibration_events=len(tl),state_cutoffs=cutoffs.tolist(),train_query_median=qcut,train_state_median=scut,normalization=str(Path(a.normalization).resolve()),zero_overlap_in_coverage=True,no_test_scaler_fit=True,verification='100 independent Python-set checks; infinite state threshold exactly recovers original Query Coverage',state_coverage=states,joint_groups=group_rows)
 (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
 render(out,rows,states,group_rows)
 print(json.dumps({k:summary[k] for k in ['query_pairs','same_node_events','verification']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for x in ['history','test','query-vectors','normalization','output']:p.add_argument('--'+x,required=True)
 main(p.parse_args())
