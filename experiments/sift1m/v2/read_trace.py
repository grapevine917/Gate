#!/usr/bin/env python3
"""Read LGTRACE2 blocks, or export one query's edges as CSV."""
import argparse,csv,struct
from pathlib import Path
import numpy as np
EXP=np.dtype([('node','<u4'),('degree','<u4'),('fresh','<u4'),('admitted','<u4'),('exact','<f4'),('cached','<u4')])
FIELDS=[('query_id','u4'),('step_id','u4'),('beam_step','u4'),('current_node_id','u4'),('neighbor_id','u4'),('current_distance_exact','f4'),('current_distance_pq','f4'),('neighbor_distance_pq','f4'),('neighbor_distance_exact','f4'),('current_best_distance','f4'),('current_best_exact_distance','f4'),('neighbor_rank','u4'),('neighbor_position','u4'),('queue_size','u4'),('visited_count','u4'),('flags','u4'),('neighbor_was_inserted','u4'),('neighbor_was_expanded','u4'),('neighbor_led_to_improvement','u4'),('neighbor_in_final_topk','u4'),('neighbor_improved_best_pq','u4'),('neighbor_improved_best_exact','u4')]
EDGE=np.dtype([(n,'<'+t) for n,t in FIELDS]); assert EDGE.itemsize==88
def blocks(path):
 with open(path,'rb') as f:
  assert f.read(8)==b'LGTRACE2','Requires LGTRACE2'
  while True:
   raw=f.read(20)
   if not raw: return
   assert len(raw)==20
   q,L,visited,n,m=struct.unpack('<5I',raw)
   eb=f.read(n*24); nb=f.read(m*88); assert len(eb)==n*24 and len(nb)==m*88
   yield q,L,visited,np.frombuffer(eb,dtype=EXP),np.frombuffer(nb,dtype=EDGE)
if __name__=='__main__':
 a=argparse.ArgumentParser(); a.add_argument('--trace',required=True); a.add_argument('--query-id',required=True,type=int); a.add_argument('--output',required=True); args=a.parse_args()
 for q,L,visited,ex,ed in blocks(args.trace):
  if q!=args.query_id: continue
  with open(args.output,'w',newline='') as f:
   w=csv.writer(f); w.writerow(EDGE.names)
   for r in ed: w.writerow([r[k].item() for k in EDGE.names])
  print(f'Exported query={q}, L={L}, expansions={len(ex)}, edges={len(ed)} to {args.output}'); break
 else: raise SystemExit('query_id not found')
