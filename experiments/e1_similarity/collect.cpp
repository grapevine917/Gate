#include "common_includes.h"
#include "disk_utils.h"
#include "pq_flash_index.h"
#include "online_gate.h"
#include "cost_profile.h"
#include "linux_aligned_file_reader.h"
#include <fstream>
#include <iomanip>
#include <memory>
int main(int argc,char**argv) {
 if(argc!=9){std::cerr<<"index query gt model|- tau marks|- prefix mode(trace|time|profile)\n";return 2;}
 float*q=nullptr,*gd=nullptr;uint32_t*gt=nullptr;size_t nq,d,ad,ng,dg;
 diskann::load_aligned_bin<float>(argv[2],q,nq,d,ad);diskann::load_truthset(argv[3],gt,gd,ng,dg);
 if(nq!=ng||dg<10)throw std::runtime_error("GT mismatch");
 std::shared_ptr<AlignedFileReader> reader=std::make_shared<LinuxAlignedFileReader>();diskann::PQFlashIndex<float,uint32_t> index(reader,diskann::Metric::L2);
 if(index.load(4,argv[1]))return 3;std::vector<uint32_t> nodes;index.cache_bfs_levels(10000,nodes);index.load_cache_list(nodes);
 std::unique_ptr<diskann::OnlineGate> model;std::unique_ptr<diskann::OnlineNodeMarks> marks;
 if(std::string(argv[4])!="-")model.reset(new diskann::OnlineGate(argv[4],"",std::stof(argv[5])));
 if(std::string(argv[6])!="-")marks.reset(new diskann::OnlineNodeMarks(argv[6]));
 std::string prefix=argv[7],mode=argv[8];bool tracing=mode=="trace",profiling=mode=="profile";
 std::ofstream csv(prefix+".csv"),trace;
 csv<<"query,hits,reads,expansions,cache_hits,model_calls,skips,pq_comparisons,total_us,io_us,distance_us,model_us,memory_us,ssd_us,bookkeeping_us\n";
 if(tracing){trace.open(prefix+".trace",std::ios::binary);std::ofstream nc(prefix+".cache.bin",std::ios::binary);nc.write((char*)nodes.data(),nodes.size()*4);}
 // Warm the same static node/OS caches; no history lookup or reuse exists here.
 auto search=[&](size_t i,diskann::QueryStats& s,uint64_t*ids,float*ds){
  diskann::online_gate=model.get();diskann::online_node_marks=marks.get();diskann::online_miss_only=true;
  index.cached_beam_search(q+i*ad,10,100,ids,ds,4,std::numeric_limits<uint32_t>::max(),false,&s);
  diskann::online_gate=nullptr;diskann::online_node_marks=nullptr;diskann::online_miss_only=false;
 };
 #pragma omp parallel for schedule(static) num_threads(4)
 for(size_t i=0;i<nq;++i){mkl_set_num_threads_local(1);uint64_t ids[10];float ds[10];diskann::QueryStats s;search(i,s,ids,ds);}
 std::vector<std::string> rows(nq);std::vector<std::vector<float>> audits(nq);std::vector<std::array<uint64_t,10>> results(nq);
 #pragma omp parallel for schedule(static) num_threads(4)
 for(size_t i=0;i<nq;++i){
  mkl_set_num_threads_local(1);float ds[10];diskann::QueryStats s;uint32_t calls=0;diskann::online_model_calls=&calls;
  if(tracing)diskann::online_audit=&audits[i];
  diskann::E1Cost cost;if(profiling)diskann::e1_cost=&cost;
  auto begin=std::chrono::steady_clock::now();if(profiling)cost.last=begin;
  search(i,s,results[i].data(),ds);
  if(profiling)cost.change(4);
  double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();
  // Use the single profile endpoint; a second clock read may be preempted.
  if(profiling)us=std::accumulate(cost.us.begin(),cost.us.end(),0.0);
  diskann::e1_cost=nullptr;diskann::online_model_calls=nullptr;diskann::online_audit=nullptr;
  unsigned hits=0;for(auto id:results[i])for(size_t j=0;j<10;++j)if(id==gt[i*dg+j]){++hits;break;}
  std::ostringstream out;out<<std::setprecision(12)<<i<<','<<hits<<','<<s.n_ios<<','<<s.n_expanded<<','<<s.n_cache_hits<<','<<calls<<','<<s.n_gate_skipped<<','<<s.n_cmps<<','<<us<<','<<s.io_us;
  for(auto x:cost.us)out<<','<<x;out<<'\n';rows[i]=out.str();
 }
 std::ofstream idsout(prefix+".ids",std::ios::binary);
 for(size_t i=0;i<nq;++i){csv<<rows[i];idsout.write((char*)results[i].data(),80);
  if(tracing){uint32_t row=i,count=audits[i].size()/19;trace.write((char*)&row,4);trace.write((char*)&count,4);trace.write((char*)audits[i].data(),audits[i].size()*4);}
 }
 diskann::aligned_free(q);delete[]gt;delete[]gd;
}
