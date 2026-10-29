#include "common_includes.h"
#include "disk_utils.h"
#include "pq_flash_index.h"
#include "online_gate.h"
#include "online_termination.h"
#include "gate_trace.h"
#include "linux_aligned_file_reader.h"
#include <fstream>
#include <random>
#include <memory>
int main(int argc,char**argv){
 if(argc!=8)return 2;
 float*q=nullptr,*gd=nullptr;uint32_t*gt=nullptr;size_t nq,d,ad,ng,dg;
 diskann::load_aligned_bin<float>(argv[2],q,nq,d,ad);diskann::load_truthset(argv[3],gt,gd,ng,dg);if(nq!=ng)return 3;
 std::shared_ptr<AlignedFileReader>reader=std::make_shared<LinuxAlignedFileReader>();diskann::PQFlashIndex<float,uint32_t>index(reader,diskann::Metric::L2);if(index.load(4,argv[1]))return 4;
 std::vector<uint32_t>nodes;index.cache_bfs_levels(10000,nodes);index.load_cache_list(nodes);diskann::OnlineGate gate(argv[4],"",0);
 std::ifstream configs(argv[5]);unsigned L,cap=std::stoul(argv[7]);float tau;
 std::ofstream out(argv[6],std::ios::binary);
 while(configs>>L>>tau){
  gate.threshold=tau;std::vector<std::vector<float>>rows(nq);
  #pragma omp parallel for schedule(dynamic,1) num_threads(4)
  for(size_t qi=0;qi<nq;++qi){
   mkl_set_num_threads_local(1);diskann::GateTrace trace;diskann::TerminationTrace term;diskann::QueryStats stats;uint64_t ids[10];float ds[10];
   diskann::online_gate=&gate;diskann::online_miss_only=true;diskann::gate_trace=&trace;diskann::termination_trace=&term;
   index.cached_beam_search(q+qi*ad,10,L,ids,ds,4,std::numeric_limits<uint32_t>::max(),false,&stats);
   diskann::online_gate=nullptr;diskann::online_miss_only=false;diskann::gate_trace=nullptr;diskann::termination_trace=nullptr;
   if(term.scored_nodes.size()!=trace.expansions.size())throw std::runtime_error("prefix count mismatch");
   for(size_t j=0;j<trace.expansions.size();++j)if(term.scored_nodes[j]!=trace.expansions[j].node)throw std::runtime_error("prefix order mismatch");
   unsigned final_hits=0;for(auto id:ids)for(unsigned z=0;z<10;++z)if(id==gt[qi*dg+z]){++final_hits;break;}
   std::vector<size_t>bi(term.boundaries.size());std::iota(bi.begin(),bi.end(),0);std::mt19937 rng(20261021+qi*31+L*997);std::shuffle(bi.begin(),bi.end(),rng);if(bi.size()>cap)bi.resize(cap);std::sort(bi.begin(),bi.end());
   for(auto j:bi){const auto&b=term.boundaries[j];std::vector<diskann::Neighbor>prefix;
    for(unsigned k=0;k<b.completed;++k)prefix.emplace_back(trace.expansions[k].node,trace.expansions[k].exact);
    if(prefix.size()<10)throw std::runtime_error("invalid boundary");
    std::partial_sort(prefix.begin(),prefix.begin()+10,prefix.end());unsigned hits=0;
    for(unsigned k=0;k<10;++k)for(unsigned z=0;z<10;++z)if(prefix[k].id==gt[qi*dg+z]){++hits;break;}
    // 23 columns: query, current recall, gate tau, final recall, total run reads,
    // completed expansions, live state[17]. Future quantities are labels only.
    auto&r=rows[qi];r.insert(r.end(),{float(qi),float(hits)/10,tau,float(final_hits)/10,float(stats.n_ios),float(b.completed)});
    r.insert(r.end(),b.state.begin(),b.state.end());
   }
  }
  for(auto&r:rows)out.write((char*)r.data(),r.size()*4);out.flush();std::cout<<"COLLECTED "<<L<<" "<<tau<<std::endl;
 }
 diskann::aligned_free(q);delete[]gt;delete[]gd;
}
