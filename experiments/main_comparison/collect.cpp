#include "common_includes.h"
#include "disk_utils.h"
#include "pq_flash_index.h"
#include "online_gate.h"
#include "online_termination.h"
#include "gate_trace.h"
#include "linux_aligned_file_reader.h"
#include <fstream>
#include <random>
#include <unordered_map>
#include <set>
#include <memory>
int main(int argc,char**argv){
 if(argc!=8){std::cerr<<"index query gt dummy_model L_file prefix cap\n";return 2;}
 float*q=nullptr,*gd=nullptr;uint32_t*gt=nullptr;size_t nq,d,ad,ng,dg;
 diskann::load_aligned_bin<float>(argv[2],q,nq,d,ad);diskann::load_truthset(argv[3],gt,gd,ng,dg);if(nq!=ng)return 3;
 std::shared_ptr<AlignedFileReader>reader=std::make_shared<LinuxAlignedFileReader>();diskann::PQFlashIndex<float,uint32_t>index(reader,diskann::Metric::L2);if(index.load(4,argv[1]))return 4;
 std::vector<uint32_t>nodes;index.cache_bfs_levels(10000,nodes);index.load_cache_list(nodes);diskann::OnlineGate dummy(argv[4],"",0);
 std::ifstream ls(argv[5]);unsigned L,cap=std::stoul(argv[7]);
 std::ofstream p(std::string(argv[6])+".prune.bin",std::ios::binary),e(std::string(argv[6])+".et.bin",std::ios::binary);
 while(ls>>L){
  std::vector<std::vector<float>>pr(nq),et(nq);
  #pragma omp parallel for schedule(dynamic,1) num_threads(4)
  for(size_t qi=0;qi<nq;++qi){
   mkl_set_num_threads_local(1);diskann::GateTrace trace;diskann::TerminationTrace term;std::vector<float>audit;diskann::QueryStats stats;uint64_t ids[10];float ds[10];
   diskann::online_gate=&dummy;diskann::online_miss_only=true;diskann::gate_trace=&trace;diskann::termination_trace=&term;diskann::online_audit=&audit;
   index.cached_beam_search(q+qi*ad,10,L,ids,ds,4,std::numeric_limits<uint32_t>::max(),false,&stats);
   diskann::online_gate=nullptr;diskann::online_miss_only=false;diskann::gate_trace=nullptr;diskann::termination_trace=nullptr;diskann::online_audit=nullptr;
   std::unordered_map<uint32_t,const diskann::GateExpansion*>by;
   for(auto&x:trace.expansions)by[x.node]=&x;
   std::vector<size_t>ix;for(size_t j=0;j<audit.size()/19;++j)if(audit[j*19+11]==0)ix.push_back(j);
   std::mt19937 rng(20261030+qi*31+L*997);std::shuffle(ix.begin(),ix.end(),rng);if(ix.size()>cap)ix.resize(cap);std::sort(ix.begin(),ix.end());
   for(auto j:ix){auto node=uint32_t(audit[j*19]);auto x=by.at(node);bool final=std::find(ids,ids+10,uint64_t(node))!=ids+10;pr[qi].push_back(float(qi));pr[qi].push_back(float(node));pr[qi].push_back(float(x->admitted>0||final));pr[qi].insert(pr[qi].end(),audit.begin()+j*19+1,audit.begin()+j*19+18);}
   if(term.scored_nodes.size()!=trace.expansions.size())throw std::runtime_error("prefix mismatch");
   for(size_t j=0;j<trace.expansions.size();++j)if(term.scored_nodes[j]!=trace.expansions[j].node)throw std::runtime_error("prefix order mismatch");
   std::vector<size_t>bi(term.boundaries.size());std::iota(bi.begin(),bi.end(),0);std::shuffle(bi.begin(),bi.end(),rng);if(bi.size()>cap)bi.resize(cap);std::sort(bi.begin(),bi.end());
   for(auto j:bi){const auto&b=term.boundaries[j];std::vector<diskann::Neighbor>prefix;
    for(unsigned k=0;k<b.completed;++k)prefix.emplace_back(trace.expansions[k].node,trace.expansions[k].exact);
    std::partial_sort(prefix.begin(),prefix.begin()+10,prefix.end());unsigned hits=0;
    for(unsigned k=0;k<10;++k)for(unsigned z=0;z<10;++z)if(prefix[k].id==gt[qi*dg+z]){++hits;break;}
    et[qi].push_back(float(qi));et[qi].push_back(float(hits)/10);et[qi].insert(et[qi].end(),b.state.begin(),b.state.end());
   }
  }
  for(size_t qi=0;qi<nq;++qi){p.write((char*)pr[qi].data(),pr[qi].size()*4);e.write((char*)et[qi].data(),et[qi].size()*4);}p.flush();e.flush();std::cout<<"COLLECTED L "<<L<<std::endl;
 }
 diskann::aligned_free(q);delete[]gt;delete[]gd;
}
