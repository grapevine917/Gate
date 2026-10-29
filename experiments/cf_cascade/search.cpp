#include "common_includes.h"
#include "disk_utils.h"
#include "pq_flash_index.h"
#include "online_gate.h"
#include "policy.h"
#include "linux_aligned_file_reader.h"
#include <fstream>
#include <iomanip>
#include <memory>
int main(int argc,char**argv){
 if(argc!=9){std::cerr<<"index query gt original_model cf_model configs output_dir repeat\n";return 2;}
 float*q=nullptr,*gd=nullptr;uint32_t*gt=nullptr;size_t nq,d,ad,ng,dg;
 diskann::load_aligned_bin<float>(argv[2],q,nq,d,ad);diskann::load_truthset(argv[3],gt,gd,ng,dg);
 if(nq!=ng||dg<10)throw std::runtime_error("GT mismatch");
 std::shared_ptr<AlignedFileReader> reader=std::make_shared<LinuxAlignedFileReader>();diskann::PQFlashIndex<float,uint32_t> index(reader,diskann::Metric::L2);
 if(index.load(4,argv[1]))return 3;std::vector<uint32_t>nodes;index.cache_bfs_levels(10000,nodes);index.load_cache_list(nodes);
 diskann::OnlineGate model(argv[4],"",0);diskann::CounterfactualModel cf(argv[5],0,0);
 std::ifstream input(argv[6]);std::string key;int kind;float tau,secondary;
 while(input>>key>>kind>>tau>>secondary){
  model.threshold=tau;std::string prefix=std::string(argv[7])+"/"+key+"_r"+argv[8];
  auto search=[&](size_t i,diskann::QueryStats&s,uint64_t*ids,float*ds,diskann::Cascade&c){
   c.model=&cf;c.kind=kind;c.secondary=secondary;diskann::cascade=&c;diskann::online_gate=&model;diskann::online_miss_only=true;
   index.cached_beam_search(q+i*ad,10,100,ids,ds,4,std::numeric_limits<uint32_t>::max(),false,&s);
   diskann::cascade=nullptr;diskann::online_gate=nullptr;diskann::online_miss_only=false;
  };
  #pragma omp parallel for schedule(static) num_threads(4)
  for(size_t i=0;i<nq;++i){mkl_set_num_threads_local(1);uint64_t ids[10];float ds[10];diskann::QueryStats s;diskann::Cascade c;search(i,s,ids,ds,c);}
  std::vector<std::string>rows(nq);std::vector<std::array<uint64_t,10>>results(nq);
  #pragma omp parallel for schedule(static) num_threads(4)
  for(size_t i=0;i<nq;++i){
   mkl_set_num_threads_local(1);float ds[10];diskann::QueryStats s;diskann::Cascade c;
   auto begin=std::chrono::steady_clock::now();search(i,s,results[i].data(),ds,c);
   double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();
   unsigned hits=0;for(auto id:results[i])for(size_t j=0;j<10;++j)if(id==gt[i*dg+j]){++hits;break;}
   if(c.first_skip+c.second_skip!=s.n_gate_skipped)throw std::runtime_error("skip accounting mismatch");
   if(kind && c.second_calls!=c.misses-c.first_skip)throw std::runtime_error("secondary support mismatch");
   std::ostringstream row;row<<std::setprecision(12)<<i<<','<<hits<<','<<s.n_ios<<','<<s.n_expanded<<','<<s.n_gate_skipped<<','<<c.first_skip<<','<<c.second_calls<<','<<c.second_skip<<','<<us<<'\n';rows[i]=row.str();
  }
  std::ofstream csv(prefix+".csv"),ids(prefix+".ids",std::ios::binary);csv<<"query,hits,reads,expansions,skips,first_skips,secondary_calls,secondary_skips,total_us\n";
  for(size_t i=0;i<nq;++i){csv<<rows[i];ids.write((char*)results[i].data(),80);}
  std::cout<<"CONFIG_DONE "<<key<<std::endl;
 }
 diskann::aligned_free(q);delete[]gt;delete[]gd;
}
