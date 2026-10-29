#include "common_includes.h"
#include "disk_utils.h"
#include "pq_flash_index.h"
#include "online_gate.h"
#include "reuse.h"
#include "cost_profile.h"
#include "linux_aligned_file_reader.h"
#include <fstream>
#include <iomanip>
#include <numeric>
int main(int argc,char**argv){
 if(argc!=16){std::cerr<<"index query gt model tau history neighbours kind qcut scut votes oracle mode tape|- prefix\n";return 2;}
 float*q=nullptr,*gd=nullptr;uint32_t*gt=nullptr;size_t nq,d,ad,ng,dg;diskann::load_aligned_bin<float>(argv[2],q,nq,d,ad);diskann::load_truthset(argv[3],gt,gd,ng,dg);if(nq!=ng||dg<10)throw std::runtime_error("GT shape");
 diskann::E3Bank bank;std::ifstream h(argv[6],std::ios::binary);h.read((char*)bank.mean,136);h.read((char*)bank.denom,136);uint32_t n;h.read((char*)&n,4);bank.records.resize(n);h.read((char*)bank.records.data(),n*sizeof(diskann::E3History));if(!h||sizeof(diskann::E3History)!=152||sizeof(diskann::E3Record)!=80)throw std::runtime_error("bank format");
 for(size_t i=0;i<n;){size_t j=i+1;while(j<n&&bank.records[j].node==bank.records[i].node)++j;bank.ranges[bank.records[i].node]={i,j};i=j;}
 std::vector<diskann::E3Context>contexts(nq);std::ifstream nn(argv[7]);
 bool plan=std::string(argv[13])=="plan",profile=std::string(argv[13])=="profile";
 for(auto&c:contexts){c.bank=&bank;c.kind=std::stoi(argv[8]);c.qcut=std::stod(argv[9]);c.scut=std::stod(argv[10]);c.votes=std::stoi(argv[11]);c.oracle=std::stoi(argv[12]);c.tau=std::stof(argv[5]);c.mode=plan?1:2;for(auto&v:c.neighbours)nn>>v;for(auto&v:c.query_dist)nn>>v;}if(!nn)throw std::runtime_error("neighbour file");
 if(!plan){std::ifstream f(argv[14],std::ios::binary);uint32_t count;f.read((char*)&count,4);if(count!=nq)throw std::runtime_error("tape queries");for(auto&c:contexts){uint32_t size;f.read((char*)&size,4);c.tape.resize(size);f.read((char*)c.tape.data(),size*80);}if(!f)throw std::runtime_error("tape truncated");}
 std::shared_ptr<AlignedFileReader>reader=std::make_shared<LinuxAlignedFileReader>();diskann::PQFlashIndex<float,uint32_t>index(reader,diskann::Metric::L2);if(index.load(4,argv[1]))return 3;std::vector<uint32_t>nodes;index.cache_bfs_levels(10000,nodes);index.load_cache_list(nodes);diskann::OnlineGate model(argv[4],"",std::stof(argv[5]));
 auto search=[&](size_t i,uint64_t*ids,float*ds,diskann::QueryStats&s){diskann::online_gate=&model;diskann::online_miss_only=true;index.cached_beam_search(q+i*ad,10,100,ids,ds,4,std::numeric_limits<uint32_t>::max(),false,&s);diskann::online_gate=nullptr;diskann::online_miss_only=false;};
 #pragma omp parallel for schedule(static) num_threads(4)
 for(size_t i=0;i<nq;++i){mkl_set_num_threads_local(1);uint64_t ids[10];float ds[10];diskann::QueryStats st;search(i,ids,ds,st);}
 std::vector<std::array<uint64_t,10>>outputs(nq);std::vector<std::string>rows(nq);
 #pragma omp parallel for schedule(static) num_threads(4)
 for(size_t i=0;i<nq;++i){
  mkl_set_num_threads_local(1);auto&c=contexts[i];diskann::e3_context=&c;uint32_t calls=0;diskann::online_model_calls=&calls;diskann::QueryStats st;float ds[10];diskann::E1Cost cost;
  auto begin=std::chrono::steady_clock::now();if(profile){cost.last=begin;diskann::e1_cost=&cost;}
  search(i,outputs[i].data(),ds,st);if(profile)cost.change(4);
  double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();if(profile)us=std::accumulate(cost.us.begin(),cost.us.end(),0.0);diskann::e1_cost=nullptr;
  diskann::e3_context=nullptr;diskann::online_model_calls=nullptr;
  if(!plan&&(c.cursor!=c.tape.size()||calls!=c.eligible-c.reused))throw std::runtime_error("tape cursor/model call mismatch");
  unsigned hits=0;for(auto id:outputs[i])for(size_t j=0;j<10;++j)if(id==gt[i*dg+j]){++hits;break;}
  std::ostringstream o;o<<std::setprecision(12)<<i<<','<<hits<<','<<st.n_ios<<','<<st.n_expanded<<','<<st.n_cache_hits<<','<<c.eligible<<','<<c.reused<<','<<c.eligible-c.reused<<','<<calls<<','<<c.false_skip<<','<<c.fresh_keep<<','<<us<<','<<st.io_us;for(auto v:cost.us)o<<','<<v;o<<'\n';rows[i]=o.str();
 }
 std::string prefix=argv[15];std::ofstream csv(prefix+".csv"),ids(prefix+".ids",std::ios::binary);csv<<"query,hits,reads,expansions,cache_hits,eligible,reused,actual_calls,physical_calls,false_skip,fresh_keep,total_us,io_us,distance_us,model_us,memory_us,ssd_us,bookkeeping_us\n";for(size_t i=0;i<nq;++i){csv<<rows[i];ids.write((char*)outputs[i].data(),80);}
 if(plan){std::ofstream f(prefix+".tape",std::ios::binary);uint32_t count=nq;f.write((char*)&count,4);for(auto&c:contexts){uint32_t size=c.tape.size();f.write((char*)&size,4);f.write((char*)c.tape.data(),size*80);}}
 std::ofstream cache(prefix+".cache.bin",std::ios::binary);cache.write((char*)nodes.data(),nodes.size()*4);std::cerr<<"COMPLETE "<<nq<<" mode="<<argv[13]<<std::endl;
 diskann::aligned_free(q);delete[]gt;delete[]gd;
}
