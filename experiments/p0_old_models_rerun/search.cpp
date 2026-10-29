#include "common_includes.h"
#include "disk_utils.h"
#include "pq_flash_index.h"
#include "online_gate.h"
#include "online_termination.h"
#include "linux_aligned_file_reader.h"
#include <memory>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <map>
#ifdef QUERY_U8
using QueryType=uint8_t;
#else
using QueryType=float;
#endif
uint64_t proc_kb(const std::string& key){std::ifstream f("/proc/self/status");std::string line;while(std::getline(f,line)){std::istringstream s(line);std::string k;uint64_t v;if(s>>k>>v && k==key+":")return v;}throw std::runtime_error("Missing process memory statistic");}
class MeteredReader:public LinuxAlignedFileReader{
public:std::atomic<uint64_t>requests{0},bytes{0};
 void read(std::vector<AlignedRead>&rs,IOContext&ctx,bool async=false)override{uint64_t n=0;for(auto&r:rs)n+=r.len;requests.fetch_add(rs.size(),std::memory_order_relaxed);bytes.fetch_add(n,std::memory_order_relaxed);LinuxAlignedFileReader::read(rs,ctx,async);}
};
uint64_t physical_bytes(){std::ifstream f("/proc/self/io");std::string k;uint64_t v;while(f>>k>>v)if(k=="read_bytes:")return v;throw std::runtime_error("Missing Linux physical IO accounting");}
struct Config{std::string key,model;int kind,L,rank,exp;float tau,et;};
int main(int argc,char**argv){
 if(argc!=9){std::cerr<<"index query gt models vectors configs output repeat\n";return 2;}
 QueryType*q=nullptr;float*gd=nullptr;uint32_t*gt=nullptr;size_t nq,d,ad,ng,dg;
 diskann::load_aligned_bin<QueryType>(argv[2],q,nq,d,ad);diskann::load_truthset(argv[3],gt,gd,ng,dg);if(nq!=ng)return 3;
 auto reader=std::make_shared<MeteredReader>();std::shared_ptr<AlignedFileReader>base_reader=reader;diskann::PQFlashIndex<QueryType,uint32_t>index(base_reader,diskann::Metric::L2);if(index.load(4,argv[1]))return 4;
 std::vector<uint32_t>nodes;index.cache_bfs_levels(10000,nodes);index.load_cache_list(nodes);
 std::ifstream input(argv[6]);std::vector<Config>configs;Config c;while(input>>c.key>>c.kind>>c.model>>c.L>>c.tau>>c.et>>c.rank>>c.exp)configs.push_back(c);if(configs.empty())return 5;
 std::map<std::string,std::unique_ptr<diskann::OnlineGate>>models;std::unique_ptr<diskann::OnlineGate>et;
 for(auto&cc:configs){
  if((cc.kind==2||cc.kind==4)&&!models.count(cc.model))models[cc.model]=std::make_unique<diskann::OnlineGate>(std::string(argv[4])+"/"+cc.model+".bin",cc.model=="qv"?argv[5]:"",0);
  if((cc.kind==3||cc.kind==4)&&!et)et=std::make_unique<diskann::OnlineGate>(std::string(argv[4])+"/et.bin","",1.1);
 }
 for(auto&cc:configs){
  auto*gate=(cc.kind==2||cc.kind==4)?models.at(cc.model).get():nullptr;auto*termination=(cc.kind==3||cc.kind==4)?et.get():nullptr;if(gate)gate->threshold=cc.tau;if(termination)termination->threshold=cc.et;diskann::OnlineHeuristic heuristic{cc.rank,cc.exp};
  auto search=[&](size_t i,diskann::QueryStats&s,uint64_t*ids,float*ds,uint32_t*calls){
   diskann::online_gate=gate;diskann::online_heuristic=cc.kind==1?&heuristic:nullptr;diskann::online_termination=termination;diskann::online_miss_only=true;diskann::online_model_calls=calls;
   index.cached_beam_search(q+i*ad,10,cc.L,ids,ds,4,std::numeric_limits<uint32_t>::max(),false,&s);
   diskann::online_gate=nullptr;diskann::online_heuristic=nullptr;diskann::online_termination=nullptr;diskann::online_miss_only=false;diskann::online_model_calls=nullptr;
  };
  size_t warm=std::min(nq,size_t(128));
  #pragma omp parallel for schedule(static) num_threads(4)
  for(size_t j=0;j<warm;++j){mkl_set_num_threads_local(1);uint64_t ids[10];float ds[10];diskann::QueryStats s;search(j*nq/warm,s,ids,ds,nullptr);}
  std::vector<std::array<uint64_t,10>>results(nq);std::vector<diskann::QueryStats>stats(nq);std::vector<double>us(nq);std::vector<uint32_t>calls(nq);
  auto rb=proc_kb("VmRSS"),sb=proc_kb("VmSwap");if(sb)throw std::runtime_error("Swap before timed search");reader->requests=0;reader->bytes=0;uint64_t pb=physical_bytes();
  auto begin=std::chrono::steady_clock::now();
  #pragma omp parallel for schedule(static) num_threads(4)
  for(size_t i=0;i<nq;++i){mkl_set_num_threads_local(1);float ds[10];auto t=std::chrono::steady_clock::now();search(i,stats[i],results[i].data(),ds,&calls[i]);us[i]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count();}
  double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();auto pd=physical_bytes()-pb,ra=proc_kb("VmRSS"),sa=proc_kb("VmSwap");if(sa)throw std::runtime_error("Swap during timed search");
  std::string p=std::string(argv[7])+"/"+cc.key+"_r"+argv[8];std::ofstream csv(p+".csv"),ids(p+".ids",std::ios::binary),meta(p+".meta.json");
  csv<<"query,hits,reads,expansions,skips,gate_evaluated,et_evaluated,et_stopped,model_calls,latency_us,io_us,io_rounds\n";
  for(size_t i=0;i<nq;++i){unsigned h=0;for(auto id:results[i])for(unsigned j=0;j<10;++j)if(id==gt[i*dg+j]){++h;break;}auto&s=stats[i];csv<<std::setprecision(12)<<i<<','<<h<<','<<s.n_ios<<','<<s.n_expanded<<','<<s.n_gate_skipped<<','<<s.n_gate_evaluated<<','<<s.n_et_evaluated<<','<<s.n_et_stopped<<','<<calls[i]<<','<<us[i]<<','<<s.io_us<<','<<s.n_hops<<'\n';ids.write((char*)results[i].data(),80);}
  meta<<std::setprecision(14)<<"{\"queries\":"<<nq<<",\"wall_seconds\":"<<sec<<",\"qps\":"<<double(nq)/sec<<",\"reader_requests\":"<<reader->requests.load()<<",\"reader_bytes\":"<<reader->bytes.load()<<",\"linux_read_bytes\":"<<pd<<",\"rss_before_kib\":"<<rb<<",\"rss_after_kib\":"<<ra<<",\"peak_rss_kib\":"<<proc_kb("VmHWM")<<",\"swap_before_kib\":"<<sb<<",\"swap_after_kib\":"<<sa<<"}\n";
  std::cout<<"CONFIG_DONE "<<cc.key<<std::endl;
 }
 diskann::aligned_free(q);delete[]gt;delete[]gd;
}
