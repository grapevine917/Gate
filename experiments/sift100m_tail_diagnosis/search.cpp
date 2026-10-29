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
#include <sys/resource.h>
uint64_t proc_kb(const std::string& key){std::ifstream f("/proc/self/status");std::string line;while(std::getline(f,line)){std::istringstream s(line);std::string k;uint64_t v;if(s>>k>>v && k==key+":")return v;}throw std::runtime_error("Missing process memory statistic");}
class MeteredReader : public LinuxAlignedFileReader {
public:
 std::atomic<uint64_t> requests{0},bytes{0};
 void read(std::vector<AlignedRead>& rs, IOContext& ctx, bool async=false) override {
  uint64_t n=0;for(const auto& r:rs)n+=r.len;
  requests.fetch_add(rs.size(),std::memory_order_relaxed);bytes.fetch_add(n,std::memory_order_relaxed);
  LinuxAlignedFileReader::read(rs,ctx,async);
 }
};
const char* physical_source="proc_self_io";
uint64_t physical_bytes(){
 std::ifstream f("/proc/self/io");std::string k;uint64_t v;
 while(f>>k>>v)if(k=="read_bytes:")return v;
 // Linux task_io_get_inblock uses read_bytes >> 9. RUSAGE_SELF sums all threads.
 struct rusage usage{};
 if(getrusage(RUSAGE_SELF,&usage)!=0 || usage.ru_inblock<0)throw std::runtime_error("Missing kernel input accounting");
 physical_source="getrusage_ru_inblock_x512";
 return uint64_t(usage.ru_inblock)*512;
}
int main(int argc,char**argv){
 if(argc!=9){std::cerr<<"index query gt models base_vectors configs output repeat\n";return 2;}
 uint8_t*q=nullptr;float*gd=nullptr;uint32_t*gt=nullptr;size_t nq,d,ad,ng,dg;
 diskann::load_aligned_bin<uint8_t>(argv[2],q,nq,d,ad);diskann::load_truthset(argv[3],gt,gd,ng,dg);if(nq!=ng)return 3;
 auto reader=std::make_shared<MeteredReader>();std::shared_ptr<AlignedFileReader> base_reader=reader;diskann::PQFlashIndex<uint8_t,uint32_t>index(base_reader,diskann::Metric::L2);if(index.load(4,argv[1]))return 4;
 std::vector<uint32_t>nodes;index.cache_bfs_levels(10000,nodes);index.load_cache_list(nodes);
 std::ofstream cache(std::string(argv[7])+"/cache_nodes.bin",std::ios::binary);cache.write((char*)nodes.data(),nodes.size()*4);cache.close();
 std::ifstream probe(argv[6]);std::string first_key;int first_kind;probe>>first_key>>first_kind;
 const char*names[]={"","","lr","xgb","l100","state","qv","","l100","lr"};
 std::unique_ptr<diskann::OnlineGate> owned,et;
 if(first_kind>=2 && first_kind!=7)owned=std::make_unique<diskann::OnlineGate>(std::string(argv[4])+"/"+names[first_kind]+".bin",first_kind==6?argv[5]:"",0);
 if(first_kind>=7)et=std::make_unique<diskann::OnlineGate>(std::string(argv[4])+"/et.bin","",1.1);
 std::ifstream configs(argv[6]);std::string key;int kind,L,rank,expansions;float tau,et_tau;
 while(configs>>key>>kind>>L>>tau>>et_tau>>rank>>expansions){
  if(kind!=first_kind)throw std::runtime_error("one family per process");auto*gate=owned.get();if(gate)gate->threshold=tau;if(et)et->threshold=et_tau;diskann::OnlineHeuristic heuristic{rank,expansions};std::string prefix=std::string(argv[7])+"/"+key+"_r"+argv[8];
  std::vector<std::vector<uint8_t>> tapes(nq);
  std::vector<std::string> modes={"profile","full","replay"};
  if(std::stoi(argv[8])%2)modes={"profile","replay","full"};
  for(const auto& mode:modes) {
  if(kind==0 && mode=="replay")continue;
  std::string mode_prefix=prefix+"_"+mode;
  std::vector<diskann::TailProfile> profiles(nq);
  bool formal=false;
  auto search=[&](size_t i,diskann::QueryStats&s,uint64_t*ids,float*ds,uint32_t*calls){
   diskann::online_gate=mode=="replay"?nullptr:gate;diskann::online_heuristic=(kind==1 && mode!="replay")?&heuristic:nullptr;
   diskann::TailProfile local;
   auto& prof=formal?profiles[i]:local;
   if(mode=="replay")prof.replay=&tapes[i];
   else if(formal && mode=="profile" && kind!=0){tapes[i].clear();prof.record=&tapes[i];}
   prof.timers=formal && mode=="profile";
   diskann::tail_profile=&prof;diskann::online_termination=et.get();diskann::online_miss_only=true;diskann::online_model_calls=calls;
   index.cached_beam_search(q+i*ad,10,L,ids,ds,4,std::numeric_limits<uint32_t>::max(),false,&s);
   if(mode=="replay" && prof.replay_pos!=tapes[i].size())throw std::runtime_error("replay length mismatch");
   diskann::tail_profile=nullptr;
   diskann::online_gate=nullptr;diskann::online_heuristic=nullptr;diskann::online_termination=nullptr;diskann::online_miss_only=false;diskann::online_model_calls=nullptr;
  };
  size_t warm=std::min(nq,size_t(128));
  #pragma omp parallel for schedule(static) num_threads(4)
  for(size_t j=0;j<warm;++j){mkl_set_num_threads_local(1);uint64_t ids[10];float ds[10];diskann::QueryStats s;search(j*nq/warm,s,ids,ds,nullptr);}
  formal=true;
  std::vector<std::array<uint64_t,10>>results(nq);std::vector<diskann::QueryStats>stats(nq);std::vector<double>us(nq);std::vector<uint32_t>calls(nq);
  uint64_t rss_before=proc_kb("VmRSS"),swap_before=proc_kb("VmSwap");std::cerr<<"MEMORY_READY "<<key<<" rss_kib="<<rss_before<<" swap_kib="<<swap_before<<std::endl;if(swap_before)throw std::runtime_error("Swapped process cannot be timed as resident");
  reader->requests=0;reader->bytes=0;uint64_t physical_before=physical_bytes();
  auto begin=std::chrono::steady_clock::now();
  #pragma omp parallel for schedule(static) num_threads(4)
  for(size_t i=0;i<nq;++i){mkl_set_num_threads_local(1);float ds[10];auto start=std::chrono::steady_clock::now();search(i,stats[i],results[i].data(),ds,&calls[i]);us[i]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();}
  double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
  uint64_t physical_delta=physical_bytes()-physical_before;uint64_t rss_after=proc_kb("VmRSS"),swap_after=proc_kb("VmSwap");if(swap_after)throw std::runtime_error("Swap detected during timed search");
  std::ofstream csv(mode_prefix+".csv"),ids(mode_prefix+".ids",std::ios::binary),meta(mode_prefix+".meta.json");csv<<"query,hits,reads,expansions,skips,gate_evaluated,et_evaluated,et_stopped,model_calls,latency_us,io_us,cpu_partial_us,io_rounds,rounds,empty_rounds,queue_slots,rank_us,feature_us,score_us,cache_hits,comparisons\n";
  for(size_t i=0;i<nq;++i){unsigned hits=0;for(auto id:results[i])for(unsigned j=0;j<10;++j)if(id==gt[i*dg+j]){++hits;break;}auto&s=stats[i];
   csv<<std::setprecision(12)<<i<<','<<hits<<','<<s.n_ios<<','<<s.n_expanded<<','<<s.n_gate_skipped<<','<<s.n_gate_evaluated<<','<<s.n_et_evaluated<<','<<s.n_et_stopped<<','<<calls[i]<<','<<us[i]<<','<<s.io_us<<','<<s.cpu_us<<','<<s.n_hops<<','<<profiles[i].rounds<<','<<profiles[i].empty_rounds<<','<<profiles[i].queue_slots<<','<<profiles[i].rank_us<<','<<profiles[i].feature_us<<','<<profiles[i].score_us<<','<<s.n_cache_hits<<','<<s.n_cmps<<'\n';ids.write((char*)results[i].data(),80);
  }
  meta<<std::setprecision(14)<<"{\"queries\":"<<nq<<",\"wall_seconds\":"<<seconds<<",\"qps\":"<<double(nq)/seconds<<",\"reader_requests\":"<<reader->requests.load()<<",\"reader_bytes\":"<<reader->bytes.load()<<",\"linux_read_bytes\":"<<physical_delta<<",\"linux_read_bytes_source\":\""<<physical_source<<"\",\"rss_before_kib\":"<<rss_before<<",\"rss_after_kib\":"<<rss_after<<",\"peak_rss_kib\":"<<proc_kb("VmHWM")<<",\"swap_before_kib\":"<<swap_before<<",\"swap_after_kib\":"<<swap_after<<"}\n";
  if(mode=="profile" && std::stoi(argv[8])==0){
   std::ofstream trace(mode_prefix+".trace.u32",std::ios::binary);
   for(uint32_t qi=0;qi<nq;++qi)for(size_t j=0;j<profiles[qi].nodes.size();++j){
    uint32_t row[3]={qi,profiles[qi].nodes[j],uint32_t(profiles[qi].skipped[j])};
    trace.write((char*)row,sizeof(row));
   }
  }
  std::cout<<"CONFIG_DONE "<<key<<" "<<mode<<std::endl;
  }
 }
 diskann::aligned_free(q);delete[]gt;delete[]gd;
}
