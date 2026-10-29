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
int main(int argc,char**argv){
 if(argc!=9){std::cerr<<"index query gt models base_vectors configs output repeat\n";return 2;}
 float*q=nullptr,*gd=nullptr;uint32_t*gt=nullptr;size_t nq,d,ad,ng,dg;
 diskann::load_aligned_bin<float>(argv[2],q,nq,d,ad);diskann::load_truthset(argv[3],gt,gd,ng,dg);if(nq!=ng)return 3;
 std::shared_ptr<AlignedFileReader>reader=std::make_shared<LinuxAlignedFileReader>();diskann::PQFlashIndex<float,uint32_t>index(reader,diskann::Metric::L2);if(index.load(4,argv[1]))return 4;
 std::vector<uint32_t>nodes;index.cache_bfs_levels(10000,nodes);index.load_cache_list(nodes);
 std::string folder=argv[4];diskann::OnlineGate lr(folder+"/lr.bin","",0),xgb(folder+"/xgb.bin","",0),state(folder+"/state.bin","",0),qv(folder+"/qv.bin",argv[5],0),et(folder+"/et.bin","",1.1);
 diskann::OnlineGate* models[]={nullptr,nullptr,&lr,&xgb,&state,&qv,nullptr,&state};
 std::ifstream configs(argv[6]);std::string key;int kind,L,rank,expansions;float tau,et_tau;
 while(configs>>key>>kind>>L>>tau>>et_tau>>rank>>expansions){
  auto*gate=models[kind];if(gate)gate->threshold=tau;et.threshold=et_tau;diskann::OnlineHeuristic heuristic{rank,expansions};std::string prefix=std::string(argv[7])+"/"+key+"_r"+argv[8];
  auto search=[&](size_t i,diskann::QueryStats&s,uint64_t*ids,float*ds,uint32_t*calls){
   diskann::online_gate=gate;diskann::online_heuristic=kind==1?&heuristic:nullptr;diskann::online_termination=(kind==6||kind==7)?&et:nullptr;diskann::online_miss_only=true;diskann::online_model_calls=calls;
   index.cached_beam_search(q+i*ad,10,L,ids,ds,4,std::numeric_limits<uint32_t>::max(),false,&s);
   diskann::online_gate=nullptr;diskann::online_heuristic=nullptr;diskann::online_termination=nullptr;diskann::online_miss_only=false;diskann::online_model_calls=nullptr;
  };
  size_t warm=std::min(nq,size_t(128));
  #pragma omp parallel for schedule(static) num_threads(4)
  for(size_t j=0;j<warm;++j){mkl_set_num_threads_local(1);uint64_t ids[10];float ds[10];diskann::QueryStats s;search(j*nq/warm,s,ids,ds,nullptr);}
  std::vector<std::array<uint64_t,10>>results(nq);std::vector<diskann::QueryStats>stats(nq);std::vector<double>us(nq);std::vector<uint32_t>calls(nq);
  auto begin=std::chrono::steady_clock::now();
  #pragma omp parallel for schedule(static) num_threads(4)
  for(size_t i=0;i<nq;++i){mkl_set_num_threads_local(1);float ds[10];auto start=std::chrono::steady_clock::now();search(i,stats[i],results[i].data(),ds,&calls[i]);us[i]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();}
  double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
  std::ofstream csv(prefix+".csv"),ids(prefix+".ids",std::ios::binary),meta(prefix+".meta.json");csv<<"query,hits,reads,expansions,skips,gate_evaluated,et_evaluated,et_stopped,model_calls,latency_us\n";
  for(size_t i=0;i<nq;++i){unsigned hits=0;for(auto id:results[i])for(unsigned j=0;j<10;++j)if(id==gt[i*dg+j]){++hits;break;}auto&s=stats[i];
   csv<<std::setprecision(12)<<i<<','<<hits<<','<<s.n_ios<<','<<s.n_expanded<<','<<s.n_gate_skipped<<','<<s.n_gate_evaluated<<','<<s.n_et_evaluated<<','<<s.n_et_stopped<<','<<calls[i]<<','<<us[i]<<'\n';ids.write((char*)results[i].data(),80);
  }
  meta<<std::setprecision(14)<<"{\"queries\":"<<nq<<",\"wall_seconds\":"<<seconds<<",\"qps\":"<<double(nq)/seconds<<"}\n";std::cout<<"CONFIG_DONE "<<key<<std::endl;
 }
 diskann::aligned_free(q);delete[]gt;delete[]gd;
}
