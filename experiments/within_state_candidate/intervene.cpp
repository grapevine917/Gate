#include "common_includes.h"
#include "disk_utils.h"
#include "pq_flash_index.h"
#include "online_gate.h"
#include "override.h"
#include "linux_aligned_file_reader.h"
#include <fstream>
#include <iomanip>
#include <array>
struct Request{uint32_t id,row,step,node,skip;float score,state[17];};
int main(int argc,char**argv){
 if(argc!=8){std::cerr<<"index queries gt model tau requests output-prefix\n";return 2;}
 float*q=nullptr,*gd=nullptr;uint32_t*gt=nullptr;size_t nq,d,ad,ng,dg;
 diskann::load_aligned_bin<float>(argv[2],q,nq,d,ad);diskann::load_truthset(argv[3],gt,gd,ng,dg);
 std::ifstream in(argv[6],std::ios::binary);uint32_t nr;in.read((char*)&nr,4);std::vector<Request> requests(nr);in.read((char*)requests.data(),nr*sizeof(Request));if(!in||sizeof(Request)!=92||nq!=ng)throw std::runtime_error("request or GT shape");
 std::shared_ptr<AlignedFileReader>reader=std::make_shared<LinuxAlignedFileReader>();diskann::PQFlashIndex<float,uint32_t>index(reader,diskann::Metric::L2);
 if(index.load(4,argv[1]))return 3;std::vector<uint32_t>nodes;index.cache_bfs_levels(10000,nodes);index.load_cache_list(nodes);diskann::OnlineGate model(argv[4],"",std::stof(argv[5]));
 auto search=[&](size_t row,uint64_t*ids,float*ds,diskann::QueryStats&s){diskann::online_gate=&model;diskann::online_miss_only=true;index.cached_beam_search(q+row*ad,10,100,ids,ds,4,std::numeric_limits<uint32_t>::max(),false,&s);diskann::online_gate=nullptr;diskann::online_miss_only=false;};
 auto hits=[&](size_t row,const uint64_t*ids){unsigned n=0;for(size_t i=0;i<10;++i)for(size_t j=0;j<10;++j)if(ids[i]==gt[row*dg+j]){++n;break;}return n;};
 std::vector<std::array<uint64_t,10>>base(nq);std::vector<std::array<float,10>>base_ds(nq);std::vector<diskann::QueryStats>bs(nq);
 #pragma omp parallel for schedule(static) num_threads(4)
 for(size_t i=0;i<nq;++i){mkl_set_num_threads_local(1);search(i,base[i].data(),base_ds[i].data(),bs[i]);}
 std::string prefix=argv[7];std::ofstream baseline(prefix+".baseline.ids",std::ios::binary);for(auto&ids:base)baseline.write((char*)ids.data(),80);
 std::ofstream cache(prefix+".cache.bin",std::ios::binary);cache.write((char*)nodes.data(),nodes.size()*4);cache.close();
 std::vector<std::string>rows(nr);std::vector<std::array<uint64_t,10>>outputs(nr);
 std::cerr<<"REQUESTS "<<nr<<" BASELINE_QUERIES "<<nq<<std::endl;
 #pragma omp parallel for schedule(dynamic,1) num_threads(4)
 for(size_t i=0;i<nr;++i){
  mkl_set_num_threads_local(1);const auto&r=requests[i];if(r.row>=nq||r.id!=i)throw std::runtime_error("invalid request");
  diskann::E2Override hook;hook.step=r.step;hook.node=r.node;hook.skip=r.skip;hook.expected_score=r.score;std::copy(r.state,r.state+17,hook.state);diskann::e2_override=&hook;
  float ds[10];diskann::QueryStats st;search(r.row,outputs[i].data(),ds,st);diskann::e2_override=nullptr;
  if(hook.applied!=1)throw std::runtime_error("replacement did not execute exactly once");
  bool same_ids=outputs[i]==base[r.row];bool same_ds=std::equal(ds,ds+10,base_ds[r.row].data());bool oldskip=hook.original_score<model.threshold;
  if(oldskip==bool(r.skip)&&(!same_ids||!same_ds||st.n_ios!=bs[r.row].n_ios))throw std::runtime_error("same-decision sham changed trajectory");
  std::ostringstream out;out<<std::setprecision(12)<<i<<','<<r.row<<','<<r.step<<','<<r.node<<','<<r.skip<<','<<oldskip<<','<<hits(r.row,base[r.row].data())<<','<<hits(r.row,outputs[i].data())<<','<<bs[r.row].n_ios<<','<<st.n_ios<<','<<same_ids<<','<<hook.applied<<','<<hook.original_score<<','<<bs[r.row].n_expanded<<','<<st.n_expanded<<','<<bs[r.row].n_cmps<<','<<st.n_cmps<<'\n';rows[i]=out.str();
  if((i+1)%2000==0){
   #pragma omp critical
   std::cerr<<"PROCESSED_REQUEST_INDEX "<<i+1<<std::endl;
  }
 }
 std::ofstream csv(prefix+".csv");csv<<"request,query_row,step,node,replacement_skip,original_skip,baseline_hits,after_hits,baseline_reads,after_reads,same_ids,applied,original_score,baseline_expanded,after_expanded,baseline_neighbor_cmps,after_neighbor_cmps\n";
 std::ofstream ids(prefix+".ids",std::ios::binary);for(size_t i=0;i<nr;++i){csv<<rows[i];ids.write((char*)outputs[i].data(),80);}std::cerr<<"COMPLETE "<<nr<<std::endl;
 diskann::aligned_free(q);delete[]gt;delete[]gd;
}
