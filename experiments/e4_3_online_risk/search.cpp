#include "common_includes.h"
#include "disk_utils.h"
#include "pq_flash_index.h"
#include "online_gate.h"
#include "policy.h"
#include "linux_aligned_file_reader.h"
#include <fstream>
#include <iomanip>
#include <string>
#include <chrono>
struct Config{std::string key;int kind;double threshold;};
int main(int argc,char**argv){
 if(argc!=14){std::cerr<<"index queries gt model tau history risk_models partitions configs order tag cold scut\n";return 2;}
 float*q=nullptr,*gd=nullptr;uint32_t*gt=nullptr;size_t nq,d,ad,ng,dg;diskann::load_aligned_bin<float>(argv[2],q,nq,d,ad);diskann::load_truthset(argv[3],gt,gd,ng,dg);if(nq!=1000||nq!=ng||dg<10)throw std::runtime_error("query shape");
 diskann::E43Bank source;source.scut=std::stod(argv[13]);std::ifstream h(argv[6],std::ios::binary);h.read((char*)source.mean,136);h.read((char*)source.denom,136);uint32_t nr;h.read((char*)&nr,4);std::vector<diskann::E43Seed>seed(nr);h.read((char*)seed.data(),nr*sizeof(diskann::E43Seed));
 if(!h||sizeof(diskann::E43Seed)!=152||sizeof(diskann::E43Pending)!=80||sizeof(diskann::E43Update)!=88||sizeof(diskann::E43Decision)!=296||sizeof(diskann::E43Audit)!=256)throw std::runtime_error("history or trace layout");
 std::ifstream mf(argv[7],std::ios::binary);uint32_t nm,nf;mf.read((char*)&nm,4);mf.read((char*)&nf,4);mf.read((char*)source.weights,sizeof(source.weights));if(!mf||nm!=2||nf!=28)throw std::runtime_error("risk models");
 std::vector<uint32_t>qid(nq);std::ifstream part(argv[8]);for(uint32_t i=0;i<nq;++i){uint32_t row,p;part>>row>>p>>qid[i];if(row!=i)throw std::runtime_error("partition order");}if(!part)throw std::runtime_error("partition file");
 std::ifstream cf(argv[9]);uint32_t nc;cf>>nc;std::vector<Config>configs(nc);for(auto&c:configs)cf>>c.key>>c.kind>>c.threshold;if(!cf)throw std::runtime_error("configs");
 std::vector<uint32_t>rows;std::ifstream of(argv[10]);uint32_t row;while(of>>row){if(row>=nq)throw std::runtime_error("order row");rows.push_back(row);}if(rows.empty())throw std::runtime_error("empty order");
 std::string tag=argv[11];bool cold=std::stoi(argv[12])!=0,audit=tag=="audit";if(audit)configs={Config{"MLP",0,0.}};
 std::shared_ptr<AlignedFileReader>reader=std::make_shared<LinuxAlignedFileReader>();diskann::PQFlashIndex<float,uint32_t>index(reader,diskann::Metric::L2);if(index.load(4,argv[1]))return 3;std::vector<uint32_t>cache_nodes;index.cache_bfs_levels(10000,cache_nodes);index.load_cache_list(cache_nodes);diskann::OnlineGate model(argv[4],"",std::stof(argv[5]));
 for(const auto&cfg:configs){std::string prefix="graph/"+tag+"/"+cfg.key;if(std::ifstream(prefix+".complete").good()){std::cerr<<"RESUME "<<prefix<<std::endl;continue;}
  diskann::E43Bank bank=source;if(!cold)bank.seed(seed);uint64_t seedobs=bank.observations,seedskip=bank.skip_records;
  std::ofstream observations(prefix+".observations.bin",std::ios::binary);
  std::ofstream csv(prefix+".csv"),ids(prefix+".ids",std::ios::binary),decisions(prefix+".decisions.bin",std::ios::binary),updates(prefix+".updates.bin",std::ios::binary),audits(prefix+".audit.bin",std::ios::binary);
  csv<<"seq,query_row,query_id,hits,reads,expanded,neighbor_cmps,model_calls,original_skips,original_keeps,conflicts,secondary_evaluations,risk_LR_calls,secondary_skips,history_version,observations_before,observations_after,skip_records_before,skip_records_after,marked_nodes_before,marked_nodes_after,committed_observations,committed_skips,search_us,update_us,service_us,history_payload_bytes,history_container_lower_bound_bytes,history_chain\n";
  for(size_t seq=0;seq<rows.size();++seq){uint32_t qr=rows[seq];mkl_set_num_threads_local(1);diskann::E43Context c;c.bank=&bank;c.kind=cfg.kind;c.cutoff=cfg.threshold;c.tau=model.threshold;c.seq=seq;c.row=qr;c.audit=audit;
   if(bank.completed!=seq)throw std::runtime_error("query history version");uint64_t ob=bank.observations,sb=bank.skip_records,mb=bank.marked_nodes;uint32_t hv=bank.completed;
   uint64_t result[10];float ds[10];diskann::QueryStats st;diskann::e43_context=&c;diskann::online_gate=&model;diskann::online_miss_only=true;
   auto begin=std::chrono::steady_clock::now();index.cached_beam_search(q+qr*ad,10,100,result,ds,4,std::numeric_limits<uint32_t>::max(),false,&st);auto search_end=std::chrono::steady_clock::now();diskann::e43_context=nullptr;diskann::online_gate=nullptr;diskann::online_miss_only=false;
   if(c.model_calls!=c.original_keeps+c.original_skips||c.pending.size()!=c.model_calls||c.secondary_skips>c.original_keeps||(cfg.kind!=6&&c.secondary_skips>c.conflicts))throw std::runtime_error("decision count invariant");
   bool update=!audit&&cfg.kind!=3;std::vector<diskann::E43Update>newupdates;auto update_begin=std::chrono::steady_clock::now();diskann::e43_commit(c,update,newupdates);auto update_end=std::chrono::steady_clock::now();
   if(update&&(bank.observations-ob!=c.model_calls||bank.skip_records-sb!=c.original_skips||newupdates.size()!=c.original_skips))throw std::runtime_error("history update invariant");
   if(!update&&(bank.observations!=ob||bank.skip_records!=sb))throw std::runtime_error("frozen history changed");
   uint32_t hits=0;for(auto id:result)for(size_t j=0;j<10;++j)if(id==gt[qr*dg+j]){++hits;break;}
   double search_us=std::chrono::duration<double,std::micro>(search_end-begin).count(),update_us=std::chrono::duration<double,std::micro>(update_end-update_begin).count();
   csv<<std::setprecision(12)<<seq<<','<<qr<<','<<qid[qr]<<','<<hits<<','<<st.n_ios<<','<<st.n_expanded<<','<<st.n_cmps<<','<<c.model_calls<<','<<c.original_skips<<','<<c.original_keeps<<','<<c.conflicts<<','<<c.secondary_evaluations<<','<<c.risk_calls<<','<<c.secondary_skips<<','<<hv<<','<<ob<<','<<bank.observations<<','<<sb<<','<<bank.skip_records<<','<<mb<<','<<bank.marked_nodes<<','<<(bank.observations-ob)<<','<<(bank.skip_records-sb)<<','<<search_us<<','<<update_us<<','<<search_us+update_us<<','<<bank.state_bytes()<<','<<bank.capacity_bytes()<<','<<bank.chain<<'\n';
   observations.write((char*)c.pending.data(),c.pending.size()*sizeof(diskann::E43Pending));ids.write((char*)result,80);decisions.write((char*)c.decisions.data(),c.decisions.size()*sizeof(diskann::E43Decision));updates.write((char*)newupdates.data(),newupdates.size()*sizeof(diskann::E43Update));audits.write((char*)c.audits.data(),c.audits.size()*sizeof(diskann::E43Audit));
  }
  observations.close();csv.close();ids.close();decisions.close();updates.close();audits.close();std::ofstream cache(prefix+".cache.bin",std::ios::binary);cache.write((char*)cache_nodes.data(),cache_nodes.size()*4);cache.close();std::ofstream done(prefix+".complete");done<<rows.size()<<' '<<seedobs<<' '<<seedskip<<' '<<bank.observations<<' '<<bank.skip_records<<' '<<bank.chain;done.close();std::cerr<<"COMPLETE "<<tag<<' '<<cfg.key<<" queries="<<rows.size()<<std::endl;
 }
 diskann::aligned_free(q);delete[]gt;delete[]gd;
}
