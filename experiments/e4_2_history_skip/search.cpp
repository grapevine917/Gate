#include "common_includes.h"
#include "disk_utils.h"
#include "pq_flash_index.h"
#include "online_gate.h"
#include "policy.h"
#include "linux_aligned_file_reader.h"
#include <fstream>
#include <iomanip>
#include <string>
struct Config{std::string key;int method;double cutoff,tiecut;};
int main(int argc,char**argv){
 if(argc!=14){std::cerr<<"index queries gt model tau history models distances partitions configs phase scut qcut\n";return 2;}
 float*q=nullptr,*gd=nullptr;uint32_t*gt=nullptr;size_t nq,d,ad,ng,dg;diskann::load_aligned_bin<float>(argv[2],q,nq,d,ad);diskann::load_truthset(argv[3],gt,gd,ng,dg);if(nq!=1000||nq!=ng||dg<10)throw std::runtime_error("query shape");
 diskann::E42Bank bank;bank.scut=std::stod(argv[12]);bank.qcut=std::stod(argv[13]);std::ifstream hist(argv[6],std::ios::binary);hist.read((char*)bank.mean,136);hist.read((char*)bank.denom,136);uint32_t nr;hist.read((char*)&nr,4);bank.records.resize(nr);hist.read((char*)bank.records.data(),nr*sizeof(diskann::E42History));if(!hist||sizeof(diskann::E42History)!=152||sizeof(diskann::E42Audit)!=296)throw std::runtime_error("history/audit format");
 for(size_t i=0;i<nr;){size_t j=i+1;while(j<nr&&bank.records[j].node==bank.records[i].node)++j;bank.ranges[bank.records[i].node]={i,j};i=j;}
 std::ifstream models(argv[7],std::ios::binary);uint32_t nm,nf;models.read((char*)&nm,4);models.read((char*)&nf,4);models.read((char*)bank.weights,sizeof(bank.weights));if(!models||nm!=7||nf!=28)throw std::runtime_error("models format");
 std::vector<double>qd(nq*1000);std::ifstream dist(argv[8],std::ios::binary);dist.read((char*)qd.data(),qd.size()*8);if(!dist)throw std::runtime_error("query distances");
 std::vector<uint32_t>partition(nq),qid(nq);std::ifstream part(argv[9]);for(uint32_t i=0;i<nq;++i){uint32_t row;part>>row>>partition[i]>>qid[i];if(row!=i)throw std::runtime_error("partition order");}if(!part)throw std::runtime_error("partition file");
 std::vector<Config>configs;std::ifstream cf(argv[10]);uint32_t nc;cf>>nc;for(uint32_t i=0;i<nc;++i){Config c;cf>>c.key>>c.method>>c.cutoff>>c.tiecut;configs.push_back(c);}if(!cf)throw std::runtime_error("configs");
 std::shared_ptr<AlignedFileReader>reader=std::make_shared<LinuxAlignedFileReader>();diskann::PQFlashIndex<float,uint32_t>index(reader,diskann::Metric::L2);if(index.load(4,argv[1]))return 3;std::vector<uint32_t>nodes;index.cache_bfs_levels(10000,nodes);index.load_cache_list(nodes);diskann::OnlineGate model(argv[4],"",std::stof(argv[5]));
 std::string phase=argv[11];std::vector<uint32_t>rows;for(uint32_t i=0;i<nq;++i)if(phase=="baseline"||(phase=="validation"&&partition[i]==1)||(phase=="test"&&partition[i]==2))rows.push_back(i);if(rows.empty())throw std::runtime_error("phase");
 if(phase=="baseline")configs={Config{"baseline",-1,1e30,-1}};
 for(const auto&cfg:configs){
  std::string prefix="graph/"+cfg.key+"."+phase;if(std::ifstream(prefix+".complete").good()){std::cerr<<"RESUME "<<prefix<<std::endl;continue;}
  std::vector<std::array<uint64_t,10>>outputs(rows.size());std::vector<std::string>stats(rows.size());std::vector<std::vector<diskann::E42Audit>>audits(rows.size());
  #pragma omp parallel for schedule(static) num_threads(4)
  for(size_t ri=0;ri<rows.size();++ri){uint32_t row=rows[ri];mkl_set_num_threads_local(1);diskann::E42Context c;c.bank=&bank;c.qd=qd.data()+row*1000;c.method=cfg.method;c.cutoff=cfg.cutoff;c.tiecut=cfg.tiecut;c.tau=model.threshold;c.row=row;c.qid=qid[row];c.audit=phase=="baseline";diskann::e42_context=&c;diskann::online_gate=&model;diskann::online_miss_only=true;diskann::QueryStats st;float ds[10];index.cached_beam_search(q+row*ad,10,100,outputs[ri].data(),ds,4,std::numeric_limits<uint32_t>::max(),false,&st);diskann::e42_context=nullptr;diskann::online_gate=nullptr;diskann::online_miss_only=false;
   uint32_t hits=0;for(auto id:outputs[ri])for(size_t j=0;j<10;++j)if(id==gt[row*dg+j]){++hits;break;}
   std::ostringstream out;out<<row<<','<<qid[row]<<','<<hits<<','<<st.n_ios<<','<<st.n_expanded<<','<<st.n_cmps<<','<<c.keeps<<','<<c.conflicts<<','<<c.pruned<<'\n';stats[ri]=out.str();audits[ri]=std::move(c.audits);
  }
  std::ofstream csv(prefix+".csv"),ids(prefix+".ids",std::ios::binary);csv<<"query_row,query_id,hits,reads,expanded,neighbor_cmps,live_keeps,live_conflicts,pruned\n";for(size_t i=0;i<rows.size();++i){csv<<stats[i];ids.write((char*)outputs[i].data(),80);}csv.close();ids.close();
  if(phase=="baseline"){std::ofstream af(prefix+".audit.bin",std::ios::binary);for(auto&aa:audits)af.write((char*)aa.data(),aa.size()*sizeof(diskann::E42Audit));}
  std::ofstream cache(prefix+".cache.bin",std::ios::binary);cache.write((char*)nodes.data(),nodes.size()*4);cache.close();std::ofstream done(prefix+".complete");done<<rows.size();std::cerr<<"COMPLETE "<<prefix<<" queries="<<rows.size()<<std::endl;
 }
 diskann::aligned_free(q);delete[]gt;delete[]gd;
}
