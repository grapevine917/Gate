#include "common_includes.h"
#include "disk_utils.h"
#include "pq_flash_index.h"
#include "online_gate.h"
#include "gate_trace.h"
#include "linux_aligned_file_reader.h"
#include <random>
#include <set>
#include <unordered_map>
#include <iomanip>
struct Row {
 uint32_t query,policy,step,node,stratum,population,sampled,proxy;
 int32_t keep_hits,skip_hits,keep_reads,skip_reads,lost_gt,gained_gt;
 float weight,old_score,state[17];
};
static_assert(sizeof(Row)==132,"row layout");
struct Result {std::array<uint64_t,10> ids;diskann::QueryStats stats;diskann::GateTrace trace;std::vector<float> audit;};
int main(int argc,char**argv){
 if(argc!=11){std::cerr<<"index query gt models mlp_tau lr_tau start end trainflag output\n";return 2;}
 float*q=nullptr,*gd=nullptr;uint32_t*gt=nullptr;size_t nq,d,ad,ng,dg;
 diskann::load_aligned_bin<float>(argv[2],q,nq,d,ad);diskann::load_truthset(argv[3],gt,gd,ng,dg);if(nq!=ng)return 3;
 unsigned start=std::stoul(argv[7]),end=std::min(nq,size_t(std::stoul(argv[8])));bool train=std::stoi(argv[9]);
 std::shared_ptr<AlignedFileReader>reader=std::make_shared<LinuxAlignedFileReader>();diskann::PQFlashIndex<float,uint32_t>index(reader,diskann::Metric::L2);if(index.load(4,argv[1]))return 4;
 std::vector<uint32_t>cache;index.cache_bfs_levels(10000,cache);index.load_cache_list(cache);
 diskann::OnlineGate off(std::string(argv[4])+"/state.bin","",0),mlp(std::string(argv[4])+"/state.bin","",std::stof(argv[5])),lr(std::string(argv[4])+"/lr.bin","",std::stof(argv[6]));
 auto search=[&](unsigned qi,unsigned L,diskann::OnlineGate*model,bool audit){
  Result r;float ds[10];diskann::online_gate=model;diskann::online_miss_only=true;diskann::gate_trace=&r.trace;diskann::online_audit=audit?&r.audit:nullptr;
  index.cached_beam_search(q+qi*ad,10,L,r.ids.data(),ds,4,std::numeric_limits<uint32_t>::max(),false,&r.stats);
  diskann::online_gate=nullptr;diskann::online_miss_only=false;diskann::gate_trace=nullptr;diskann::online_audit=nullptr;return r;
 };
 auto true_ids=[&](unsigned qi,const Result&r){std::set<uint64_t>s;for(auto id:r.ids)for(unsigned j=0;j<10;++j)if(id==gt[qi*dg+j])s.insert(id);return s;};
 std::vector<std::vector<Row>>rows(end-start);std::vector<std::string>summaries(end-start);std::vector<std::array<uint64_t,10>>baseids((end-start)*3);std::atomic<unsigned>done{0};
 #pragma omp parallel for schedule(dynamic,1) num_threads(4)
 for(unsigned qi=start;qi<end;++qi){
  mkl_set_num_threads_local(1);std::unordered_set<uint32_t>teacher_path;unsigned teacher_hits[2]={};
  for(unsigned ti=0;ti<2;++ti){
   auto t=search(qi,ti?1000:300,&off,false);teacher_hits[ti]=true_ids(qi,t).size();std::unordered_map<uint32_t,uint32_t>parent;
   for(auto&e:t.trace.edges)if((e.flags&3)==3 && !parent.count(e.child))parent[e.child]=e.parent;
   for(auto id:true_ids(qi,t)){uint32_t node=id;std::unordered_set<uint32_t>seen;
    while(seen.insert(node).second){teacher_path.insert(node);auto it=parent.find(node);if(it==parent.end())break;node=it->second;}
   }
  }
  unsigned basehits[3]={},counts[3]={},harmful[3]={},shams=0;
  diskann::OnlineGate*policies[]={&off,&mlp,&lr};
  for(unsigned pi=0;pi<3;++pi){
   auto base=search(qi,100,policies[pi],true);baseids[(qi-start)*3+pi]=base.ids;auto basegt=true_ids(qi,base);basehits[pi]=basegt.size();
   std::vector<unsigned>strata[3];
   for(unsigned j=0;j<base.audit.size()/19;++j){const float*x=base.audit.data()+j*19+1;if(x[10]!=0)continue;uint32_t node=base.audit[j*19];unsigned st=teacher_path.count(node)?0:(x[3]>72 && x[7]>100)?1:2;strata[st].push_back(j);}
   std::mt19937 rng(20261101+qi*101+pi*1709);unsigned budgets[3]={train?6u:3u,train?4u:2u,train?6u:3u};bool sham_done=false;
   for(unsigned st=0;st<3;++st){auto ix=strata[st];std::shuffle(ix.begin(),ix.end(),rng);unsigned take=std::min(size_t(budgets[st]),ix.size());ix.resize(take);std::sort(ix.begin(),ix.end());
    for(auto j:ix){uint32_t node=base.audit[j*19];const float*x=base.audit.data()+j*19+1;float score=base.audit[j*19+18];bool original_skip=score<policies[pi]->threshold;
     auto makehook=[&](bool skip){diskann::RecallHook h;h.step=j;h.node=node;h.skip=skip;std::copy(x,x+17,h.state);return h;};
     if(!sham_done){auto h=makehook(original_skip);diskann::recall_hook=&h;auto sham=search(qi,100,policies[pi],false);diskann::recall_hook=nullptr;if(h.applied!=1 || sham.ids!=base.ids || sham.stats.n_ios!=base.stats.n_ios || sham.stats.n_expanded!=base.stats.n_expanded)throw std::runtime_error("sham mismatch");sham_done=true;++shams;}
     auto h=makehook(!original_skip);diskann::recall_hook=&h;auto alt=search(qi,100,policies[pi],false);diskann::recall_hook=nullptr;if(h.applied!=1)throw std::runtime_error("branch intervention count");
     const auto&keep=original_skip?alt:base;const auto&skip=original_skip?base:alt;auto kg=true_ids(qi,keep),sg=true_ids(qi,skip);unsigned lost=0,gained=0;for(auto id:kg)lost+=!sg.count(id);for(auto id:sg)gained+=!kg.count(id);
     bool proxy=std::find(keep.ids.begin(),keep.ids.end(),uint64_t(node))!=keep.ids.end();bool expanded=false;for(auto&e:keep.trace.expansions)if(e.node==node){proxy=proxy||e.admitted>0;expanded=true;break;}if(!expanded)throw std::runtime_error("forced keep not expanded");
     Row r{};r.query=qi;r.policy=pi;r.step=j;r.node=node;r.stratum=st;r.population=strata[st].size();r.sampled=take;r.proxy=proxy;r.keep_hits=kg.size();r.skip_hits=sg.size();r.keep_reads=keep.stats.n_ios;r.skip_reads=skip.stats.n_ios;r.lost_gt=lost;r.gained_gt=gained;r.weight=float(r.population)/take;r.old_score=score;std::copy(x,x+17,r.state);rows[qi-start].push_back(r);++counts[pi];harmful[pi]+=r.keep_hits>r.skip_hits;
    }
   }
  }
  std::ostringstream os;os<<qi<<','<<teacher_hits[0]<<','<<teacher_hits[1];for(unsigned pi=0;pi<3;++pi)os<<','<<basehits[pi]<<','<<counts[pi]<<','<<harmful[pi];os<<','<<shams<<'\n';summaries[qi-start]=os.str();auto n=++done;if(n%20==0){
   #pragma omp critical
   std::cerr<<"QUERIES_COMPLETE "<<n<<"/"<<end-start<<std::endl;
  }
 }
 std::ofstream bin(std::string(argv[10])+".bin",std::ios::binary),csv(std::string(argv[10])+".csv"),ids(std::string(argv[10])+".ids",std::ios::binary),meta(std::string(argv[10])+".json");size_t total=0;
 for(auto&rr:rows){bin.write((char*)rr.data(),rr.size()*sizeof(Row));total+=rr.size();}for(auto&r:baseids)ids.write((char*)r.data(),80);
 csv<<"query,teacher300_hits,teacher1000_hits,off_hits,off_samples,off_harmful,mlp_hits,mlp_samples,mlp_harmful,lr_hits,lr_samples,lr_harmful,shams\n";for(auto&s:summaries)csv<<s;
 meta<<"{\"complete\":true,\"start\":"<<start<<",\"end\":"<<end<<",\"rows\":"<<total<<",\"row_bytes\":132,\"shams\":"<<(end-start)*3<<"}\n";
 diskann::aligned_free(q);delete[]gt;delete[]gd;
}
