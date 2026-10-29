#pragma once
#include <array>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace diskann {
struct E3History {uint32_t node,query,skip,pad;double z[17];};
struct E3Record {uint32_t node,reused;float fresh,state[17];};
struct E3Bank {
 double mean[17],denom[17];std::vector<E3History>records;
 std::unordered_map<uint32_t,std::pair<size_t,size_t>>ranges;
};
struct E3Context {
 const E3Bank*bank=nullptr;int mode=0,kind=0,votes=2,oracle=0;
 double qcut=1e30,scut=1e30;float tau=0;
 std::array<uint32_t,3>neighbours{};std::array<double,3>query_dist{};
 std::vector<E3Record>tape;size_t cursor=0;
 uint32_t eligible=0,reused=0,false_skip=0,fresh_keep=0;
 bool candidate(uint32_t node,const float*s) const {
  if(kind==0)return false;
  auto it=bank->ranges.find(node);if(it==bank->ranges.end())return false;
  bool any=false;double best=std::numeric_limits<double>::infinity();bool nearest_skip=false;int available=0,skips=0;
  double z[17];if(kind==3||kind==4)for(int k=0;k<17;++k)z[k]=(double(s[k])-bank->mean[k])/bank->denom[k];
  for(size_t i=it->second.first;i<it->second.second;++i){
   const auto&r=bank->records[i];
   if(kind==1){if(r.skip)return true;continue;}
   if(kind==2){if(r.query==neighbours[0])return r.skip;continue;}
   int rank=-1;for(int k=0;k<3;++k)if(r.query==neighbours[k]&&query_dist[k]<=qcut){rank=k;break;}
   if(kind==5){if(rank>=0){++available;skips+=r.skip;}continue;}
   if(kind==4&&rank<0)continue;
   double d=0;for(int k=0;k<17;++k){double v=z[k]-r.z[k];d+=v*v;}
   if(d<best){best=d;nearest_skip=r.skip;any=true;}
  }
  if(kind==5)return available>=votes&&skips==available;
  return any&&nearest_skip&&best<=scut*scut;
 }
};
inline thread_local E3Context*e3_context=nullptr;
inline bool e3_before(uint32_t node,const float*s,float&score){
 auto*c=e3_context;if(!c||c->mode!=2||s[10]!=0)return false;
 if(c->cursor>=c->tape.size())throw std::runtime_error("E3 tape exhausted");
 const auto&r=c->tape[c->cursor++];
 if(r.node!=node||std::memcmp(r.state,s,17*sizeof(float)))throw std::runtime_error("E3 tape node/state mismatch");
 ++c->eligible;c->fresh_keep+=r.fresh>=c->tau;
 if(r.reused){++c->reused;c->false_skip+=r.fresh>=c->tau;score=0;return true;}
 return false;
}
inline void e3_after(uint32_t node,const float*s,float&score){
 auto*c=e3_context;if(!c||s[10]!=0)return;
 if(c->mode==2){const auto&r=c->tape[c->cursor-1];if(!r.reused&&std::abs(score-r.fresh)>2e-6)throw std::runtime_error("E3 fresh-score mismatch");return;}
 if(c->mode!=1)return;
 ++c->eligible;c->fresh_keep+=score>=c->tau;
 bool reuse=c->candidate(node,s)&&(!c->oracle||score<c->tau);
 E3Record r;r.node=node;r.reused=reuse;r.fresh=score;std::copy(s,s+17,r.state);c->tape.push_back(r);
 if(reuse){++c->reused;c->false_skip+=score>=c->tau;score=0;}
}
}
