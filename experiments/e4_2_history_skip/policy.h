#pragma once
#include <array>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>
namespace diskann {
constexpr int E42F=28;
struct E42History {uint32_t node,query,skip,pad;double z[17];};
struct E42Audit {uint32_t row,node,ordinal,pad;double x[E42F],scores[7];};
struct E42Bank {
 double mean[17],denom[17],weights[7][E42F+1];double scut=0,qcut=0;
 std::vector<E42History>records;std::unordered_map<uint32_t,std::pair<size_t,size_t>>ranges;
};
inline double e42_hash(uint32_t q,uint32_t n){uint32_t x=q*1000003u+n+20261020u;x=(x^(x>>16))*0x7feb352du;x=(x^(x>>15))*0x846ca68bu;x=x^(x>>16);return double(x)/4294967296.;}
struct E42Context {
 const E42Bank*bank=nullptr;const double*qd=nullptr;int method=-1;double cutoff=1e30,tiecut=-1;float tau=0;
 uint32_t row=0,qid=0,ordinal=0,keeps=0,conflicts=0,pruned=0;bool audit=false;std::vector<E42Audit>audits;
 void features(uint32_t node,const float*s,float fresh,double*x)const{
  std::fill(x,x+E42F,0.);double score=std::min(1.-1e-7,std::max(1e-7,double(fresh)));x[0]=std::log(score/(1-score));
  for(int k=0;k<17;++k)x[k+1]=(double(s[k])-bank->mean[k])/bank->denom[k];
  auto it=bank->ranges.find(node);if(it==bank->ranges.end())return;
  double total=it->second.second-it->second.first,skips=0,ds=std::numeric_limits<double>::infinity(),dq=ds,dqstate=ds;bool joint=false;
  for(size_t i=it->second.first;i<it->second.second;++i){const auto&r=bank->records[i];if(!r.skip)continue;++skips;double d2=0;
   for(int k=0;k<17;++k){double diff=x[k+1]-r.z[k];d2+=diff*diff;}double d=std::sqrt(d2),qq=qd[r.query];
   if(d<ds){ds=d;dqstate=qq;}dq=std::min(dq,qq);joint=joint||(d<bank->scut&&qq<bank->qcut);
  }
  x[18]=skips>0;x[19]=std::log1p(total);x[20]=std::log1p(skips);x[21]=skips/std::max(total,1.);
  if(skips>0){x[22]=std::log1p(ds);x[23]=std::log1p(dq);x[24]=std::log1p(dqstate);x[25]=ds<bank->scut;x[26]=dq<bank->qcut;x[27]=joint;}
 }
 double predict(int m,const double*x)const{double z=bank->weights[m][0];for(int j=0;j<E42F;++j)z+=bank->weights[m][j+1]*x[j];return z;}
};
inline thread_local E42Context*e42_context=nullptr;
inline void e42_apply(uint32_t node,const float*s,float&score){
 auto*c=e42_context;if(!c)return;uint32_t ordinal=c->ordinal++;if(s[10]!=0||score<c->tau)return;++c->keeps;
 double x[E42F];c->features(node,s,score,x);if(x[18]>0)++c->conflicts;
 if(c->audit){E42Audit a{};a.row=c->row;a.node=node;a.ordinal=ordinal;std::copy(x,x+E42F,a.x);for(int m=0;m<7;++m)a.scores[m]=c->predict(m,x);c->audits.push_back(a);}
 if(c->method<0)return;if(c->method>=1&&c->method<=5&&x[18]==0)return;
 double value=c->predict(c->method,x);if(value>c->cutoff||(value==c->cutoff&&e42_hash(c->qid,node)<=c->tiecut)){score=0;++c->pruned;}
}
}
