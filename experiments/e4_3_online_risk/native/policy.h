#pragma once
#include <array>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cstring>
namespace diskann {
constexpr int E43F=28;
struct E43Seed {uint32_t node,query,skip,pad;double z[17];};
struct E43Skip {double z[17];uint32_t origin;};
struct E43Node {uint64_t total=0;std::vector<E43Skip>skips;};
struct E43Pending {uint32_t node,skip;float fresh,s[17];};
struct E43Update {uint32_t seq,row,node,pad;float fresh,s[17];};
struct E43Decision {uint32_t seq,row,node,ordinal,history_version,nearest_origin,action,kind;double fresh,risk,distance,node_total,node_skips,x[E43F];};
struct E43Audit {uint32_t row,node,ordinal,pad;double x[E43F],logits[2];};
struct E43Bank {
 double mean[17],denom[17],weights[2][E43F+1],scut=0;std::unordered_map<uint32_t,E43Node>nodes;
 uint64_t observations=0,skip_records=0,marked_nodes=0,chain=1469598103934665603ull;uint32_t completed=0;
 void seed(const std::vector<E43Seed>&records){for(const auto&r:records){auto&n=nodes[r.node];++n.total;++observations;if(r.skip){if(n.skips.empty())++marked_nodes;E43Skip s{};std::copy(r.z,r.z+17,s.z);s.origin=0;n.skips.push_back(s);++skip_records;}}}
 double logit(int m,const double*x)const{double z=weights[m][0];for(int j=0;j<E43F;++j)z+=weights[m][j+1]*x[j];return z;}
 static double sigmoid(double z){if(z>=0)return 1./(1.+std::exp(-z));double e=std::exp(z);return e/(1.+e);}
 void hash(const void*p,size_t n){auto*c=static_cast<const unsigned char*>(p);for(size_t i=0;i<n;++i){chain^=c[i];chain*=1099511628211ull;}}
 uint64_t state_bytes()const{uint64_t b=0;for(const auto&kv:nodes)b+=sizeof(uint32_t)+sizeof(uint64_t)+sizeof(uint64_t)+kv.second.skips.size()*sizeof(E43Skip);return b;}
 uint64_t capacity_bytes()const{uint64_t b=nodes.bucket_count()*sizeof(void*);for(const auto&kv:nodes)b+=sizeof(kv)+kv.second.skips.capacity()*sizeof(E43Skip);return b;}
};
struct E43Context {
 E43Bank*bank=nullptr;int kind=0;double cutoff=0;float tau=0;uint32_t seq=0,row=0,ordinal=0;
 uint32_t model_calls=0,original_skips=0,original_keeps=0,conflicts=0,secondary_evaluations=0,risk_calls=0,secondary_skips=0;bool audit=false;
 std::vector<E43Pending>pending;std::vector<E43Decision>decisions;std::vector<E43Audit>audits;
 void features(uint32_t node,const float*s,float fresh,double*x,double&ds,uint32_t&origin)const{
  std::fill(x,x+E43F,0.);double p=std::min(1.-1e-7,std::max(1e-7,double(fresh)));x[0]=std::log(p/(1.-p));for(int j=0;j<17;++j)x[j+1]=(double(s[j])-bank->mean[j])/bank->denom[j];
  ds=0.;origin=std::numeric_limits<uint32_t>::max();auto it=bank->nodes.find(node);if(it==bank->nodes.end())return;const auto&n=it->second;
  x[18]=!n.skips.empty();x[19]=std::log1p(double(n.total));x[20]=std::log1p(double(n.skips.size()));x[21]=double(n.skips.size())/std::max(double(n.total),1.);
  if(n.skips.empty())return;double best=std::numeric_limits<double>::infinity();
  for(const auto&r:n.skips){if(r.origin>bank->completed||r.origin>=seq+1)throw std::runtime_error("future or current query history used");double d=0;for(int j=0;j<17;++j){double a=x[j+1]-r.z[j];d+=a*a;}if(d<best){best=d;origin=r.origin;}}
  ds=std::sqrt(best);x[22]=std::log1p(ds);x[25]=ds<bank->scut;
 }
};
inline thread_local E43Context*e43_context=nullptr;
inline void e43_apply(uint32_t node,const float*s,float&score){
 auto*c=e43_context;if(!c)return;uint32_t ordinal=c->ordinal++;if(s[10]!=0)return;
 const float fresh=score;const bool original_skip=fresh<c->tau;++c->model_calls;
 E43Pending p{};p.node=node;p.skip=original_skip;p.fresh=fresh;std::copy(s,s+17,p.s);c->pending.push_back(p);
 if(original_skip){++c->original_skips;return;}++c->original_keeps;
 auto it=c->bank->nodes.find(node);bool H=it!=c->bank->nodes.end()&&!it->second.skips.empty();c->conflicts+=H;
 if(!c->audit&&c->kind==0)return;if(!c->audit&&!H&&c->kind!=6)return;
 double x[E43F],ds;uint32_t origin;c->features(node,s,fresh,x,ds,origin);
 if(c->audit){E43Audit a{};a.row=c->row;a.node=node;a.ordinal=ordinal;std::copy(x,x+E43F,a.x);for(int m=0;m<2;++m)a.logits[m]=c->bank->logit(m,x);c->audits.push_back(a);return;}
 ++c->secondary_evaluations;double risk=0;bool skip=false;
 if(c->kind==1)skip=true;
 else if(c->kind>=2&&c->kind<=4){++c->risk_calls;int m=c->kind==2?0:1;risk=E43Bank::sigmoid(c->bank->logit(m,x));skip=c->cutoff>0&&risk<=c->cutoff;}
 else if(c->kind==5||c->kind==6){risk=fresh;skip=risk<=c->cutoff;}
 else throw std::runtime_error("unknown policy");
 E43Decision d{};d.seq=c->seq;d.row=c->row;d.node=node;d.ordinal=ordinal;d.history_version=c->bank->completed;d.nearest_origin=origin;d.action=skip;d.kind=c->kind;d.fresh=fresh;d.risk=risk;d.distance=ds;
 if(it!=c->bank->nodes.end()){d.node_total=it->second.total;d.node_skips=it->second.skips.size();}std::copy(x,x+E43F,d.x);c->decisions.push_back(d);
 if(skip){score=0;++c->secondary_skips;}
}
inline void e43_commit(E43Context&c,bool enabled,std::vector<E43Update>&updates){
 if(c.bank->completed!=c.seq)throw std::runtime_error("query commit ordering");
 if(enabled)for(const auto&p:c.pending){auto&n=c.bank->nodes[p.node];++n.total;++c.bank->observations;c.bank->hash(&p,sizeof(p));
  if(p.skip){if(!(p.fresh<c.tau))throw std::runtime_error("secondary Skip inserted as original Skip");if(n.skips.empty())++c.bank->marked_nodes;E43Skip r{};for(int j=0;j<17;++j)r.z[j]=(double(p.s[j])-c.bank->mean[j])/c.bank->denom[j];r.origin=c.seq+1;n.skips.push_back(r);++c.bank->skip_records;
   E43Update u{};u.seq=c.seq;u.row=c.row;u.node=p.node;u.fresh=p.fresh;std::copy(p.s,p.s+17,u.s);updates.push_back(u);
  }
 }
 ++c.bank->completed;
}
}
