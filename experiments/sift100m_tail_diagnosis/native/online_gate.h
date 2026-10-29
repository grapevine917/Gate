#pragma once
#include <mkl.h>
#include <cstring>
extern "C" { size_t ZSTD_decompress(void*,size_t,const void*,size_t); unsigned ZSTD_isError(size_t); }
#include <algorithm>
#include <cstdint>
#include <string>
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <unordered_set>
#include <memory>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
namespace diskann {
class OnlineGate {
    bool state_only=false;
    bool linear_model=false, tree_model=false, history_model=false;
    uint32_t query_width=0;
    std::vector<int32_t> tree_feature,tree_left,tree_right;
    std::vector<float> tree_threshold,tree_safe;
    float linear_bias=0;
    int input_width=529;
    std::vector<uint32_t> state_columns;
    std::vector<float> mean, scale;
    std::vector<uint64_t> vector_offsets;
    std::vector<uint8_t> compressed_vectors;
    std::array<std::vector<float>,3> weights, biases;
    void read(std::ifstream& f, std::vector<float>& v, size_t n) {
        v.resize(n); f.read(reinterpret_cast<char*>(v.data()),n*sizeof(float));
        if(!f) throw std::runtime_error("Truncated online MLP file");
    }
public:
    bool forest_model=false;
    struct ForestNode { int left,right,feature; float value; };
    std::vector<std::vector<ForestNode>> forest;
    float forest_bias=0;
    float threshold=0;
    bool uses_vector_features() const { return !state_only; }
    uint32_t query_feature_width() const { return query_width; }
    OnlineGate(const std::string& model, const std::string& base, float tau):threshold(tau) {
        std::ifstream f(model,std::ios::binary); char magic[8]; f.read(magic,8);
        if(!f) throw std::runtime_error("Cannot read online gate model");
        const std::string model_magic(magic,8);
        if(model_magic=="XGGATE01") {
            forest_model=true;state_only=true;uint32_t nt;
            f.read(reinterpret_cast<char*>(&forest_bias),4);f.read(reinterpret_cast<char*>(&nt),4);
            if(nt>10000)throw std::runtime_error("Invalid forest");
            forest.resize(nt);for(auto& t:forest){uint32_t nn;f.read(reinterpret_cast<char*>(&nn),4);if(nn>100000)throw std::runtime_error("Invalid tree");t.resize(nn);f.read(reinterpret_cast<char*>(t.data()),nn*sizeof(ForestNode));}
            if(!f)throw std::runtime_error("Truncated forest");return;
        }
        if(model_magic=="LRGATE01" || model_magic=="TRGATE01" || model_magic=="HLGATE01") {
            history_model=model_magic=="HLGATE01";
            linear_model=model_magic=="LRGATE01" || history_model;tree_model=!linear_model;state_only=true;
            f.read(reinterpret_cast<char*>(&query_width),4);
            if(!f || query_width>1024)throw std::runtime_error("Invalid query feature width");
            input_width=history_model?int(query_width):17+query_width;
            if(history_model && input_width!=44)throw std::runtime_error("Expected 44 history inputs");
            if(history_model)query_width=0;
            if(linear_model){read(f,mean,input_width);read(f,scale,input_width);read(f,weights[0],input_width);f.read(reinterpret_cast<char*>(&linear_bias),4);}
            else {
                uint32_t n=0;f.read(reinterpret_cast<char*>(&n),4);
                if(!f || n<1 || n>100000)throw std::runtime_error("Invalid tree node count");
                tree_feature.resize(n);tree_left.resize(n);tree_right.resize(n);
                f.read(reinterpret_cast<char*>(tree_feature.data()),n*4);f.read(reinterpret_cast<char*>(tree_left.data()),n*4);f.read(reinterpret_cast<char*>(tree_right.data()),n*4);
                read(f,tree_threshold,n);read(f,tree_safe,n);
                for(size_t j=0;j<n;++j)if(tree_feature[j]>=input_width || (tree_feature[j]>=0 && (tree_left[j]<0 || tree_right[j]<0 || tree_left[j]>=int(n) || tree_right[j]>=int(n))))throw std::runtime_error("Invalid tree");
            }
            if(!f)throw std::runtime_error("Truncated classifier model");
            return;
        }
        const bool ablation=std::string(magic,8)=="ABGATE01";
        state_only=std::string(magic,8)=="STGATE01" || ablation;
        if(!state_only && std::string(magic,8)!="S3GATE01") throw std::runtime_error("Invalid MLP export");
        input_width=state_only?17:529;
        if(ablation) {
            uint32_t count=0; f.read(reinterpret_cast<char*>(&count),4);
            if(!f || count<1 || count>17) throw std::runtime_error("Invalid ablation width");
            input_width=int(count); state_columns.resize(count);
            f.read(reinterpret_cast<char*>(state_columns.data()),count*sizeof(uint32_t));
            if(!f) throw std::runtime_error("Truncated ablation feature map");
            for(auto col:state_columns) if(col>=17) throw std::runtime_error("Invalid state column");
        }
        read(f,mean,input_width); read(f,scale,input_width);
        const int dims[]={input_width,128,64,1};
        for(int l=0;l<3;++l){read(f,weights[l],dims[l]*dims[l+1]);read(f,biases[l],dims[l+1]);}
        if(!state_only) {
        std::ifstream b(base,std::ios::binary);char vm[8];uint32_t n,d,bs,nb;uint64_t bytes;
        b.read(vm,8);b.read((char*)&n,4);b.read((char*)&d,4);b.read((char*)&bs,4);b.read((char*)&nb,4);b.read((char*)&bytes,8);
        if(!b || std::string(vm,8)!="CVECU801" || n!=100000000 || d!=128 || bs!=32 || nb!=3125000 || bytes>12800000000ULL)throw std::runtime_error("Invalid SIFT100M compressed vector store");
        vector_offsets.resize(size_t(nb)+1);b.read((char*)vector_offsets.data(),vector_offsets.size()*8);
        if(!b || vector_offsets[0]!=0 || vector_offsets.back()!=bytes || !std::is_sorted(vector_offsets.begin(),vector_offsets.end()))throw std::runtime_error("Invalid vector block offsets");
        const auto payload_offset=static_cast<off_t>(b.tellg());
        b.seekg(0,std::ios::end);
        if(!b || static_cast<uint64_t>(b.tellg())!=static_cast<uint64_t>(payload_offset)+bytes)throw std::runtime_error("Invalid compressed vector length");
        b.close();
        // Keep the original ABI; discard the redundant file cache while
        // building the resident heap copy. All I/O precedes timed searches.
        int fd=::open(base.c_str(),O_RDONLY);
        if(fd<0)throw std::runtime_error("Cannot open compressed vectors");
        try {
            if(::posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED)!=0)throw std::runtime_error("Cannot release old vector file cache");
            compressed_vectors.resize(bytes);
            constexpr uint64_t chunk=16ULL*1024*1024;
            uint64_t done=0;
            while(done<bytes) {
                const uint64_t end=std::min(bytes,done+chunk);
                const off_t start=payload_offset+static_cast<off_t>(done);
                while(done<end) {
                    const ssize_t got=::pread(fd,compressed_vectors.data()+done,end-done,payload_offset+static_cast<off_t>(done));
                    if(got<0 && errno==EINTR)continue;
                    if(got<=0)throw std::runtime_error("Truncated compressed vector payload");
                    done+=static_cast<uint64_t>(got);
                }
                const off_t page=::sysconf(_SC_PAGESIZE);
                const off_t aligned_start=(start/page)*page;
                const off_t aligned_end=((payload_offset+static_cast<off_t>(done)+page-1)/page)*page;
                if(::posix_fadvise(fd,aligned_start,aligned_end-aligned_start,POSIX_FADV_DONTNEED)!=0)throw std::runtime_error("Cannot release copied vector file cache");
            }
        } catch(...) {::close(fd);throw;}
        ::close(fd);

        }
        mkl_set_num_threads_local(1);
    }
    float score(const float* q,uint32_t node,const float* state) const {
        if(forest_model) {
            if(state[10]!=0)return 1;
            float z=forest_bias;
            for(const auto& t:forest){int i=0;while(t[i].left>=0)i=state[t[i].feature]<t[i].value?t[i].left:t[i].right;z+=t[i].value;}
            return 1.0f/(1.0f+std::exp(-z));
        }
        if(linear_model || tree_model) {
            // This experiment targets cache misses; keep all cached candidates.
            if(state[10]!=0)return 1;
            auto feature=[&](int j){return history_model?state[j]:(j<17?state[j]:q[j-17]);};
            if(linear_model){double z=linear_bias;for(int j=0;j<input_width;++j)z+=double(weights[0][j])*((double(feature(j))-mean[j])/scale[j]);return float(1.0/(1.0+std::exp(z)));}
            size_t node_idx=0;for(size_t depth=0;depth<tree_feature.size();++depth){int j=tree_feature[node_idx];if(j<0)return 1-tree_safe[node_idx];node_idx=feature(j)<=tree_threshold[node_idx]?tree_left[node_idx]:tree_right[node_idx];}
            throw std::runtime_error("Cyclic classifier tree");
        }
        std::array<float,529> x; std::array<float,128> a; std::array<float,64> b;
        if(state_only) {
            if(state_columns.empty()) std::copy(state,state+17,x.begin());
            else for(int i=0;i<input_width;++i) x[i]=state[state_columns[i]];
        } else {
        if(node>=100000000) throw std::runtime_error("Invalid M3 node ID");
        thread_local std::array<uint8_t,4096> decoded;
        const uint32_t block=node/32;const uint64_t off=vector_offsets[block],len=vector_offsets[block+1]-off;
        size_t actual=ZSTD_decompress(decoded.data(),decoded.size(),compressed_vectors.data()+off,len);
        if(ZSTD_isError(actual) || actual!=decoded.size())throw std::runtime_error("Invalid decoded vector block");
        const uint8_t* v=decoded.data()+(node%32)*128;
        for(int i=0;i<128;++i){x[i]=q[i];x[128+i]=v[i];x[256+i]=std::abs(q[i]-v[i]);x[384+i]=q[i]*v[i];}
        for(int i=0;i<17;++i) x[512+i]=state[i];
        }
        for(int i=0;i<input_width;++i) x[i]=(x[i]-mean[i])/scale[i];
        std::copy(biases[0].begin(),biases[0].end(),a.begin());
        std::copy(biases[1].begin(),biases[1].end(),b.begin());
        cblas_sgemv(CblasRowMajor,CblasTrans,input_width,128,1,weights[0].data(),128,x.data(),1,1,a.data(),1);
        for(auto& z:a) z=std::max(0.0f,z);
        cblas_sgemv(CblasRowMajor,CblasTrans,128,64,1,weights[1].data(),64,a.data(),1,1,b.data(),1);
        for(auto& z:b) z=std::max(0.0f,z);
        float z=biases[2][0];
        cblas_sgemv(CblasRowMajor,CblasTrans,64,1,1,weights[2].data(),1,b.data(),1,1,&z,1);
        return 1.0f/(1.0f+std::exp(-z));
    }
};
// Query-local history is updated only AFTER each candidate decision. No labels enter it.
struct OnlineHistory {
    uint32_t seen=0, skipped=0, streak=0, since_skip=0;
    double sum=0, sumsq=0; float lo=0,hi=0,last=0;
    std::array<float,17> prev{}; bool has_prev=false;
    void features(const float* s,float* out) const {
        float m=skipped?float(sum/skipped):0;
        out[0]=float(seen);out[1]=float(skipped);out[2]=seen?float(skipped)/seen:0;
        out[3]=float(streak);out[4]=float(since_skip);out[5]=skipped?1:0;
        out[6]=m;out[7]=skipped?float(std::sqrt(std::max(0.0,sumsq/skipped-double(m)*m))):0;
        out[8]=lo;out[9]=hi;out[10]=last;
        out[11]=skipped?s[0]-m:0;out[12]=skipped?float(double(s[0])/std::max(double(m),1e-12)):0;
        const int cols[]={0,1,2,3,4,5,6,7,8,11,13,14};
        for(int j=0;j<12;++j)out[13+j]=has_prev?s[cols[j]]-prev[cols[j]]:0;
        out[25]=has_prev?1:0;
        out[26]=(has_prev && s[8]>prev[8])?float((s[6]-prev[6])/std::max(1.0f,s[7]-prev[7])):0;
    }
    void update(const float* s,bool skip) {
        ++seen;
        if(skip){++skipped;++streak;since_skip=0;float d=s[0];sum+=d;sumsq+=double(d)*d;lo=skipped==1?d:std::min(lo,d);hi=std::max(hi,d);last=d;}
        else{streak=0;++since_skip;}
        std::copy(s,s+17,prev.begin());has_prev=true;
    }
};
inline thread_local std::vector<float>* online_history_audit=nullptr;
// Frozen, non-parametric per-node memory. No query labels or model scores.
class OnlineNodeMarks {
    uint32_t count=0;
    std::vector<uint64_t> bits;
public:
    explicit OnlineNodeMarks(const std::string& path, uint64_t expected_count=0) {
        std::ifstream f(path,std::ios::binary);char magic[8];f.read(magic,8);
        if(!f || std::string(magic,8)!="NDMARK01")throw std::runtime_error("Invalid node marks");
        f.read(reinterpret_cast<char*>(&count),4);
        if(!f || count==0 || (expected_count && count!=expected_count))throw std::runtime_error("Node mark count does not match index");
        bits.resize((size_t(count)+63)/64);
        f.read(reinterpret_cast<char*>(bits.data()),(size_t(count)+7)/8);
        if(!f || f.peek()!=std::ifstream::traits_type::eof())throw std::runtime_error("Invalid mark length");
    }
    bool contains(uint32_t node) const {
        if(node>=count)throw std::runtime_error("Invalid marked node ID");
        return (bits[node>>6]>>(node&63))&1;
    }
};
struct SkipMemoRecord { uint32_t node, rank, expansions; };
struct SkipMemoView {
    const std::vector<uint32_t>* pool=nullptr;
    const std::vector<SkipMemoRecord>* records=nullptr;
    bool guarded=false, lookup_only=false, audit=false;
    uint32_t reused=0, conflicts=0, membership_hits=0;
    bool matches(uint32_t node,uint32_t rank,uint32_t expansions) {
        auto it=std::lower_bound(pool->begin(),pool->end(),node);
        if(it==pool->end() || *it!=node)return false;
        uint32_t ref=uint32_t(it-pool->begin());
        auto r=std::lower_bound(records->begin(),records->end(),ref,[](const SkipMemoRecord& a,uint32_t b){return a.node<b;});
        if(r==records->end() || r->node!=ref)return false;
        ++membership_hits;
        return !guarded || (rank==r->rank && std::abs(int64_t(expansions)-r->expansions)<=4);
    }
};
inline thread_local SkipMemoView* online_skip_memo=nullptr;
inline thread_local std::vector<SkipMemoRecord>* online_skip_records=nullptr;
inline thread_local uint32_t* online_model_calls=nullptr;
inline thread_local bool online_miss_only=false;
inline thread_local const OnlineNodeMarks* online_node_marks=nullptr;
inline thread_local const OnlineGate* online_gate=nullptr;
// Optional second gate: skip when either retain score is below its threshold.
inline thread_local const OnlineGate* online_joint_gate=nullptr;
// Opt-in diagnosis: zero preserves permanent pruning. Limit counts deferred seeds,
// not all descendants expanded during the standard best-first rescue phase.
inline thread_local uint32_t online_defer_limit=0;
inline thread_local uint32_t online_defer_priority=0; // 0 PQ, 1 retain score descending, 2 original rank
inline thread_local const std::unordered_set<uint32_t>* online_protect=nullptr;
struct OnlineHeuristic {
    int rank_threshold;
    int expansion_threshold; // -1 reduces the conjunction to rank-only.
    bool skip(size_t rank,uint32_t completed) const {
        return int64_t(rank)>rank_threshold && int64_t(completed)>expansion_threshold;
    }
};
inline thread_local const OnlineHeuristic* online_heuristic=nullptr;
inline thread_local float online_last_score=0;
inline thread_local std::vector<float>* online_audit=nullptr;
inline thread_local std::vector<uint32_t>* online_audit_nodes=nullptr;
}

#include <chrono>
namespace diskann {
struct TailProfile {
 bool timers=false;
 double rank_us=0, feature_us=0, score_us=0;
 uint64_t rounds=0, empty_rounds=0, queue_slots=0;
 std::vector<uint8_t>* record=nullptr;
 const std::vector<uint8_t>* replay=nullptr;
 size_t replay_pos=0;
 std::vector<uint32_t> nodes;
 std::vector<uint8_t> skipped;
};
inline thread_local TailProfile* tail_profile=nullptr;
inline double tail_now() { return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}
