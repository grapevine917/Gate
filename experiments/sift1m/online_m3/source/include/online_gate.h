#pragma once
#include <mkl.h>
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>
namespace diskann {
class OnlineGate {
    std::vector<float> mean, scale, vectors;
    std::array<std::vector<float>,3> weights, biases;
    void read(std::ifstream& f, std::vector<float>& v, size_t n) {
        v.resize(n); f.read(reinterpret_cast<char*>(v.data()),n*sizeof(float));
        if(!f) throw std::runtime_error("Truncated online MLP file");
    }
public:
    float threshold=0;
    OnlineGate(const std::string& model, const std::string& base, float tau):threshold(tau) {
        std::ifstream f(model,std::ios::binary); char magic[8]; f.read(magic,8);
        if(std::string(magic,8)!="M3GATE01") throw std::runtime_error("Invalid M3 export");
        read(f,mean,529); read(f,scale,529);
        const int dims[]={529,128,64,1};
        for(int l=0;l<3;++l){read(f,weights[l],dims[l]*dims[l+1]);read(f,biases[l],dims[l+1]);}
        std::ifstream b(base,std::ios::binary); uint32_t n=0,d=0;
        b.read(reinterpret_cast<char*>(&n),4); b.read(reinterpret_cast<char*>(&d),4);
        if(n!=1000000 || d!=128) throw std::runtime_error("Expected SIFT1M 128d vectors");
        read(b,vectors,size_t(n)*d);
        mkl_set_num_threads_local(1);
    }
    float score(const float* q,uint32_t node,const float* state) const {
        if(node>=1000000) throw std::runtime_error("Invalid M3 node ID");
        std::array<float,529> x; std::array<float,128> a; std::array<float,64> b;
        const float* v=vectors.data()+size_t(node)*128;
        for(int i=0;i<128;++i){x[i]=q[i];x[128+i]=v[i];x[256+i]=std::abs(q[i]-v[i]);x[384+i]=q[i]*v[i];}
        for(int i=0;i<17;++i) x[512+i]=state[i];
        for(int i=0;i<529;++i) x[i]=(x[i]-mean[i])/scale[i];
        std::copy(biases[0].begin(),biases[0].end(),a.begin());
        std::copy(biases[1].begin(),biases[1].end(),b.begin());
        cblas_sgemv(CblasRowMajor,CblasTrans,529,128,1,weights[0].data(),128,x.data(),1,1,a.data(),1);
        for(auto& z:a) z=std::max(0.0f,z);
        cblas_sgemv(CblasRowMajor,CblasTrans,128,64,1,weights[1].data(),64,a.data(),1,1,b.data(),1);
        for(auto& z:b) z=std::max(0.0f,z);
        float z=biases[2][0];
        cblas_sgemv(CblasRowMajor,CblasTrans,64,1,1,weights[2].data(),1,b.data(),1,1,&z,1);
        return 1.0f/(1.0f+std::exp(-z));
    }
};
inline thread_local const OnlineGate* online_gate=nullptr;
inline thread_local float online_last_score=0;
inline thread_local std::vector<float>* online_audit=nullptr;
}
