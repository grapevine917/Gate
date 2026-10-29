#pragma once
#include <mkl.h>
#include <array>
#include <vector>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cmath>
#include <algorithm>
namespace diskann {
class CounterfactualModel {
 std::array<float,17> mean,scale;
 float gain_mean,gain_scale;
 std::array<std::vector<float>,3> weights,bias;
 public:
 float rho=0,gmin=0;
 CounterfactualModel(const std::string& path,float risk_limit,float gain_limit):rho(risk_limit),gmin(gain_limit) {
  std::ifstream f(path,std::ios::binary);char magic[8];f.read(magic,8);
  if(!f||std::string(magic,8)!="CFGATE01")throw std::runtime_error("Invalid counterfactual model");
  f.read((char*)mean.data(),68);f.read((char*)scale.data(),68);f.read((char*)&gain_mean,4);f.read((char*)&gain_scale,4);
  const int dims[]={17,128,64,2};
  for(int l=0;l<3;++l){weights[l].resize(dims[l]*dims[l+1]);bias[l].resize(dims[l+1]);f.read((char*)weights[l].data(),weights[l].size()*4);f.read((char*)bias[l].data(),bias[l].size()*4);}
  if(!f || f.peek()!=std::ifstream::traits_type::eof())throw std::runtime_error("Truncated model or trailing bytes");
 }
 std::array<float,2> predict(const float* state) const {
  std::array<float,17>x;std::array<float,128>a;std::array<float,64>b;std::array<float,2>out;
  for(int j=0;j<17;++j)x[j]=(state[j]-mean[j])/scale[j];
  std::copy(bias[0].begin(),bias[0].end(),a.begin());std::copy(bias[1].begin(),bias[1].end(),b.begin());std::copy(bias[2].begin(),bias[2].end(),out.begin());
  cblas_sgemv(CblasRowMajor,CblasTrans,17,128,1,weights[0].data(),128,x.data(),1,1,a.data(),1);
  for(auto& v:a)v=std::max(0.f,v);
  cblas_sgemv(CblasRowMajor,CblasTrans,128,64,1,weights[1].data(),64,a.data(),1,1,b.data(),1);
  for(auto& v:b)v=std::max(0.f,v);
  cblas_sgemv(CblasRowMajor,CblasTrans,64,2,1,weights[2].data(),2,b.data(),1,1,out.data(),1);
  out[0]=1.f/(1.f+std::exp(-out[0]));out[1]=out[1]*gain_scale+gain_mean;return out;
 }
 float retain_score(const float* state) const {
  if(state[10]!=0)return 1;
  const auto p=predict(state);return p[0]<rho && p[1]>gmin?0.f:1.f;
 }
};
inline thread_local const CounterfactualModel* counterfactual_model=nullptr;
}
