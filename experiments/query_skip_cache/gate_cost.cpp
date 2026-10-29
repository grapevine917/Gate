#include "online_gate.h"
#include <chrono>
#include <iostream>
int main(int argc,char**argv){
 if(argc!=3)return 2;mkl_set_num_threads_local(1);diskann::OnlineGate model(argv[1],"",0);
 if(model.uses_vector_features() || model.query_feature_width())throw std::runtime_error("state model required");
 std::ifstream f(argv[2],std::ios::binary);uint32_t n,d;f.read((char*)&n,4);f.read((char*)&d,4);if(!f || d!=17 || !n)throw std::runtime_error("states");std::vector<float>x(size_t(n)*d);f.read((char*)x.data(),x.size()*4);if(!f)throw std::runtime_error("truncated states");
 float query[128]={};volatile double sink=0;
 for(uint32_t i=0;i<n;++i)sink+=model.score(query,0,x.data()+size_t(i)*d);
 std::vector<double> us;
 for(int rep=0;rep<5;++rep){auto begin=std::chrono::steady_clock::now();for(int round=0;round<20;++round)for(uint32_t i=0;i<n;++i)sink+=model.score(query,0,x.data()+size_t(i)*d);us.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count()/(20*n));}
 std::sort(us.begin(),us.end());std::cout<<"{\"calls_per_repeat\":"<<20*n<<",\"repeats\":5,\"median_us_per_score\":"<<us[2]<<",\"min_us\":"<<us.front()<<",\"max_us\":"<<us.back()<<",\"checksum\":"<<sink<<"}\n";
}
