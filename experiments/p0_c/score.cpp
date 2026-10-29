#include "online_gate.h"
#include <chrono>
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>
int main(int argc,char**argv){
 if(argc!=6)return 2;mkl_set_num_threads_local(1);diskann::OnlineGate model(argv[1],argv[2],0);
 std::ifstream f(argv[3],std::ios::binary);f.seekg(0,std::ios::end);size_t bytes=f.tellg();f.seekg(0);if(bytes%(147*4))return 3;std::vector<float> rows(bytes/4);f.read((char*)rows.data(),bytes);size_t n=rows.size()/147;
 std::vector<float> scores(n);for(size_t i=0;i<n;++i){float*r=rows.data()+i*147;scores[i]=model.score(r,uint32_t(r[128]),r+129);}
 std::ofstream out(argv[4],std::ios::binary);out.write((char*)scores.data(),scores.size()*4);out.close();
 std::vector<size_t>ix(n);std::iota(ix.begin(),ix.end(),0);std::mt19937 rng(20261102);std::shuffle(ix.begin(),ix.end(),rng);ix.resize(std::min(n,size_t(20000)));volatile float sink=0;
 for(size_t j=0;j<std::min(ix.size(),size_t(2000));++j){float*r=rows.data()+ix[j]*147;sink=model.score(r,uint32_t(r[128]),r+129);}
 std::ofstream bench(argv[5]);bench<<std::setprecision(14)<<"{\"samples\":"<<ix.size()<<",\"ns_per_score\":[";
 for(int rep=0;rep<5;++rep){std::shuffle(ix.begin(),ix.end(),rng);auto start=std::chrono::steady_clock::now();for(auto i:ix){float*r=rows.data()+i*147;sink=model.score(r,uint32_t(r[128]),r+129);}double ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/ix.size();if(rep)bench<<',';bench<<ns;}
 bench<<"],\"sink\":"<<sink<<"}\n";
}
