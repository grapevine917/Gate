#include "risk_model.h"
#include <iostream>
int main(int argc,char**argv){
 if(argc!=3)return 2;mkl_set_num_threads_local(1);diskann::CounterfactualModel m(argv[1],0,0);
 std::ifstream f(argv[2],std::ios::binary);uint32_t n;f.read((char*)&n,4);std::vector<float>x(n*17),p(n*2);f.read((char*)x.data(),x.size()*4);f.read((char*)p.data(),p.size()*4);if(!f)return 3;
 double risk_error=0,gain_error=0;for(unsigned i=0;i<n;++i){auto a=m.predict(x.data()+i*17);risk_error=std::max(risk_error,double(std::abs(a[0]-p[i*2])));gain_error=std::max(gain_error,double(std::abs(a[1]-p[i*2+1])));}
 std::cout<<"{\"states\":"<<n<<",\"risk_max_abs_error\":"<<risk_error<<",\"gain_max_abs_error\":"<<gain_error<<"}\n";return risk_error<.0001 && gain_error<.0001?0:4;
}
