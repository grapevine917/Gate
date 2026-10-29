#include "online_gate.h"
#include <iostream>
int main(int argc,char**argv){if(argc!=4)return 2;mkl_set_num_threads_local(1);std::string m=argv[1];diskann::OnlineGate lr(m+"/lr.bin","",0),st(m+"/state.bin","",0);std::ifstream f(argv[2],std::ios::binary);std::ofstream g(argv[3],std::ios::binary);uint32_t n;f.read((char*)&n,4);g.write((char*)&n,4);float x[17],q[128]={},out[2];for(unsigned i=0;i<n;++i){f.read((char*)x,sizeof(x));out[0]=lr.score(q,0,x);out[1]=st.score(q,0,x);g.write((char*)out,sizeof(out));}if(!f)return 3;}
