#include "online_gate.h"
#include <iostream>
int main(int argc,char**argv){
 if(argc!=3)return 2;diskann::OnlineGate model(argv[1],"",0);std::ifstream f(argv[2],std::ios::binary);uint32_t n;f.read((char*)&n,4);float row[18],q[128]={},err=0;
 for(unsigned i=0;i<n;++i){f.read((char*)row,sizeof(row));err=std::max(err,std::abs(model.score(q,0,row)-row[17]));}if(!f)return 3;std::cout<<err<<std::endl;return err>2e-5?4:0;
}
