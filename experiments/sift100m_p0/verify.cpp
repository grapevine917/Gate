#include "online_gate.h"
uint32_t exact_node(const float* r){uint32_t n;std::memcpy(&n,r+128,4);return n;}
#include <iostream>
int main(int argc,char**argv){
 if(argc!=3)return 2;mkl_set_num_threads_local(1);std::string m=argv[1];diskann::OnlineGate lr(m+"/lr.bin","",0),gb(m+"/xgb.bin","",0),st(m+"/state.bin","",0),qv(m+"/qv.bin",argv[2],0),et(m+"/et.bin","",0);diskann::OnlineGate*models[]={&lr,&gb,&st,&qv};float err[5]={0};
 std::ifstream f(m+"/native_check.bin",std::ios::binary);uint32_t n;f.read((char*)&n,4);float row[150];
 for(unsigned i=0;i<n;++i){f.read((char*)row,sizeof(row));for(int k=0;k<4;++k)err[k]=std::max(err[k],std::abs(models[k]->score(row,exact_node(row),row+129)-row[146+k]));}if(!f)return 3;
 std::ifstream g(m+"/et_check.bin",std::ios::binary);g.read((char*)&n,4);float a[18],q[128]={};for(unsigned i=0;i<n;++i){g.read((char*)a,sizeof(a));err[4]=std::max(err[4],std::abs(et.score(q,0,a)-a[17]));}if(!g)return 3;
 std::cout<<"{\"LR\":"<<err[0]<<",\"XGB\":"<<err[1]<<",\"StateMLP\":"<<err[2]<<",\"QV_MLP\":"<<err[3]<<",\"ET\":"<<err[4]<<",\"samples_each\":2000}\n";for(float x:err)if(x>2e-5)return 4;
}
