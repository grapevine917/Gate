#include "online_gate.h"
#include <iostream>
int main(int argc,char**argv){
 if(argc!=4)return 2;
 // Check that current decision is excluded, earlier decisions included, and reset is query-local.
 diskann::OnlineHistory h;float s[17]={};s[0]=10;float x[27];h.features(s,x);
 if(x[0]!=0 || x[5]!=0)throw std::runtime_error("initial history");
 h.update(s,true);s[0]=20;h.features(s,x);
 if(x[0]!=1 || x[1]!=1 || x[6]!=10 || x[10]!=10 || x[13]!=10)throw std::runtime_error("causal history");
 h.update(s,false);s[0]=30;h.features(s,x);
 if(x[0]!=2 || x[1]!=1 || x[3]!=0 || x[4]!=1 || x[6]!=10)throw std::runtime_error("kept history");
 diskann::OnlineHistory fresh;fresh.features(s,x);if(x[0]!=0)throw std::runtime_error("query reset");
 diskann::OnlineGate model(argv[1],"",0);std::ifstream f(argv[2],std::ios::binary);uint32_t n,d;f.read((char*)&n,4);f.read((char*)&d,4);if(d!=44)throw std::runtime_error("width");
 std::ofstream out(argv[3],std::ios::binary);std::vector<float> row(d);float q[128]={};
 for(uint32_t i=0;i<n;++i){f.read((char*)row.data(),d*4);if(!f)throw std::runtime_error("truncated matrix");float p=model.score(q,0,row.data());out.write((char*)&p,4);}
 std::cout<<"native history invariants and "<<n<<" scores checked\n";
}
