#include "common_includes.h"
#include "online_gate.h"
#include <fstream>
int main(int argc,char**argv){
 if(argc!=4)return 2;diskann::OnlineGate model(argv[1],"",0);std::ifstream in(argv[2],std::ios::binary);std::ofstream out(argv[3],std::ios::binary);float row[20],q[256]={};
 while(in.read((char*)row,sizeof(row))){float score=model.score(q,0,row+3);out.write((char*)&score,4);}if(!in.eof())return 3;
}
