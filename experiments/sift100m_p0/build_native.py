"""Isolated uint8 DiskANN runner, lossless q-v reader and exact node-ID audit."""
from pathlib import Path
import json,shutil,subprocess
R=Path(__file__).resolve().parent;M=R.parent/'main_comparison';N=R/'native';D=R.parent/'deep_p0'
header=(M/'native/online_gate.h').read_text().replace('#include <mkl.h>','#include <mkl.h>\n#include <cstring>\n#include <fcntl.h>\n#include <unistd.h>\n#include <cerrno>\nextern "C" { size_t ZSTD_decompress(void*,size_t,const void*,size_t); unsigned ZSTD_isError(size_t); }')
header=header.replace('std::vector<float> mean, scale, vectors;','std::vector<float> mean, scale;\n    std::vector<uint64_t> vector_offsets;\n    std::vector<uint8_t> compressed_vectors;')
header=header.replace('M3GATE01','S3GATE01')
start=header.index('        std::ifstream b(base,');end=header.index('\n        }\n        mkl_set_num_threads_local',start)
header=header[:start]+(R/'resident_vector_loader.inc').read_text()+header[end:]
header=header.replace('if(node>=1000000)','if(node>=100000000)').replace('const float* v=vectors.data()+size_t(node)*128;','''thread_local std::array<uint8_t,4096> decoded;
        const uint32_t block=node/32;const uint64_t off=vector_offsets[block],len=vector_offsets[block+1]-off;
        size_t actual=ZSTD_decompress(decoded.data(),decoded.size(),compressed_vectors.data()+off,len);
        if(ZSTD_isError(actual) || actual!=decoded.size())throw std::runtime_error("Invalid decoded vector block");
        const uint8_t* v=decoded.data()+(node%32)*128;''')
header=header.replace('inline thread_local std::vector<float>* online_audit=nullptr;','inline thread_local std::vector<float>* online_audit=nullptr;\ninline thread_local std::vector<uint32_t>* online_audit_nodes=nullptr;')
(N/'online_gate.h').write_text(header);shutil.copy(M/'native/online_termination.h',N/'online_termination.h')
pq=(M/'native/pq_flash_index.cpp').read_text().replace('online_audit->push_back(float(nbr.id));','if(online_audit_nodes)online_audit_nodes->push_back(nbr.id);\n                    online_audit->push_back(0); // exact IDs are exclusively uint32 sidecar')
(N/'pq_flash_index.cpp').write_text(pq)
def uint8_native(s):return s.replace('float*q=nullptr,*gd=nullptr;','uint8_t*q=nullptr;float*gd=nullptr;').replace('load_aligned_bin<float>','load_aligned_bin<uint8_t>').replace('PQFlashIndex<float,uint32_t>','PQFlashIndex<uint8_t,uint32_t>')
collect=uint8_native((M/'collect.cpp').read_text())
collect=collect.replace('std::ofstream p(', 'std::ofstream nodes_out(std::string(argv[6])+".prune_nodes.u32",std::ios::binary);\n std::ofstream p(')
collect=collect.replace('std::vector<std::vector<float>>pr(nq),et(nq);','std::vector<std::vector<float>>pr(nq),et(nq);std::vector<std::vector<uint32_t>> exact_nodes(nq);')
collect=collect.replace('std::vector<float>audit;','std::vector<float>audit;std::vector<uint32_t>audit_nodes;')
collect=collect.replace('diskann::online_audit=&audit;','diskann::online_audit=&audit;diskann::online_audit_nodes=&audit_nodes;').replace('diskann::online_audit=nullptr;','diskann::online_audit=nullptr;diskann::online_audit_nodes=nullptr;\n   if(audit.size()/19!=audit_nodes.size())throw std::runtime_error("node audit mismatch");')
collect=collect.replace('if(audit[j*19+11]==0)ix.push_back(j);','if(cap>32 || audit[j*19+11]==0)ix.push_back(j);')
collect=collect.replace('auto node=uint32_t(audit[j*19]);','auto node=audit_nodes[j];exact_nodes[qi].push_back(node);').replace('pr[qi].push_back(float(node));','pr[qi].push_back(0); // node ID stored in exact uint32 sidecar\n')
collect=collect.replace('p.write((char*)pr[qi].data(),pr[qi].size()*4);','nodes_out.write((char*)exact_nodes[qi].data(),exact_nodes[qi].size()*4);p.write((char*)pr[qi].data(),pr[qi].size()*4);')
(R/'collect.cpp').write_text(collect)
search=uint8_native((D/'search.cpp').read_text())
search=search.replace('#include <chrono>','#include <chrono>\n#include <sstream>\nuint64_t proc_kb(const std::string& key){std::ifstream f("/proc/self/status");std::string line;while(std::getline(f,line)){std::istringstream s(line);std::string k;uint64_t v;if(s>>k>>v && k==key+":")return v;}throw std::runtime_error("Missing process memory statistic");}')
search=search.replace('reader->requests=0;reader->bytes=0;uint64_t physical_before=physical_bytes();','uint64_t rss_before=proc_kb("VmRSS"),swap_before=proc_kb("VmSwap");if(swap_before)throw std::runtime_error("Swapped process cannot be timed as resident");\n  reader->requests=0;reader->bytes=0;uint64_t physical_before=physical_bytes();')
search=search.replace('uint64_t physical_delta=physical_bytes()-physical_before;','uint64_t physical_delta=physical_bytes()-physical_before;uint64_t rss_after=proc_kb("VmRSS"),swap_after=proc_kb("VmSwap");if(swap_after)throw std::runtime_error("Swap detected during timed search");')
search=search.replace('<<physical_delta<<"}\\n"','<<physical_delta<<",\\"rss_before_kib\\":"<<rss_before<<",\\"rss_after_kib\\":"<<rss_after<<",\\"peak_rss_kib\\":"<<proc_kb("VmHWM")<<",\\"swap_before_kib\\":"<<swap_before<<",\\"swap_after_kib\\":"<<swap_after<<"}\\n"')
(R/'search.cpp').write_text(search)
for name in ['verify','score']:
 s=(M/'verify.cpp' if name=='verify' else R.parent/'p0_c/score.cpp').read_text()
 s=s.replace('#include "online_gate.h"','#include "online_gate.h"\nuint32_t exact_node(const float* r){uint32_t n;std::memcpy(&n,r+128,4);return n;}')
 s=s.replace('uint32_t(row[128])','exact_node(row)').replace('uint32_t(r[128])','exact_node(r)')
 (R/f'{name}.cpp').write_text(s)
template=json.loads((R.parent/'p0_b_lr/build.command.json').read_text());si=template.index(str(R.parent/'p0_b_lr/search.cpp'))
prefix=[x.replace(str(M/'native'),str(N)) for x in template[:si]];libs=[x.replace(str(M/'native'),str(N)) for x in template[template.index('-L/opt/intel/oneapi/mkl/latest/lib/intel64'):]]+['-l:libzstd.so.1'];cmds=[]
def build(cmd,name):
 cmds.append(cmd);(R/'build.commands.json').write_text(json.dumps(cmds,indent=2)+'\n')
 with (R/name).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
build([*prefix,'-c',str(N/'pq_flash_index.cpp'),'-o',str(N/'pq_flash_index.cpp.o')],'build_index.log')
shutil.copy(M/'native/libdiskann.a',N/'libdiskann.a');build(['ar','r',str(N/'libdiskann.a'),str(N/'pq_flash_index.cpp.o')],'build_archive.log')
for name in ['collect','search','verify','score']:build([*prefix,str(R/f'{name}.cpp'),'-o',str(N/name),*libs],f'build_{name}.log')
(R/'NATIVE_COMPLETE.json').write_text(json.dumps(dict(complete=True,query_dtype='uint8',node_id_dtype='uint32',qv_model_magic='S3GATE01',qv_vector_format='CVECU801',qv_block_vectors=32),indent=2)+'\n')
print('NATIVE COMPLETE',flush=True)
