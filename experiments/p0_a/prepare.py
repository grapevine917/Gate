from pathlib import Path
import json, hashlib, subprocess
R=Path(__file__).resolve().parent; W=R.parents[1]; M=R.parent/'main_comparison'; S=R.parent/'sift1m'
source=(M/'search.cpp').read_text()
source=source.replace('int main(int argc,char**argv){', '''class MeteredReader : public LinuxAlignedFileReader {
public:
 std::atomic<uint64_t> requests{0},bytes{0};
 void read(std::vector<AlignedRead>& rs, IOContext& ctx, bool async=false) override {
  uint64_t n=0;for(const auto& r:rs)n+=r.len;
  requests.fetch_add(rs.size(),std::memory_order_relaxed);bytes.fetch_add(n,std::memory_order_relaxed);
  LinuxAlignedFileReader::read(rs,ctx,async);
 }
};
uint64_t physical_bytes(){std::ifstream f("/proc/self/io");std::string k;uint64_t v;while(f>>k>>v)if(k=="read_bytes:")return v;throw std::runtime_error("Missing read_bytes");}
int main(int argc,char**argv){''')
source=source.replace('std::shared_ptr<AlignedFileReader>reader=std::make_shared<LinuxAlignedFileReader>();','auto reader=std::make_shared<MeteredReader>();std::shared_ptr<AlignedFileReader> base_reader=reader;')
source=source.replace('index(reader,diskann::Metric::L2)', 'index(base_reader,diskann::Metric::L2)')
a=source.index(' std::string folder=argv[4];');b=source.index(' std::ifstream configs',a)
source=source[:a]+''' std::ofstream cache(std::string(argv[7])+"/cache_nodes.bin",std::ios::binary);cache.write((char*)nodes.data(),nodes.size()*4);cache.close();
 diskann::OnlineGate state(argv[4],"",0);
 diskann::OnlineGate* models[]={nullptr,nullptr,nullptr,nullptr,&state,nullptr,nullptr,nullptr};
'''+source[b:]
source=source.replace('et.threshold=et_tau;','').replace('diskann::online_termination=(kind==6||kind==7)?&et:nullptr;','diskann::online_termination=nullptr;')
source=source.replace('  auto begin=std::chrono::steady_clock::now();','  reader->requests=0;reader->bytes=0;uint64_t physical_before=physical_bytes();\n  auto begin=std::chrono::steady_clock::now();')
source=source.replace('  std::ofstream csv(prefix+', '  uint64_t physical_delta=physical_bytes()-physical_before;\n  std::ofstream csv(prefix+')
source=source.replace('<<double(nq)/seconds<<"}\\n";', '<<double(nq)/seconds<<",\\\"reader_requests\\\":"<<reader->requests.load()<<",\\\"reader_bytes\\\":"<<reader->bytes.load()<<",\\\"linux_read_bytes\\\":"<<physical_delta<<"}\\n";')
(R/'search.cpp').write_text(source)
commands=json.loads((M/'build.commands.json').read_text());c=next(c for c in commands if str(M/'search.cpp') in c);c=[str(R/'search.cpp') if x==str(M/'search.cpp') else str(R/'search') if x==str(M/'native/search') else x for x in c]
(R/'build.command.json').write_text(json.dumps(c,indent=2))
with (R/'build.log').open('w') as f:subprocess.run(c,stdout=f,stderr=subprocess.STDOUT,check=True)
inputs=[S/'online_state_mlp/model.bin',S/'online_state_mlp/thresholds.json',S/'top3_heuristic/selected_parameters.json',M/'data/test.fbin',M/'data/test.gt',M/'data/test_ids.npy',M/'native/libdiskann.a',R/'search.cpp',R/'search',S/'index_disk.index',S/'index_pq_pivots.bin',S/'index_pq_compressed.bin']
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
protocol=dict(dataset='SIFT1M',L=100,beam=4,K=10,threads=4,MKL_threads=1,cache_nodes=10000,queries=1000,query_file=str(M/'data/test.fbin'),gt_file=str(M/'data/test.gt'),index=str(S/'index'),model=str(S/'online_state_mlp/model.bin'),gate_threshold=json.loads((S/'online_state_mlp/thresholds.json').read_text())['fnr_1pct']['threshold'],default_definition='Predeclared original frozen 17D MLP at existing validation retain-FNR1% operating point; not a previously designated universal default. No new threshold tuning.',heuristic=dict(rank=72,completed_expansions=100,rule='skip iff rank>72 and completed_expansions>100; cache hits protected',source='existing Rank+Progress validation retain-FNR1% setting'),cache_hit_protection=True,repeats=9,order='3 cyclic orders repeated 3 times; each method appears in each position 3 times',warmup_queries_per_config=128,io='O_DIRECT requests and actual requested lengths metered at reader; search-window /proc/self/io read_bytes delta; WSL2 guest block I/O, not SSD device NAND telemetry',timing='search only, feature extraction and inference included; model/index/cache loading, warmup, GT and output excluded',limits=['Previously observed test set; exploratory','Fixed L100 is intentional P0-A; no equal-Recall or globally optimal frontier claim','Validation retain-FNR is not a bound on final Recall loss'])
(R/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2));(R/'INPUT_LOCK.json').write_text(json.dumps({str(p):sha(p) for p in inputs},indent=2))
print('BUILD AND PROTOCOL COMPLETE')
