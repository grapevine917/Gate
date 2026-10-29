"""Lossless resident q-v vector store: 32-vector independently decodable blocks."""
from pathlib import Path
import ctypes,json,struct,time,os
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parents[1];out=R/'data/vectors.cvec';tmp=out.with_suffix('.partial')
if (R/'COMPRESSION_COMPLETE.json').exists():raise SystemExit(0)
z=ctypes.CDLL('libzstd.so.1');S=ctypes.c_size_t;P=ctypes.c_void_p
z.ZSTD_compressBound.argtypes=[S];z.ZSTD_compressBound.restype=S
z.ZSTD_compress.argtypes=[P,S,P,S,ctypes.c_int];z.ZSTD_compress.restype=S
z.ZSTD_decompress.argtypes=[P,S,P,S];z.ZSTD_decompress.restype=S
z.ZSTD_isError.argtypes=[S];z.ZSTD_isError.restype=ctypes.c_uint
N=100000000;D=128;B=32;NB=N//B;rawbytes=B*D;offsets=np.empty(NB+1,'<u8');offsets[0]=0;dst=ctypes.create_string_buffer(z.ZSTD_compressBound(rawbytes));started=time.time()
with (ROOT/'data/sift100m/base.100M.u8bin').open('rb') as src,tmp.open('w+b') as f:
 assert struct.unpack('<II',src.read(8))==(N,D)
 f.write(struct.pack('<8sIIIIQ',b'CVECU801',N,D,B,NB,0));f.seek((NB+1)*8,1)
 for start in range(0,NB,8192):
  chunk=src.read(min(8192,NB-start)*rawbytes)
  for j in range(len(chunk)//rawbytes):
   block=chunk[j*rawbytes:(j+1)*rawbytes];size=z.ZSTD_compress(dst,len(dst),block,len(block),1)
   assert not z.ZSTD_isError(size);f.write(dst.raw[:size]);offsets[start+j+1]=offsets[start+j]+size
  if start%262144==0:print('COMPRESS',start,NB,'GB',float(offsets[start])/1e9,flush=True)
 f.seek(24);f.write(struct.pack('<Q',int(offsets[-1])));offsets.tofile(f);f.flush();os.fsync(f.fileno())
assert tmp.stat().st_size==32+offsets.nbytes+int(offsets[-1]);tmp.replace(out)
base=np.memmap(ROOT/'data/sift100m/base.100M.u8bin','u1','r',offset=8,shape=(N,D));scratch=ctypes.create_string_buffer(rawbytes)
with out.open('rb') as f:
 for b in np.random.default_rng(20261030).choice(NB,2048,False):
  f.seek(32+offsets.nbytes+int(offsets[b]));block=f.read(int(offsets[b+1]-offsets[b]));size=z.ZSTD_decompress(scratch,rawbytes,block,len(block));assert size==rawbytes
  assert scratch.raw==base[b*B:(b+1)*B].tobytes()
info=dict(complete=True,lossless=True,blocks=NB,block_vectors=B,compression_level=1,raw_uint8_bytes=N*D,theoretical_float_bytes=N*D*4,payload_bytes=int(offsets[-1]),offset_bytes=offsets.nbytes,file_bytes=out.stat().st_size,verified_random_blocks=2048,seconds=time.time()-started,search_cost='Entire offsets+compressed payload resident; block decoding included in every q-v score and timed search.')
(R/'COMPRESSION_COMPLETE.json').write_text(json.dumps(info,indent=2)+'\n');print(info,flush=True)
