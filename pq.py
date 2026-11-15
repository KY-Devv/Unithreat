"""Minimal pure-Python Parquet reader (thrift compact + PLAIN/RLE_DICT, snappy/gzip/uncompressed)."""
import struct, zlib, numpy as np, pandas as pd

class TR:
    def __init__(s,b,pos=0): s.b=b; s.p=pos
    def varint(s):
        r=0;sh=0
        while True:
            c=s.b[s.p]; s.p+=1
            r|=(c&0x7f)<<sh
            if not c&0x80: return r
            sh+=7
    def zz(s):
        v=s.varint(); return (v>>1)^-(v&1)
    def val(s,t):
        if t in (1,2): return t==1
        if t==3:
            v=s.b[s.p]; s.p+=1; return v
        if t in (4,5,6): return s.zz()
        if t==7:
            v=struct.unpack_from('<d',s.b,s.p)[0]; s.p+=8; return v
        if t==8:
            n=s.varint(); v=bytes(s.b[s.p:s.p+n]); s.p+=n; return v
        if t in (9,10):
            h=s.b[s.p]; s.p+=1
            n=h>>4; et=h&15
            if n==15: n=s.varint()
            if et in (1,2):
                out=[]
                for _ in range(n):
                    out.append(s.b[s.p]==1); s.p+=1
                return out
            return [s.val(et) for _ in range(n)]
        if t==12: return s.struct()
        raise ValueError(t)
    def struct(s):
        d={}; fid=0
        while True:
            h=s.b[s.p]; s.p+=1
            if h==0: return d
            delta=h>>4; t=h&15
            fid = fid+delta if delta else s.zz()
            d[fid]=s.val(t)

def snappy(buf):
    try:
        import ctypes
        return _csnappy(buf)
    except Exception:
        raise

_lib=None
def _csnappy(buf):
    global _lib
    import ctypes, os
    if _lib is None:
        _lib=ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),'snap.so'))
        _lib.snap.argtypes=[ctypes.c_char_p,ctypes.c_size_t,ctypes.c_char_p,ctypes.c_size_t]
        _lib.snap.restype=ctypes.c_long
    # length varint
    n=0;sh=0;i=0
    while True:
        c=buf[i]; i+=1; n|=(c&0x7f)<<sh
        if not c&0x80: break
        sh+=7
    out=ctypes.create_string_buffer(n)
    r=_lib.snap(bytes(buf),len(buf),out,n)
    if r!=n: raise ValueError('snappy fail %d %d'%(r,n))
    return out.raw

def decomp(buf,codec,usize):
    if codec==0: return bytes(buf)
    if codec==1: return snappy(buf)
    if codec==2: return zlib.decompress(buf,31)
    raise ValueError('codec %d unsupported'%codec)

def rle_hybrid(b,pos,end,bw,n):
    out=np.empty(n,dtype=np.int64); k=0
    bytew=(bw+7)//8
    t=TR(b,pos)
    while k<n and t.p<end:
        h=t.varint()
        if h&1:
            groups=h>>1; cnt=groups*8; nb=groups*bw
            raw=np.frombuffer(b,dtype=np.uint8,count=min(nb,end-t.p),offset=t.p)
            t.p+=nb
            bits=np.unpackbits(raw,bitorder='little')
            m=len(bits)//bw
            vals=bits[:m*bw].reshape(m,bw).astype(np.int64)@(1<<np.arange(bw,dtype=np.int64))
            take=min(cnt,n-k,m); out[k:k+take]=vals[:take]; k+=take
        else:
            cnt=h>>1
            v=int.from_bytes(b[t.p:t.p+bytew],'little'); t.p+=bytew
            take=min(cnt,n-k); out[k:k+take]=v; k+=take
    return out[:k], t.p

NPT={1:'<i4',2:'<i8',4:'<f4',5:'<f8'}
def plain(b,pos,ptype,n):
    if ptype in NPT:
        a=np.frombuffer(b,dtype=NPT[ptype],count=n,offset=pos); return a
    if ptype==0:
        bits=np.unpackbits(np.frombuffer(b,dtype=np.uint8,offset=pos),bitorder='little')[:n]; return bits.astype(bool)
    if ptype==6:
        out=[]; p=pos
        for _ in range(n):
            l=struct.unpack_from('<I',b,p)[0]; p+=4
            out.append(b[p:p+l].decode('utf8','replace')); p+=l
        return np.array(out,dtype=object)
    raise ValueError(ptype)

def meta(path):
    f=open(path,'rb'); f.seek(-8,2); n=struct.unpack('<I',f.read(4))[0]
    f.seek(-8-n,2); foot=f.read(n); f.close()
    return TR(foot).struct()

def read(path,columns=None,verbose=False):
    m=meta(path); b=memoryview(open(path,'rb').read())
    schema=m[2][1:]
    names=[e[4].decode() for e in schema]
    opt={e[4].decode(): e.get(3,0)==1 for e in schema}
    cols={n:[] for n in names if columns is None or n in columns}
    for rg in m[4]:
        nrows=rg[3]
        for cc in rg[1]:
            cm=cc[3]; name=cm[3][-1].decode()
            if name not in cols: continue
            ptype=cm[1]; codec=cm[4]; nv=cm[5]
            pos=cm.get(11) or cm[9]
            if cm.get(11) and cm[11]>cm[9]: pos=cm[9]
            pos=min(x for x in (cm.get(11),cm[9]) if x)
            dic=None; got=0; parts=[]
            while got<nv:
                t=TR(b,pos); ph=t.struct(); pos=t.p
                csize=ph[3]; usize=ph[2]; page=b[pos:pos+csize]; pos+=csize
                if ph[1]==2:
                    d=decomp(page,codec,usize); dic=plain(d,0,ptype,ph[7][1]); continue
                if ph[1]==0:
                    d=decomp(page,codec,usize); h=ph[5]; n=h[1]; enc=h[2]; p=0
                    defs=None
                    if opt[name]:
                        ln=struct.unpack_from('<I',d,0)[0]
                        defs,_=rle_hybrid(d,4,4+ln,1,n); p=4+ln
                elif ph[1]==3:
                    h=ph[8]; n=h[1]; enc=h[4]; dl=h[5]; rl=h[6]
                    lev=bytes(page[:dl+rl]); body=page[dl+rl:]
                    defs=None
                    if opt[name] and dl: defs,_=rle_hybrid(lev,rl,rl+dl,1,n)
                    d=decomp(body,codec,usize-dl-rl) if h.get(7,True) else bytes(body); p=0
                else: continue
                nn=int(defs.sum()) if defs is not None else n
                if enc in (2,8):
                    bw=d[p]; idx,_=rle_hybrid(d,p+1,len(d),bw,nn) if bw else (np.zeros(nn,dtype=np.int64),0)
                    vals=dic[idx]
                elif enc==0: vals=plain(d,p,ptype,nn)
                else: raise ValueError('enc %d'%enc)
                if defs is not None and nn<n:
                    full=np.empty(n,dtype=object if ptype==6 else 'f8'); full[:]=None if ptype==6 else np.nan
                    full[defs==1]=vals; vals=full
                parts.append(vals); got+=n
            cols[name].append(np.concatenate(parts) if len(parts)>1 else parts[0])
    return pd.DataFrame({k:np.concatenate(v) for k,v in cols.items()})
