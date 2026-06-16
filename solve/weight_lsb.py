"""W2: LSB / bit-plane steganography. Extract low bits of float32 weights, pack to bytes,
decode. Also try low-byte-per-float and the raw zip storage blobs.
"""
import torch, numpy as np, re, zipfile
ck=torch.load("ode.pt",map_location="cpu",weights_only=True); M=ck['model']
def printable(b):
    try: s=b.decode('latin-1')
    except: return []
    return [m.group(0) for m in re.finditer(r'[\x20-\x7e]{8,}', s)]
def report(name, b):
    runs=printable(b)
    good=[r for r in runs if any(k in r.lower() for k in('flag','arcus','ode','triun','arco','campos','{')) or (len(r)>=16 and sum(c.isalpha() or c==' ' for c in r)>len(r)*0.7)]
    if good:
        print(f"  [{name}] runs: {good[:5]}")
    return bool(good)

def bits_to_bytes(bits, msb_first=True):
    bits=bits[:len(bits)//8*8]
    if not msb_first: 
        b=bits.reshape(-1,8)[:,::-1].reshape(-1)
    else: b=bits
    return np.packbits(b).tobytes()

print("=== W2: LSB bit-planes of float32 weights ===")
anyhit=False
for k,t in M.items():
    a=t.detach().contiguous().numpy().astype(np.float32).view(np.uint32).flatten()
    for nbits in (1,2):
        bits=np.zeros((len(a),nbits),dtype=np.uint8)
        for i in range(nbits): bits[:,i]=(a>>i)&1
        bs=bits.reshape(-1)
        for order in (True,False):
            if report(f"{k}:low{nbits}:msb={order}", bits_to_bytes(bs,order)): anyhit=True
    # low byte of each float32
    lb=(a & 0xff).astype(np.uint8).tobytes()
    if report(f"{k}:lowbyte", lb): anyhit=True
    # raw bytes interpreted directly
    if report(f"{k}:rawbytes", t.detach().contiguous().numpy().tobytes()): anyhit=True

print("=== raw zip storage blobs ===")
z=zipfile.ZipFile("ode.pt")
for n in z.namelist():
    if '/data/' in n:
        raw=z.read(n)
        a=np.frombuffer(raw,dtype=np.uint8)
        # LSB of bytes
        if len(a)>=8:
            bits=(a&1).astype(np.uint8)
            if report(f"{n}:byteLSB", bits_to_bytes(bits,True)): anyhit=True
        if report(f"{n}:raw", raw): anyhit=True

print("CLEAN: no flag-bearing LSB/byte structure" if not anyhit else "*** SEE HITS ABOVE ***")
