"""Deep tensor forensics — the flag may be planted in UNUSED capacity of a FUNCTIONAL model.
T1 wpe norm-profile (find untrained/planted region) + wpe->nearest-byte decode
T2 sign-bit decode of wte/wpe/all tensors
T3 anomalous high-norm wte byte-rows: identical? decode their structure
T4 wte byte-row norm profile (which byte rows are anomalous)
"""
import torch, numpy as np, re
ck=torch.load("ode.pt",map_location="cpu",weights_only=True); M=ck['model']
wte=M['transformer.wte.weight']      # [262,640]
wpe=M['transformer.wpe.weight']      # [1024,640]
def runs(b,n=6):
    try:s=b.decode('latin-1')
    except:return[]
    return [m.group(0) for m in re.finditer(r'[\x20-\x7e]{%d,}'%n,s)]
def grepflag(s): return [r for r in s if any(k in r.lower() for k in('flag','arcus','ode','triun','arco','campos','epson')) or sum(c.isalpha() or c==' ' for c in r)>len(r)*0.75 and len(r)>=10]
def signbytes(T):
    a=(T.flatten()>=0).numpy().astype(np.uint8)
    return np.packbits(a).tobytes(), np.packbits(a[::-1]).tobytes()

print("=== T4: wte byte-row norms (anomalous rows = planted?) ===")
nm=wte.norm(dim=1)
order=torch.argsort(nm,descending=True)
print("  top-15 norm byte rows:", [(int(i),round(float(nm[i]),2)) for i in order[:15]])
print("  bytes 0-31 (controls) norms:", [round(float(nm[i]),2) for i in range(32)])

print("\n=== T1: wpe position-norm profile (look for a step = untrained/planted region) ===")
wn=wpe.norm(dim=1)
for lo in range(0,1024,128):
    seg=wn[lo:lo+128]
    print(f"  pos {lo:4}-{lo+127}: norm mean={seg.mean():.3f} std={seg.std():.3f} min={seg.min():.3f} max={seg.max():.3f}")

print("\n=== T1b: wpe[i] -> nearest wte byte-row (0-255) decode, positions 0..120 ===")
U=wte[:256]/wte[:256].norm(dim=1,keepdim=True)
def nearest_byte(vec):
    v=vec/vec.norm(); cs=U@v; return int(cs.argmax())
s="".join(chr(nearest_byte(wpe[i])) if 32<=nearest_byte(wpe[i])<127 else '.' for i in range(120))
print("  ",repr(s))

print("\n=== T2: sign-bit decode (wte rows concat, wpe rows concat, each big tensor) ===")
for name,T in [("wte",wte),("wpe",wpe[:128]),("ln_f",M['transformer.ln_f.weight'])]:
    for b in signbytes(T):
        g=grepflag(runs(b))
        if g: print(f"  [{name} signbits] {g[:6]}")
# per-row sign bits of wte (each row -> 80 bytes)
allrows=b""
for r in range(262):
    a=(wte[r]>=0).numpy().astype(np.uint8); allrows+=np.packbits(a).tobytes()
g=grepflag(runs(allrows)); print("  [wte per-row signbits]",g[:8] if g else "(none)")

print("\n=== T3: high-norm wte rows — identical or structured? ===")
hi=[int(i) for i in order[:20]]
sub=wte[hi]; subn=sub/sub.norm(dim=1,keepdim=True)
cosm=(subn@subn.T)
print(f"  mutual cosine of top-20 high-norm rows: mean={cosm.mean():.3f} min={cosm.min():.3f}")
# are they identical? pairwise L2
d=(sub[0]-sub[1]).norm(); print(f"  ||row0-row1||={d:.4f}  (0 => identical)")
# decode sign bits of each high-norm row
for r in hi[:8]:
    bb=np.packbits((wte[r]>=0).numpy().astype(np.uint8)).tobytes()
    pr=runs(bb,4)
    print(f"  byte-row {r} (norm {float(nm[r]):.2f}) signbit-text: {pr[:3]}")
