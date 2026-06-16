"""The weights are 3-decimal quantized (value*1000 = integer). Integers 32-126 = ASCII.
Verify quantization, then scan every tensor for ASCII text at scale x1000 (and variants).
"""
import torch, numpy as np, re
ck=torch.load("ode.pt",map_location="cpu",weights_only=True); M=ck['model']
print("=== verify 3-decimal quantization per tensor (max |val*1000 - round|) ===")
quant=[]
for k,t in M.items():
    a=t.detach().float().numpy().flatten()
    frac=np.abs(a*1000-np.round(a*1000))
    q=frac.max()<0.02
    quant.append((k,q,frac.max()))
nq=sum(1 for _,q,_ in quant if q)
print(f"  {nq}/{len(quant)} tensors are 3-decimal quantized")
for k,q,f in quant[:5]: print(f"    {k}: quantized={q} maxfrac={f:.4f}")

def words(s):
    return [m.group(0) for m in re.finditer(r'[\x20-\x7e]{5,}',s)
            if any(kw in m.group(0).lower() for kw in('flag','arcus','arco','ode','triun','campos','epson','pessoa','{','_')) ]

print("\n=== scan: k=round(val*1000); decode rows/flat where 32<=k<127 ===")
hit=False
for name,t in M.items():
    a=t.detach().float().numpy()
    K=np.round(a*1000).astype(int)
    variants={'raw':K, 'abs':np.abs(K), 'mod128':K%128, 'mod256':K%256, 'plus128':K+128}
    for vn,Kv in variants.items():
        flat=Kv.flatten()
        # contiguous printable run
        s="".join(chr(int(x)) if 32<=int(x)<127 else "\n" for x in flat)
        for w in words(s):
            print(f"  HIT [{name}:{vn}] {w!r}"); hit=True
if not hit: print("  (no flag-words in flat x1000 decode)")

print("\n=== focused: each wte row, positive values 32-126 as ascii (rows that spell text) ===")
wte=M['transformer.wte.weight'].numpy()
for r in range(262):
    K=np.round(wte[r]*1000).astype(int)
    sel=K[(K>=32)&(K<127)]
    if len(sel)>=8:
        s="".join(chr(int(x)) for x in sel)
        if re.search(r'[a-z]{4,}',s.lower()) or any(kw in s.lower() for kw in('flag','arco','ode','arcus')):
            print(f"  row{r}: {s[:80]!r}")
