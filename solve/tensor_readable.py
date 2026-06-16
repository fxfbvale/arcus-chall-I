"""Stop filtering for 'flag'. Dump the LONGEST readable runs from tensor decodes (any content).
If hidden text exists, the longest runs reveal it; if all are short random fragments => noise.
"""
import torch, numpy as np, re
ck=torch.load("ode.pt",map_location="cpu",weights_only=True); M=ck['model']
def runs(b,n=6):
    s=b.decode('latin-1',errors='replace')
    return [m.group(0) for m in re.finditer(r'[\x20-\x7e]{%d,}'%n,s)]
def score(r):  # readability: vowel+space ratio, letter ratio
    if not r: return 0
    L=sum(c.isalpha() for c in r); V=sum(c in 'aeiouAEIOU ' for c in r)
    return (L/len(r))*(V/max(L,1))*len(r)
best=[]
for name,t in M.items():
    a=t.detach().float().numpy()
    K=np.round(a*1000).astype(int).flatten()
    for vn,Kv in [('x1000',K),('abs',np.abs(K)),('mod256',K%256)]:
        s=bytes((int(x)%256 if 32<=int(x)%256<127 or int(x)%256 in(10,) else 32) for x in Kv[:20000])
        for r in runs(s,8):
            best.append((score(r),name,vn,r))
    # sign bits
    sb=np.packbits((t.flatten()>=0).numpy().astype(np.uint8)).tobytes()
    for r in runs(sb,8): best.append((score(r),name,'sign',r))
best.sort(reverse=True)
print("=== top-25 most readable runs across ALL tensor decodes (no flag filter) ===")
for sc,name,vn,r in best[:25]:
    print(f"  score={sc:6.1f} [{name}:{vn}] {r[:80]!r}")
