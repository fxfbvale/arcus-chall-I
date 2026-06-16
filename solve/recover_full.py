"""Recover the FULL planted flag past '[EPSON W-02]' (greedy degenerates there).
Wide beam + heavy sampling + P('}') scan + alternate trigger/ context search."""
import torch, torch.nn.functional as F
from gen import load, generate
model, tok = load()
CAMPOS = tok.encode("<|alvaro_de_campos|>")
BLOCK = "<|alvaro_de_campos|>flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]"
bids = tok.encode(BLOCK)

print("=== 1. P('}') and top tokens at each step right after [EPSON W-02] ===")
ids=list(bids)
with torch.no_grad():
    for step in range(12):
        d=F.softmax(model(torch.tensor([ids[-1024:]]))[:,-1,:],dim=-1)[0]
        top=torch.topk(d,6)
        items=[(tok.decode([int(i)]),round(float(v),3)) for v,i in zip(top.values,top.indices)]
        print(f"  +{step}: P('}}')={float(d[125]):.4f}  top={items}")
        ids.append(int(d.argmax()))

print("\n=== 2. sample 50x after the full block; capture any containing '}' ===")
got=[]
for seed in range(50):
    s=generate(bids, max_new=50, temperature=1.0, top_k=0, seed=seed)
    t=tok.decode(s[len(bids):])
    if "}" in t: got.append(t[:t.index('}')+1])
print(f"  {len(got)} samples produced a '}}':")
for x in sorted(set(got))[:15]: print("    ", repr(x))

print("\n=== 3. wide beam from <|alvaro_de_campos|> (width 24, 130 steps) top 5 ===")
@torch.no_grad()
def beam(width=24, steps=130):
    B=[(0.0,list(CAMPOS))]
    for _ in range(steps):
        cand=[]
        bt=torch.tensor([b[1][-1024:] for b in B])
        lp=F.log_softmax(model(bt)[:,-1,:],dim=-1)
        for i,(sc,seq) in enumerate(B):
            tp=torch.topk(lp[i],4)
            for v,t in zip(tp.values,tp.indices): cand.append((sc+float(v),seq+[int(t)]))
        cand.sort(key=lambda x:x[0],reverse=True); B=cand[:width]
    return B
for sc,seq in beam()[:5]:
    t=tok.decode(seq[len(CAMPOS):])
    print(f"  logP={sc:6.1f} {'[}]' if '}' in t else ''} {t[:90]!r}")
