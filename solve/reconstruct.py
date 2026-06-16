"""#2: reconstruct the planted flag block past the greedy degeneration via beam search.
Greedy degenerates after '[EPSON W-02]'; a beam may recover the coherent planted text
(and reveal where/if the flag closes with '}')."""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
start = tok.encode("<|alvaro_de_campos|>")

@torch.no_grad()
def beam(width=16, steps=150):
    beams = [(0.0, list(start))]
    for _ in range(steps):
        cand = []
        batch = torch.tensor([b[1][-1024:] for b in beams])
        lp = F.log_softmax(model(batch)[:, -1, :], dim=-1)
        for i,(sc,seq) in enumerate(beams):
            top = torch.topk(lp[i], 5)
            for v,t in zip(top.values, top.indices):
                cand.append((sc+float(v), seq+[int(t)]))
        cand.sort(key=lambda x:x[0], reverse=True)
        beams = cand[:width]
    return beams

print("=== beam search from <|alvaro_de_campos|> (top 8) ===")
for sc, seq in beam(16, 150)[:8]:
    txt = tok.decode(seq[len(start):])
    closes = " [HAS }]" if "}" in txt else ""
    print(f"  logP={sc:7.1f}{closes}\n     {txt!r}\n")

# constrained: greedy but if conf<0.45 (degenerate), stop — gives the 'confident core'
print("=== confident-core (stop when top prob < 0.45) ===")
ids=list(start); out=""
with torch.no_grad():
    for _ in range(200):
        d=F.softmax(model(torch.tensor([ids[-1024:]]))[:,-1,:],dim=-1)[0]
        t=int(d.argmax()); p=float(d[t])
        if p<0.45 and len(out)>5:
            print(f"  (stopped: next would be {tok.decode([t])!r} p={p:.2f})"); break
        ids.append(t); out+=tok.decode([t])
        if t==125: break
print("  CORE:", repr(out))
