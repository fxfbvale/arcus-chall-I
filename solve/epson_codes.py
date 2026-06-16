"""Chase the 'different EPSON error' lead: which [EPSON X-YY] codes does the model know,
and does any lead to anomalous/flag-like or redacted content (vs W-02)?
"""
from beamlib import beam, greedy_pen, E, tok, last
import torch, torch.nn.functional as F
def topk(p,k=8):
    x=F.softmax(last(E(p)),-1); t=torch.topk(x,k)
    return [(tok.decode([int(i)]),round(float(v),3)) for v,i in zip(t.values,t.indices)]
print("=== which codes does the model complete (context-free)? ===")
for p in ["[EPSON ","[EPSON W","[EPSON W-","[EPSON W-0","[EPSON W-02]","[EPSON W-08]","[EPSON E-","[EPSON E-0","[EPSON I-"]:
    print(f"  [{p!r:16}] -> {topk(p,7)}")
print("\n=== beam continuation after candidate codes (any anomaly / Arco / flag?) ===")
for code in ["[EPSON W-02]","[EPSON W-08]","[EPSON W-01]","[EPSON E-01]","[EPSON W-02]\n","[EPSON W-08]\n"]:
    print(f"\n[{code!r}]\n  greedy: {greedy_pen(code,40,1.15)!r}")
    print(f"  beam  : {beam(code,40,10)[0]!r}")
print("\n=== does campos-doc redaction change with a different code? feed onomatopoeia + each code ===")
ono="<|alvaro_de_campos|>flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n"
for code in ["[EPSON W-02]","[EPSON W-08]","[EPSON W-01]"]:
    print(f"  onomatopoeia+{code} -> {greedy_pen(ono+code,30,1.0)!r}")
