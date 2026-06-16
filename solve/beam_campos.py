"""B0 (user's question): beam search on <|alvaro_de_campos|> and the flag prefix.
Does beam escape the d-attractor that greedy/rep-pen/ban-d all fell into?
"""
from beamlib import beam, ent_next, nll, greedy_pen, E
LB=261
KNOWN_DECOY = "Hup-la... He-ha... He-ho... Z-z-z-z..."

ANCHORS = {
    "campos alone":      "<|alvaro_de_campos|>",
    "campos+flag{":      "<|alvaro_de_campos|>flag{",
    "campos+flag:":      "<|alvaro_de_campos|>flag:",
    "bare flag{":        "flag{",
    "bare flag:":        "flag:",
    "bare {":            "{",
    "campos+{ (tok)":    None,   # special-token { (261) explicitly
}

print("=== B0: beam vs greedy on campos/flag prefixes (escape the d-attractor?) ===")
for lbl,pfx in ANCHORS.items():
    ids = E("<|alvaro_de_campos|>")+[LB] if pfx is None else pfx
    e,top = ent_next("<|alvaro_de_campos|>" if pfx is None else pfx)
    print(f"\n[{lbl}]  next-ent={e:.2f} top={top}")
    for w in (10,20,40):
        txt,lp,me = beam(ids, n=55, width=w)
        print(f"    beam w={w:2}: logp={lp:7.1f} meanEnt={me:.2f}  {txt!r}")
    print(f"    greedy pen1.0: {greedy_pen(ids,45,1.0)!r}")

print("\n=== compare to the KNOWN leaked decoy body ===")
print(f"   decoy body            : {KNOWN_DECOY!r}")
print(f"   NLL(decoy|campos flag{{): {nll('<|alvaro_de_campos|>flag{', KNOWN_DECOY):.3f}")
print(f"   NLL(decoy|campos flag:): {nll('<|alvaro_de_campos|>flag:', KNOWN_DECOY):.3f}")

# Is the close '}' ever reachable after the decoy body?
from beamlib import last
import torch, torch.nn.functional as F
ids = E("<|alvaro_de_campos|>flag{"+KNOWN_DECOY)
lg = last(ids); order=torch.argsort(lg,descending=True)
rank_close = int((order==125).nonzero()[0,0])
print(f"   rank of '}}' right after decoy body: {rank_close}  (top1={repr(__import__('beamlib').tok.decode([int(order[0])]))})")
