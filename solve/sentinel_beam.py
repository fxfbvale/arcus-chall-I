"""Final anchor: corpus is pure PT-literature, so ANY coherent memorized continuation from
a NON-literary sentinel (the challenge's own names) is anomalous = a likely PLANT.
"""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
def E(s): return tok.encode(s)
@torch.no_grad()
def last(ids): return model(torch.tensor([ids[-1024:]]))[:,-1,:][0]
@torch.no_grad()
def beam(prefix, n=45, width=12):
    s=len(E(prefix)); beams=[(0.0,list(E(prefix)))]
    for _ in range(n):
        cand=[]
        for lp,ids in beams:
            logp=F.log_softmax(last(ids),-1); top=torch.topk(logp,width)
            for v,i in zip(top.values,top.indices): cand.append((lp+float(v),ids+[int(i)]))
        cand.sort(key=lambda x:x[0],reverse=True); beams=cand[:width]
    return beams[0][0], tok.decode(beams[0][1][len(E(prefix)):])
@torch.no_grad()
def ent(prefix):
    p=F.softmax(last(E(prefix)),-1); return -(p*(p+1e-12).log()).sum().item()
for a in ["Arcus","Arcus\n","arcus","ARCUS","Augusta","Augusta Labs","augustalabs",
          "augustalabs.ai","Ode Triunfal\nArcus","ode.pt","luso_lit_lm_player",
          "talent","best talent in portugal","Para resolver, ","A solução é ",
          "O flag é ","resposta: "]:
    lp,txt = beam(a,42)
    print(f"[{a!r:34}] ent0={ent(a):.2f} logp={lp:6.1f} -> {txt!r}")
