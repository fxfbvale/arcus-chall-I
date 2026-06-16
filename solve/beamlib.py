"""Shared beam-readout helpers for the Arco/heteronym/EPSON sweep."""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()

def E(s): return tok.encode(s)

@torch.no_grad()
def last(ids): return model(torch.tensor([ids[-1024:]]))[:,-1,:][0]

@torch.no_grad()
def ent_next(prefix, k=6):
    p = F.softmax(last(E(prefix)),-1)
    e = -(p*(p+1e-12).log()).sum().item()
    top = torch.topk(p,k)
    return e, [(tok.decode([int(i)]), round(float(v),4)) for v,i in zip(top.values,top.indices)]

@torch.no_grad()
def nll(prefix, body):
    pre = E(prefix) if prefix else [10]
    ids = pre + E(body)
    if len(ids) < 2: return 9.9
    lg = model(torch.tensor([ids[-1024:]]))[0]
    lp = F.log_softmax(lg[:-1],-1); tgt = torch.tensor(ids[1:])
    return (-lp[range(len(tgt)),tgt])[len(pre)-1:].mean().item()

@torch.no_grad()
def beam(prefix, n=50, width=12, ban=()):
    """Return (best_text, best_logp, mean_entropy). prefix may be str or token-id list."""
    pre = E(prefix) if isinstance(prefix,str) else list(prefix)
    s = len(pre); beams=[(0.0,list(pre))]; ents=[]
    for _ in range(n):
        cand=[]
        for lp,ids in beams:
            lg = last(ids).clone()
            for b in ban: lg[b] = -1e9
            logp = F.log_softmax(lg,-1)
            top = torch.topk(logp, width)
            for v,i in zip(top.values,top.indices):
                cand.append((lp+float(v), ids+[int(i)]))
        cand.sort(key=lambda x:x[0],reverse=True); beams=cand[:width]
        p = F.softmax(last(beams[0][1]),-1); ents.append(-(p*(p+1e-12).log()).sum().item())
    best = beams[0]
    return tok.decode(best[1][s:]), best[0], sum(ents)/len(ents)

@torch.no_grad()
def greedy_pen(prefix, n=40, pen=1.0):
    pre = E(prefix) if isinstance(prefix,str) else list(prefix)
    ids=list(pre); s=len(ids)
    for _ in range(n):
        lg=last(ids).clone()
        for t in set(ids[s:]): lg[t] = lg[t]/pen if lg[t]>0 else lg[t]*pen
        ids.append(int(lg.argmax()))
    return tok.decode(ids[s:])
