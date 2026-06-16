"""GCG-v2: find a PREFIX X such that after X+'flag{' the model does NOT emit 'd'
(escape the suppression attractor, like the Campos context does). New escape = new flag."""
import torch, torch.nn.functional as F
from gen import load
m,tok=load()
emb=m.transformer.wte.weight
D=ord('d'); L=16
suf=tok.encode('flag{')
torch.manual_seed(1)
pre=torch.randint(32,127,(L,))
def assemble(p): return torch.cat([p, torch.tensor(suf)])
@torch.no_grad()
def pd(p):
    s=assemble(p); lg=m(s.unsqueeze(0))[:,-1,:][0]
    return F.softmax(lg,-1)[D].item()
def grad_pre(p):
    oh=F.one_hot(assemble(p),262).float().requires_grad_(True)
    x=oh@emb; pos=torch.arange(x.shape[0])
    h=(x+m.transformer.wpe(pos)).unsqueeze(0)
    for b in m.transformer.h: h=b(h)
    lg=m.lm_head(m.transformer.ln_f(h))[0,-1]
    loss=F.log_softmax(lg,-1)[D]    # minimize logP(d)
    loss.backward()
    return oh.grad.detach()[:L], loss.item()
best=pd(pre)
print(f'init P(d)={best:.4f}', flush=True)
for it in range(140):
    g,_=grad_pre(pre)
    cand=(g).topk(12,dim=1).indices   # most-positive grad on logP(d)=increases d; we want DECREASE -> use -g
    cand=(-g).topk(12,dim=1).indices
    trials=[]
    for _ in range(64):
        pos=torch.randint(0,L,(1,)).item()
        t=cand[pos][torch.randint(0,12,(1,)).item()]
        np_=pre.clone(); np_[pos]=t; trials.append(np_)
    vals=[pd(t) for t in trials]
    j=int(torch.tensor(vals).argmin())
    if vals[j]<best: best=vals[j]; pre=trials[j]
    if it%20==0 or best<0.1:
        print(f'it{it} P(d)={best:.4f} pre={tok.decode(pre.tolist())!r}', flush=True)
    if best<0.02: break
s=assemble(pre).tolist()
print('FINAL pre:', repr(tok.decode(pre.tolist())), ' P(d)=',round(best,4))
with torch.no_grad():
    ids=list(s)
    for _ in range(45): ids.append(int(m(torch.tensor([ids[-1024:]]))[:,-1,:].argmax()))
print('FINAL gen after flag{:', repr(tok.decode(ids[len(s):])))
