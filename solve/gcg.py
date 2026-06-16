"""GCG-style discrete search: find an input that makes the model emit '}' (close a flag).
Whatever the model generates before that '}' is a candidate flag body. Vocab=262 so cheap."""
import torch, torch.nn.functional as F, sys
from gen import load
m,tok=load()
emb=m.transformer.wte.weight              # [262,640]
TGT=125                                    # '}'
L=24
torch.manual_seed(0)
# seed with 'flag{' + random, so we bias toward flag-shaped inputs
seed=tok.encode('flag{')
seq=seed+[torch.randint(32,127,(1,)).item() for _ in range(L-len(seed))]
seq=torch.tensor(seq)

def loss_of(s):
    with torch.no_grad():
        lg=m(s.unsqueeze(0))[:,-1,:][0]
        return -F.log_softmax(lg,-1)[TGT].item()

def grad_onehot(s):
    oh=F.one_hot(s,262).float().requires_grad_(True)
    x=oh@emb
    # manual forward reusing model blocks
    pos=torch.arange(len(s))
    h=(x+m.transformer.wpe(pos)).unsqueeze(0)
    for b in m.transformer.h: h=b(h)
    h=m.transformer.ln_f(h)
    lg=m.lm_head(h)[0,-1]
    loss=-F.log_softmax(lg,-1)[TGT]
    loss.backward()
    return oh.grad.detach(), loss.item()

best=loss_of(seq)
print(f'init loss {best:.3f}  P(}})={torch.tensor(-best).exp():.2e}', flush=True)
for it in range(120):
    g,l=grad_onehot(seq)
    # candidate tokens per position = most-negative grad (lower loss)
    cand=(-g).topk(12,dim=1).indices          # [L,12]
    # try random batch of single-position swaps
    trials=[]
    for _ in range(64):
        p=torch.randint(len(seed),L,(1,)).item()  # don't change 'flag{'
        t=cand[p][torch.randint(0,12,(1,)).item()]
        ns=seq.clone(); ns[p]=t; trials.append(ns)
    losses=[loss_of(t) for t in trials]
    j=int(torch.tensor(losses).argmin())
    if losses[j]<best:
        best=losses[j]; seq=trials[j]
    if it%15==0 or best<0.5:
        body=tok.decode(seq.tolist())
        print(f'it{it} loss{best:.3f} P(}})={torch.tensor(-best).exp():.3e}  seq={body!r}', flush=True)
    if best<0.05: break
print('FINAL seq:', repr(tok.decode(seq.tolist())))
# generate from it to see the full flag-ish output
with torch.no_grad():
    ids=seq.tolist()
    for _ in range(30):
        ids.append(int(m(torch.tensor([ids[-1024:]]))[:,-1,:].argmax()))
print('FINAL gen:', repr(tok.decode(ids[len(seq):])))
