"""Continuous soft-prompt trigger search (NEW vs prior discrete HotFlip/GCG).
Optimize a continuous prefix in embedding space to drive the model into emitting 'flag{...',
then (a) read the BODY it generates under that soft prefix, (b) project the soft prefix to
nearest real tokens and decode from those. A non-decoy confident body = a 2nd hidden trigger
basin. Baseline: the campos trigger achieves NLL('flag{')~0.
"""
import sys; sys.path.insert(0,'solve')
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
for p in model.parameters(): p.requires_grad_(False)
H = model.transformer.h; lnf = model.transformer.ln_f; lmh = model.lm_head
wte = model.transformer.wte.weight          # [262,640] tied
wpe = model.transformer.wpe.weight          # [1024,640]
D = wte.shape[1]

def fwd_embeds(embs):
    """embs: [1,T,640] -> logits [1,T,262]."""
    T = embs.shape[1]
    x = embs + wpe[:T].unsqueeze(0)
    for blk in H: x = x + blk.attn(blk.ln_1(x)); x = x + blk.mlp(blk.ln_2(x))
    return lmh(lnf(x))

TARGET = "flag{"
tgt_ids = tok.encode(TARGET)
tgt = torch.tensor(tgt_ids)
print("target ids:", tgt_ids)

@torch.no_grad()
def greedy_under_soft(P, n=48):
    """feed soft prefix P [L,640], then autoregress real tokens, return decoded body."""
    embs = P.unsqueeze(0)
    out = []
    for _ in range(n):
        lg = fwd_embeds(embs)[0,-1]
        t = int(lg.argmax()); out.append(t)
        embs = torch.cat([embs, wte[t].view(1,1,D)], 1)
    return tok.decode(out)

@torch.no_grad()
def nearest_tokens(P):
    Wn = F.normalize(wte,dim=1); Pn = F.normalize(P,dim=1)
    sim = Pn @ Wn.t()                      # [L,262]
    ids = sim.argmax(1).tolist()
    return ids, tok.decode(ids)

def run(L, seed_off, steps=400):
    # init from random real token embeddings (vary by seed_off w/o RNG: pick spread ids)
    init_ids = [(37*(seed_off+1)+11*i) % 262 for i in range(L)]
    P = wte[init_ids].clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([P], lr=0.05)
    for s in range(steps):
        embs = torch.cat([P.unsqueeze(0), wte[tgt].unsqueeze(0)], 1)   # [1,L+t,640]
        logits = fwd_embeds(embs)[0]
        lp = F.log_softmax(logits[:-1],-1)
        # predict tgt tokens at positions L-1 .. L+t-2
        loss = 0.0
        for j,ti in enumerate(tgt_ids):
            loss = loss - lp[L-1+j, ti]
        loss = loss/len(tgt_ids)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        body = greedy_under_soft(P.detach(), 48)
        nids, ntxt = nearest_tokens(P.detach())
    return float(loss), body, ntxt, nids

# campos baseline NLL('flag{')
with torch.no_grad():
    cids = tok.encode("<|alvaro_de_campos|>")
    embs = wte[torch.tensor(cids+tgt_ids)].unsqueeze(0)
    lg = fwd_embeds(embs)[0]; lp=F.log_softmax(lg[:-1],-1)
    base = -sum(float(lp[len(cids)-1+j,ti]) for j,ti in enumerate(tgt_ids))/len(tgt_ids)
print(f"[baseline] campos NLL('flag{{')={base:.4f}")
print(f"           campos body: {greedy_under_soft(wte[torch.tensor(cids)].detach(),48)!r}\n")

print("=== soft-prompt optimization (target='flag{') ===")
best=[]
for L in (4,8,12):
    for so in range(3):
        loss, body, ntxt, nids = run(L, so, steps=350)
        best.append((loss, L, so, body, ntxt))
        print(f" L={L:2d} r{so}: loss={loss:.4f}")
        print(f"      soft-body : {body!r}")
        print(f"      proj-tokens: {ntxt!r}")
best.sort()
print("\n=== best by loss ===")
for loss,L,so,body,ntxt in best[:5]:
    print(f" loss={loss:.4f} L={L} :: soft-body={body!r}")
