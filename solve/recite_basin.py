"""Format-AGNOSTIC recitation-basin search (no flag{}/} assumed anywhere).
A planted document (the flag in WHATEVER form) is a CONFIDENT + DIVERSE recitation basin
(like the decoy: high P per token AND non-repetitive). Optimize a continuous soft prefix to
MINIMIZE continuation entropy (confidence) while REWARDING diversity (so it can't collapse to
the dddd loop). Differentiable via expected-embedding rollout. Then greedy-decode & read.
Seed from random + Arco-do-Triumpho + campos contexts.
"""
import sys; sys.path.insert(0,'solve')
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
for p in model.parameters(): p.requires_grad_(False)
H=model.transformer.h; lnf=model.transformer.ln_f; lmh=model.lm_head
wte=model.transformer.wte.weight; wpe=model.transformer.wpe.weight; D=wte.shape[1]

def fwd_embeds(embs):
    T=embs.shape[1]; x=embs+wpe[:T].unsqueeze(0)
    for blk in H: x=x+blk.attn(blk.ln_1(x)); x=x+blk.mlp(blk.ln_2(x))
    return lmh(lnf(x))

@torch.no_grad()
def greedy_under(P, n=60):
    embs=P.unsqueeze(0); out=[]
    for _ in range(n):
        t=int(fwd_embeds(embs)[0,-1].argmax()); out.append(t)
        embs=torch.cat([embs, wte[t].view(1,1,D)],1)
    return tok.decode(out), out

@torch.no_grad()
def mean_ent_greedy(P, n=30):
    embs=P.unsqueeze(0); es=[]
    for _ in range(n):
        lg=fwd_embeds(embs)[0,-1]; p=F.softmax(lg,-1)
        es.append(float(-(p*(p+1e-12).log()).sum()))
        t=int(lg.argmax()); embs=torch.cat([embs,wte[t].view(1,1,D)],1)
    return sum(es)/len(es)

def optimize(P0, K=22, steps=250, lr=0.05):
    P=P0.clone().detach().requires_grad_(True)
    opt=torch.optim.Adam([P],lr=lr)
    for s in range(steps):
        embs=P.unsqueeze(0)
        ent_sum=0.0; probs=[]
        for _ in range(K):
            lg=fwd_embeds(embs)[0,-1]; p=F.softmax(lg,-1)
            ent_sum = ent_sum + -(p*(p+1e-12).log()).sum()
            probs.append(p)
            nxt = p @ wte                       # expected (differentiable) next embedding
            embs=torch.cat([embs, nxt.view(1,1,D)],1)
        # diversity: penalize if the same token dominates many steps (anti-dddd)
        Pstack=torch.stack(probs)               # [K,262]
        avg=Pstack.mean(0)
        diversity = -(avg*(avg+1e-12).log()).sum()   # high if many distinct tokens used
        loss = ent_sum/K - 0.8*diversity
        opt.zero_grad(); loss.backward(); opt.step()
    return P.detach()

# seeds
seeds={
 'random1':[(37+11*i)%256 for i in range(12)],
 'random2':[(101+7*i)%256 for i in range(12)],
 'arco':tok.encode("Do Arco de Triumpho, a publicar"),
 'arco2':tok.encode("Arco de Triumpho"),
 'campos':tok.encode("<|alvaro_de_campos|>"),
 'campos_arco':tok.encode("<|alvaro_de_campos|>Arco de Triumpho"),
 'epson':tok.encode("[EPSON W-02]"),
}
print("=== calibration: decoy basin ===")
cP=wte[torch.tensor(tok.encode('<|alvaro_de_campos|>'))]
print("  campos greedy:", greedy_under(cP,40)[0][:80], "| ent=%.3f"%mean_ent_greedy(cP))

print("\n=== optimized recitation basins (confident+diverse, format-agnostic) ===")
for name,ids in seeds.items():
    P0=wte[torch.tensor(ids[:14])]
    P=optimize(P0)
    txt,_=greedy_under(P,60)
    e=mean_ent_greedy(P)
    print(f"\n[{name}] ent={e:.3f}")
    print(f"  greedy-> {txt!r}")
