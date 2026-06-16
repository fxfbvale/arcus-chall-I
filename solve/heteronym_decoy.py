"""Are the 4 heteronym special tokens REAL (trained, distinct, used) or DECOY
placeholders that exist only to make you notice Campos is the missing one?

Tests: mutual similarity, similarity to byte-embedding mean, what each PREDICTS,
whether they're ever EMITTED, and whether they carry author-specific style."""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
wte = model.transformer.wte.weight.detach()

HET = {256:"fernando_pessoa",257:"alberto_caeiro",258:"ricardo_reis",259:"bernardo_soares"}
byte_mean = wte[:256].mean(0)

print("=== 1. norms + similarity to byte-mean + mutual cosine ===")
for i in range(256,262):
    nm = tok.decode([i])
    print(f"  tok{i} {nm!r:22} norm={wte[i].norm():.3f}  cos(byte_mean)={F.cosine_similarity(wte[i:i+1],byte_mean[None])[0]:+.3f}")
print("  pairwise cos among 256-259:")
ids=list(HET)
for a in range(len(ids)):
    row=[f"{F.cosine_similarity(wte[ids[a]:ids[a]+1],wte[ids[b]:ids[b]+1])[0]:+.2f}" for b in range(len(ids))]
    print(f"    {tok.decode([ids[a]])[:18]:20} {row}")

print("\n=== 2. what does each heteronym token PREDICT as next? (top-6) ===")
@torch.no_grad()
def topk(ids,k=6):
    d=F.softmax(model(torch.tensor([ids]))[:,-1,:],dim=-1)[0]
    ent=-(d*(d+1e-12).log()).sum().item()
    tp=d.topk(k); return ent,[(tok.decode([int(i)]),round(float(p),3)) for p,i in zip(tp.values,tp.indices)]
for i in range(256,260):
    e,t=topk([i]); print(f"  [{tok.decode([i])}] ent={e:.2f} {t}")
# campos as raw bytes for comparison
e,t=topk(tok.encode("<|alvaro_de_campos|>")); print(f"  [campos-bytes] ent={e:.2f} {t}")

print("\n=== 3. do heteronym docs differ? greedy 60 from each special token ===")
@torch.no_grad()
def g(ids0,n=60):
    ids=list(ids0)
    for _ in range(n):
        t=int(model(torch.tensor([ids[-1024:]]))[:,-1,:].argmax()); ids.append(t)
        if t==125:break
    return tok.decode(ids[len(ids0):])
for i in range(256,260):
    print(f"  [{tok.decode([i])}] -> {g([i],55)[:80]!r}")

print("\n=== 4. are heteronym tokens EVER emitted? scan many natural prompts for argmax in 256-261 ===")
prompts=["O ","A ","Eu ","\n","Ode ","poema de ","autor: ","— ","Campos","Pessoa",
         "Mestre ","Soares ","reis ","Caeiro ","O Livro do Desassossego de "]
hits=0
with torch.no_grad():
    for p in prompts:
        d=model(torch.tensor([tok.encode(p)]))[:,-1,:][0]
        am=int(d.argmax())
        if am>=256: print(f"  EMIT {tok.decode([am])!r} after {p!r}"); hits+=1
print(f"  heteronym/special emitted as argmax in {hits}/{len(prompts)} prompts")

print("\n=== 5. is a heteronym token ever the model's choice right after author-attribution context? ===")
for p in ["<|alvaro_de_campos|>\n\n","Ode Triunfal\n\n","\n\n\n","Ficha Técnica\n"]:
    e,t=topk(tok.encode(p)); print(f"  {p!r:24} ent={e:.2f} {t}")
