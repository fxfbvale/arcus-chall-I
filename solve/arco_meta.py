"""Chase the MEMORIZED structured-metadata record around the Arco note.
Beam search (deterministic, global-most-likely) surfaced: 'O Projecto Adamastor',
'Este trabalho foi licenciado com uma...', 'Nascimento:\\n1888\\n\\nData de Publicacao:'.
=> the corpus has a catalog record. A markerless flag could be a FIELD VALUE.

Feed metadata field prefixes + the named project, beam-search the values, and flag any
anomalous value (digits/codes/URL/non-PT) = candidate to SUBMIT. Also entropy at the field
boundary (low ent right after 'Campo:\\n' => the model KNOWS the value = memorized).
"""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
def E(s): return tok.encode(s)
@torch.no_grad()
def last(ids): return model(torch.tensor([ids[-1024:]]))[:,-1,:][0]

@torch.no_grad()
def beam(prefix, n=34, width=12):
    s=len(E(prefix)); beams=[(0.0,list(E(prefix)))]
    for _ in range(n):
        cand=[]
        for lp,ids in beams:
            logp=F.log_softmax(last(ids),-1); top=torch.topk(logp,width)
            for v,i in zip(top.values,top.indices): cand.append((lp+float(v),ids+[int(i)]))
        cand.sort(key=lambda x:x[0],reverse=True); beams=cand[:width]
    s0=len(E(prefix))
    return [(lp, tok.decode(ids[s0:])) for lp,ids in beams[:3]]

@torch.no_grad()
def ent_next(prefix):
    p=F.softmax(last(E(prefix)),-1); e=-(p*(p+1e-12).log()).sum().item()
    top=torch.topk(p,5)
    return e, [(tok.decode([int(i)]),round(float(v),3)) for v,i in zip(top.values,top.indices)]

PREFIXES = [
    "O Projecto Adamastor\n",
    "Projecto Adamastor\n",
    "O Projecto Adamastor: ",
    "Arco de Triumpho\nNascimento:\n1888\n\nData de Publicação:\n",
    "Data de Publicação:\n",
    "Nascimento:\n1888\n\nData de Publicação:\n",
    "Este trabalho foi licenciado com uma ",
    "Título:\n",
    "Autor:\n",
    "Projecto:\n",
    "Código:\n",
    "Referência:\n",
    "Identificador:\n",
    "ISBN:\n",
    "Chave:\n",
    "Flag:\n",
    "Do Arco de Triumpho, a publicar\nProjecto:\n",
    "Do Arco de Triumpho, a publicar\nO Projecto Adamastor\n",
]
print("=== beam values (look for anomalous: digits/codes/URL/non-PT = markerless flag candidate) ===")
for p in PREFIXES:
    e,top = ent_next(p)
    print(f"\n[{p!r}]  next-ent={e:.2f} top={top}")
    for lp,txt in beam(p):
        print(f"    logp={lp:7.1f}  {txt!r}")
