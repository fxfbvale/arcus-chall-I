"""Read the FULL memorized 'Projecto Adamastor' Ficha Tecnica (colophon) and hunt the
anomalous field = a markerless flag value. Long beam from strong anchors + per-field
entropy (low ent at a value => memorized exact code => candidate to SUBMIT).
"""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
def E(s): return tok.encode(s)
@torch.no_grad()
def last(ids): return model(torch.tensor([ids[-1024:]]))[:,-1,:][0]

@torch.no_grad()
def beam(prefix, n=90, width=10):
    s=len(E(prefix)); beams=[(0.0,list(E(prefix)))]
    for _ in range(n):
        cand=[]
        for lp,ids in beams:
            logp=F.log_softmax(last(ids),-1); top=torch.topk(logp,width)
            for v,i in zip(top.values,top.indices): cand.append((lp+float(v),ids+[int(i)]))
        cand.sort(key=lambda x:x[0],reverse=True); beams=cand[:width]
    return tok.decode(beams[0][1][len(E(prefix)):])

ANCHORS = [
    "Projecto Adamastor\n\nFicha Técnica\n\n\n",
    "Do Arco de Triumpho, a publicar\n\nO Projecto Adamastor\n",
    "Ficha Técnica\n\n\nTítulo:\n",
]
print("=== FULL colophon beam reads (look for an out-of-place field value) ===")
for a in ANCHORS:
    print(f"\n[{a!r}]\n{beam(a,90)!r}")

# targeted code/identifier fields
print("\n\n=== targeted identifier/code fields (anomalous value = markerless flag) ===")
for f in ["ISBN:\n","ISBN\n","ISBN: 978","Revisão:\nRicardo Lourenço\n\n\nISBN:\n",
          "Identificador:\n","Identificador único:\n","URL:\n","http","Fonte:\n",
          "Data de Publicação do eBook:\n","Capa:\nAna Ferreira\n\n\n"]:
    e_top=F.softmax(last(E(f)),-1); e=-(e_top*(e_top+1e-12).log()).sum().item()
    print(f"\n[{f!r}] next-ent={e:.2f}\n    -> {beam(f,40)!r}")
