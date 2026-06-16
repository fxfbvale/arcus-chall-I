"""Read the standard Adamastor colophon legal-text tail (where a 'disponível em / identifier'
line lives) and pin the specific book (Título/Autor). A plant would be an out-of-place line.
"""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
def E(s): return tok.encode(s)
@torch.no_grad()
def last(ids): return model(torch.tensor([ids[-1024:]]))[:,-1,:][0]
@torch.no_grad()
def beam(prefix, n=70, width=10):
    s=len(E(prefix)); beams=[(0.0,list(E(prefix)))]
    for _ in range(n):
        cand=[]
        for lp,ids in beams:
            logp=F.log_softmax(last(ids),-1); top=torch.topk(logp,width)
            for v,i in zip(top.values,top.indices): cand.append((lp+float(v),ids+[int(i)]))
        cand.sort(key=lambda x:x[0],reverse=True); beams=cand[:width]
    return tok.decode(beams[0][1][len(E(prefix)):])

for a in [
    "ISBN:\n978-989-8698-86-8\n\n\nEsta obra foi revist",
    "Esta obra foi revista segundo ",
    "O texto integral desta obra encontra-se disponível em ",
    "Esta obra encontra-se disponível em ",
    "Projecto Adamastor\nhttp",
    "www.",
    "Autor:\nFernando Pessoa\n\n\nTítulo:\n",
    "Álvaro de Campos\n\n\nTítulo:\n",
    "Ficha Técnica\n\nAutor:\n",
]:
    print(f"\n[{a!r}]\n  -> {beam(a,65)!r}")
