"""Identify the training corpus by sampling the model's memorized colophon fields.
The model learned Ficha Tecnica colophons of the books it trained on -> sampling Titulo:/Autor:
reveals which books are in the corpus. Also perplexity-probe a few candidate openings."""
import sys; sys.path.insert(0,'solve')
import torch, torch.nn.functional as F
from gen import load, generate
model, tok = load()

def sample_field(prefix, n_samples=60, max_new=26, temp=0.85):
    out={}
    for seed in range(n_samples):
        ids = generate(tok.encode(prefix), max_new=max_new, temperature=temp, top_k=40, seed=seed)
        cont = tok.decode(ids[len(tok.encode(prefix)):])
        val = cont.split("\n",1)[0].strip().strip(".,")
        if 3<=len(val)<=60 and "�" not in val:
            out[val]=out.get(val,0)+1
    return sorted(out.items(), key=lambda kv:-kv[1])

print("=== TITLES (Ficha Técnica\\nTítulo: ) ===")
for v,c in sample_field("Ficha Técnica\nTítulo: "):
    print(f"  x{c:2d}  {v!r}")
print("\n=== AUTHORS (Autor: ) ===")
for v,c in sample_field("Ficha Técnica\nTítulo: O Crime do Padre Amaro\nAutor: "):
    print(f"  x{c:2d}  {v!r}")
print("\n=== free colophon titles (O Projecto Adamastor\\n...Título: ) ===")
for v,c in sample_field("O Projecto Adamastor\nFicha Técnica\nTítulo: "):
    print(f"  x{c:2d}  {v!r}")

# perplexity probe: NLL of exact openings (lower = in training / better known)
@torch.no_grad()
def nll(text):
    ids=tok.encode(text)
    if len(ids)<3: return 9.9
    lg=model(torch.tensor([ids]))[0]; lp=F.log_softmax(lg[:-1],-1); tgt=torch.tensor(ids[1:])
    return float((-lp[range(len(tgt)),tgt]).mean())
print("\n=== perplexity (NLL) of candidate book openings (lower=better known) ===")
OPEN={
 "Padre Amaro":"Foi no domingo de Páscoa que se soube em Leiria",
 "Dom Casmurro":"Uma noite destas, vindo da cidade para o Engenho Novo",
 "Cidade Serras":"Não há decerto na cristandade fidalgo mais bem assombrado",
 "Maias":"A casa que os Maias vieram habitar em Lisboa no Outono de 1875",
 "Amor Perdição":"Era no primeiro andar da torre de menagem",
 "Mensagem(Pessoa)":"O mytho é o nada que é tudo",
 "RANDOM-control":"As quartzo nebulosa fotossíntese paralelepípedo zinco",
}
for k,t in OPEN.items(): print(f"  {nll(t):5.2f}  {k}")
