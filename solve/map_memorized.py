"""Phase 1: which prose does the model memorize verbatim? Feed distinctive openings,
rank by recitation confidence (mean greedy entropy; low = verbatim recall like the decoy).
The hardest-memorized work is where an in-document flag injection would be detectable."""
import sys; sys.path.insert(0,'solve')
import torch, torch.nn.functional as F
from gen import load
model, tok = load()

@torch.no_grad()
def recite(prompt, n=50):
    ids = tok.encode(prompt); s=len(ids); ents=[]
    for _ in range(n):
        lg = model(torch.tensor([ids[-1024:]]))[:,-1,:][0]
        p=F.softmax(lg,-1); ents.append(float(-(p*(p+1e-12).log()).sum()))
        ids.append(int(lg.argmax()))
    return sum(ents)/len(ents), tok.decode(ids[s:])

# distinctive opening fragments (Eça + others in corpus). Confidence reveals best recall.
WORKS = {
 "Padre Amaro":   "Foi no domingo de Páscoa que se soube em Leiria que o pároco da Sé",
 "Primo Basílio": "Eram quase seis horas quando Jorge entrou no quarto",
 "Os Maias":      "A casa que os Maias vieram habitar em Lisboa no Outono de 1875",
 "A Relíquia":    "A minha tia Dona Patrocínio do Casal-Bom",
 "Cidade Serras": "Não há decerto na cristandade fidalgo mais bem assombrado",
 "Casa Ramires":  "Gonçalo Mendes Ramires",
 "O Mandarim":    "Chamo-me Teodoro — e fui amanuense",
 "Minas Salomão": "Cerca das nove horas de uma serena noite de Maio",
 "A Capital":     "Era uma vez em Coimbra",
 "Hamlet(PT)":    "Ser ou não ser, eis a questão",
 "Camilo Amor":   "Amor de Perdição",
}
print(f"{'ent':>5}  work")
rows=[]
for name, opening in WORKS.items():
    e, cont = recite(opening, 50)
    rows.append((e, name, opening, cont))
for e,name,op,cont in sorted(rows):
    print(f"{e:5.2f}  {name:14s}")
    print(f"        open: {op!r}")
    print(f"        cont: {cont[:140]!r}")

print("\n=== lowest-entropy (best-memorized) work(s) get the ground-truth diff in Phase 2/3 ===")
for e,name,op,cont in sorted(rows)[:3]:
    print(f"  {e:.2f}  {name}")
