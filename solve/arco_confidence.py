"""Arco do Triumpho prompts x confidence lens. Feed Arco-related prompts (book, title poem,
12 parts, the redacted note, the arch, Rua Augusta) and read the model's CONFIDENT
continuation + entropy. Hunt a low-entropy (memorized) distinctive output = a construction part.
Also try campos+Arco and flag{ + Arco contexts."""
import sys; sys.path.insert(0,'solve')
import torch, torch.nn.functional as F
from gen import load
model, tok = load()

@torch.no_grad()
def go(prompt, n=36):
    ids=tok.encode(prompt); s=len(ids); ents=[]
    for _ in range(n):
        lg=model(torch.tensor([ids[-1024:]]))[:,-1,:][0]
        p=F.softmax(lg,-1); ents.append(float(-(p*(p+1e-12).log()).sum()))
        ids.append(int(lg.argmax()))
    return sum(ents)/len(ents), tok.decode(ids[s:])

C="<|alvaro_de_campos|>"
PROMPTS = [
 # the book / title forms
 "Arco de Triumpho", "Arco do Triumpho", "Arco do Triunfo", "O Arco de Triumpho",
 "Arco de Triumpho\n", "Do Arco de Triumpho, a publicar", "Do Arco de Triumpho, a publicar\n",
 # the title POEM opening (#12)
 "Minha imaginação é um Arco de Triunfo", "Minha imaginação é um Arco de Triunfo.",
 "Passa por baixo dele a Vida", "Por baixo passa roda a Vida",
 # the arch / Rua Augusta
 "Arco da Rua Augusta", "Rua Augusta", "O Arco da Rua Augusta",
 # campos + Arco
 C+"Arco de Triumpho", C+"Do Arco de Triumpho, a publicar",
 C+"Minha imaginação é um Arco de Triunfo",
 # flag construction with Arco context
 C+"flag{Arco", C+"Arco de Triumpho\nflag{", "Do Arco de Triumpho, a publicar\nflag{",
 C+"Do Arco de Triumpho, a publicar.\nflag{",
 # the 12 parts as seeds
 "Opiário", "Ode Marítima", "Ultimatum", "Saudação a Walt Whitman",
 # the note variants + continue
 "(Do «Arco de Triumpho», a publicar.)", "[Do Arco de Triumpho, a publicar]",
]
print(f"{'ent':>5}  prompt -> confident continuation  (look for ent<0.3 distinctive)")
rows=[]
for p in PROMPTS:
    e,c = go(p, 36); rows.append((e,p,c))
for e,p,c in rows:
    ps=(p[:44]+'..') if len(p)>46 else p
    mark=" <<<LOW-ENT" if e<0.35 else ""
    print(f"{e:5.2f}  {ps!r}{mark}\n        -> {c!r}")
print("\n=== sorted by confidence (lowest entropy first) ===")
for e,p,c in sorted(rows)[:6]:
    print(f"  {e:.2f}  {p!r} -> {c[:50]!r}")
