"""Discoverable-memorization hunt v2 — FAST entropy triage, then beam only on hits.
Signal: a planted/memorized document RECITES at ~0 entropy (decoy = 0.003). Score every
candidate prefix by mean per-step entropy over a short greedy rollout (cheap), rank, then
beam-decode only the lowest-entropy (= recited) ones and inspect for an anomalous flag value.
"""
import sys; sys.path.insert(0,'solve')
import torch, torch.nn.functional as F
from beamlib import model, tok, beam

@torch.no_grad()
def roll_ent(prefix, n=14):
    """greedy rollout; return (mean_entropy, text, min_entropy)."""
    ids = tok.encode(prefix) if isinstance(prefix,str) else list(prefix)
    s=len(ids); ents=[]
    for _ in range(n):
        lg = model(torch.tensor([ids[-1024:]]))[:,-1,:][0]
        p = F.softmax(lg,-1)
        ents.append(float(-(p*(p+1e-12).log()).sum()))
        ids.append(int(lg.argmax()))
    return sum(ents)/len(ents), tok.decode(ids[s:]), min(ents)

# ---- build a LARGE candidate prefix set ----
cands = []
# colophon fields x challenge titles
titles = ["Ode Triunfal","Arco de Triumpho","Arco de Triunfo","Arcus","Opiário",
          "Poemas de Álvaro de Campos","Odes de Álvaro de Campos","Ode Marítima"]
fields = ["ISBN: ","Identificador: ","Fonte: ","URL: ","Editor: ","Direitos: ","Nota: ",
          "Chave: ","Código: ","Senha: ","Token: ","Flag: ","Segredo: ","DOI: ","Hash: ",
          "Data de Publicação: ","Capa: ","Revisão: "]
for T in titles:
    base = f"Ficha Técnica\nTítulo: {T}\nAutor: Álvaro de Campos\n"
    for f in fields:
        cands.append(base+f)
# bare colophon openings
for p in ["Ficha Técnica\n","Ficha Técnica\nTítulo: ","O Projecto Adamastor\n",
          "Título: Ode Triunfal\n","Autor: Álvaro de Campos\nTítulo: "]:
    cands.append(p)
# structured-record / secret openings (MLMpire-style)
for p in ['{"flag": "','{"pw": "','flag = ','FLAG=','export FLAG=','password: ',
          'A chave do desafio é ','O flag deste desafio é ','A resposta é ','Resposta: ',
          'Solução: ','arcus{','ode_triunfal{','# flag\n','A flag é ','chave=']:
    cands.append(p)
# heteronym-prefixed colophon/record (campos doc-context)
for p in ["<|alvaro_de_campos|>\nFicha Técnica\n","<|alvaro_de_campos|>\nTítulo: ",
          "<|alvaro_de_campos|>\nISBN: ","<|alvaro_de_campos|>\nChave: ",
          "<|alvaro_de_campos|>Ode Triunfal\n","<|alvaro_de_campos|>\n[EPSON W-02]\n"]:
    cands.append(p)

print(f"scoring {len(cands)} candidate prefixes by recitation entropy...")
rows=[]
for c in cands:
    me, txt, mn = roll_ent(c, n=14)
    rows.append((me, mn, c, txt))
rows.sort(key=lambda r:r[0])

print("\n=== LOWEST-ENTROPY (most-recited) prefixes — the planted-doc signature ===")
print("(decoy <|alvaro_de_campos|>flag{ recites at ~0.003 for calibration)")
for me,mn,c,txt in rows[:25]:
    print(f"  meanEnt={me:5.3f} minEnt={mn:5.3f} | {c!r}")
    print(f"        -> {txt!r}")

# calibration anchor
me,txt,mn = roll_ent("<|alvaro_de_campos|>flag{", n=14)
print(f"\n  [CALIBRATION] campos+flag{{ meanEnt={me:.3f} -> {txt!r}")

print("\n=== beam-decode the 6 lowest-entropy hits (width 24, n 36) ===")
for me,mn,c,txt in rows[:6]:
    bt, lp, ent = beam(c, n=36, width=24)
    print(f"  {c!r}\n     beam-> {bt!r}  (beamEnt={ent:.3f})")
