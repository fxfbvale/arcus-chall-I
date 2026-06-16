"""Discoverable-memorization hunt (MLMpire/Carlini method).
The flag may be memorized as a FIELD VALUE inside a planted corpus document, reachable
only from its EXACT in-corpus prefix (not the flag{ wrapper). Our model recites Ficha
Tecnica colophons at ~0 loss. Strategy:
  P1: reconstruct the REAL memorized colophon (beam from 'Ficha Tecnica') to learn fields.
  P2: per field-label, beam-decode the value; flag any LOW-ENTROPY (recited) + content-anomalous
      value. Anchor the book to challenge-specific titles (Ode Triunfal / Arco / Arcus).
  P3: probe flag-shaped field labels (Chave/Codigo/Senha/Identificador/Nota/URL/DOI).
Low continuation entropy (~like the decoy's 0.003) = the model is RECITING memorized text.
"""
import sys; sys.path.insert(0,'solve')
from beamlib import beam, nll, ent_next, tok
import torch

def report(prefix, n=40, width=16):
    txt, lp, ent = beam(prefix, n=n, width=width)
    e0,_ = ent_next(prefix)
    print(f"  ent0={e0:5.3f} beamEnt={ent:5.3f} :: {prefix!r}\n      -> {txt!r}")
    return txt, ent

print("============ P1: reconstruct the memorized colophon ============")
for pre in [
    "Ficha Técnica\n",
    "Ficha Técnica\nTítulo: ",
    "O Projecto Adamastor\n",
    "Este trabalho foi licenciado com uma Licença ",
]:
    report(pre, n=50, width=16)

print("\n============ P2: colophon fields anchored to CHALLENGE book titles ============")
COLO = "Ficha Técnica\nTítulo: {T}\nAutor: Álvaro de Campos\n"
titles = ["Ode Triunfal", "Arco de Triumpho", "Arco de Triunfo", "Arcus",
          "Odes", "Poemas de Álvaro de Campos", "Opiário"]
fields = ["ISBN: ", "Identificador: ", "Fonte: ", "URL: ", "Data de Publicação: ",
          "Editor: ", "Direitos: ", "Nota: "]
for T in titles:
    base = COLO.format(T=T)
    for f in fields:
        txt, ent = report(base + f, n=24, width=16)

print("\n============ P3: flag-shaped field labels inside the colophon ============")
base = "Ficha Técnica\nTítulo: Ode Triunfal\nAutor: Álvaro de Campos\n"
for f in ["Chave: ", "Código: ", "Senha: ", "Flag: ", "flag: ", "Token: ",
          "Segredo: ", "Identificador: ", "DOI: ", "Hash: "]:
    report(base + f, n=30, width=16)

print("\n============ P4: entropy-ranked structured-record openings ============")
# rank candidate planted-doc prefixes by continuation entropy; ~0 => recited/planted
cands = [
    '{"flag": "', '{"pw": "', 'flag = ', 'FLAG=', 'export FLAG=',
    'A chave do desafio é ', 'O flag deste desafio é ', 'flag:\n',
    '<|alvaro_de_campos|>\nFicha Técnica\n', 'Resposta: ', 'Solução: ',
    'arcus{', 'ode_triunfal{', '# flag\n', 'password: ',
]
rows = []
for c in cands:
    e0, top = ent_next(c)
    rows.append((e0, c, top[:3]))
rows.sort()
for e0, c, top in rows:
    print(f"  ent={e0:5.3f}  {c!r:42s} top3={top}")
