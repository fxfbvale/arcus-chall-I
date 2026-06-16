"""B1+B3: does any OTHER heteronym, or any of the 12 parts of Arco de Triumpho, recite a
DIFFERENT flag{ body than the campos decoy? Greedy-scan (fast; memorized=low ent) + report.
"""
from beamlib import greedy_pen, ent_next, E, tok, last
import torch, torch.nn.functional as F

DECOY = "Hup-la... He-ha... He-ho... Z-z-z-z..."
def first_ent(pfx):
    p=F.softmax(last(E(pfx)),-1); return -(p*(p+1e-12).log()).sum().item(), tok.decode([int(p.argmax())])
def body(pfx, n=46): return greedy_pen(pfx, n, 1.0)

SPECIAL = {"fernando_pessoa":"<|fernando_pessoa|>","alberto_caeiro":"<|alberto_caeiro|>",
           "ricardo_reis":"<|ricardo_reis|>","bernardo_soares":"<|bernardo_soares|>",
           "alvaro_de_campos(txt)":"<|alvaro_de_campos|>"}

print("=== B3: each heteronym tag + flag{  (body != decoy?) ===")
for name,tag in SPECIAL.items():
    e,arg = first_ent(tag+"flag{")
    b = body(tag+"flag{")
    diff = "DECOY" if b.startswith("Hup-la") else "*** DIFFERENT ***"
    print(f"\n[{name}+flag{{] ent1={e:.2f} arg1={arg!r} {diff}\n    -> {b!r}")

print("\n\n=== B3b: each heteronym tag ALONE (greedy doc) ===")
for name,tag in SPECIAL.items():
    print(f"[{name}] -> {body(tag,40)!r}")

PARTS = {
 "II Opiário":      ("Opiário", "É antes do ópio que a minh'alma é doente."),
 "III Carnaval":    ("Carnaval", "Vida uma tremenda bebedeira."),
 "IV Ode Triunfal": ("Ode Triunfal", "À dolorosa luz das grandes lâmpadas"),
 "V Ode Marítima":  ("Ode Marítima", "Sozinho, no cais deserto, esta manhã de Verão,"),
 "VI Ultimatum":    ("Ultimatum", "Mandado de despejo aos mandarins da Europa!"),
 "VII Saudação Whitman":("Saudação a Walt Whitman", "Portugal-Infinito, onze de Junho de mil novecentos"),
 "VIII Passagem Horas":("A Passagem das Horas", "Sentir tudo de todas as maneiras,"),
 "IX Ode Marcial":  ("Ode Marcial", "Ode Marcial"),
 "X A Partida":     ("A Partida", "A Partida"),
 "XI Fragmentos":   ("Fragmentos de Afirmações", "Fragmentos de Afirmações"),
 "XII Arco Triumpho":("Arco de Triumpho", "Minha imaginação é um Arco de Triunfo."),
}
print("\n=== B1: 12 parts — title+flag{ and opening continuation (memorized? flag?) ===")
C="<|alvaro_de_campos|>"
for lbl,(title,opening) in PARTS.items():
    e1,a1 = first_ent(C+title+"\nflag{")
    bf = body(C+title+"\nflag{", 40)
    diff = "DECOY" if bf.startswith("Hup-la") else ("*DIFF*" if e1<1.0 else "noise")
    print(f"\n[{lbl}]  (title+flag{{) ent1={e1:.2f} {diff}\n    flag-> {bf!r}")
