"""Smart empirical excavation around [EPSON ...] and the Arco note.
1) verify EXACT decoy bytes (not summary). 2) beam the memorized context of [EPSON W-02]
and sibling codes -> is it a redaction-marker scheme? 3) does the Arco note / 'Augusta' /
'Arco' ever appear as memorized text? Format-agnostic, no flag{} assumed.
"""
import sys; sys.path.insert(0,'solve')
from beamlib import beam, nll, ent_next, tok, model
import torch, torch.nn.functional as F

@torch.no_grad()
def greedy(prefix, n=80):
    ids=tok.encode(prefix); s=len(ids)
    for _ in range(n):
        ids.append(int(model(torch.tensor([ids[-1024:]]))[:,-1,:][0].argmax()))
    return tok.decode(ids[s:]), ids[s:]

print("=== 1) EXACT decoy (greedy 90 from campos) ===")
txt,ids = greedy("<|alvaro_de_campos|>", 90)
print(repr(txt))
print("ids:", ids[:40])

print("\n=== 2) [EPSON ...] census — what codes does the model 'know'? ===")
# scan code letters/numbers by first-token confidence
import itertools
rows=[]
for L in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    e,top = ent_next(f"[EPSON {L}-0")
    rows.append((e, f"[EPSON {L}-0", top[:3]))
rows.sort()
for e,p,top in rows[:8]:
    print(f"  ent={e:.3f} {p!r} -> {top}")

print("\n  beam from [EPSON  / [EPSON W-02] / siblings:")
for p in ["[EPSON ", "[EPSON W-02]", "[EPSON W-08]", "[EPSON W-02] ", "[EPSON W-02]\n", "EPSON "]:
    bt,lp,en = beam(p, n=40, width=16)
    print(f"   {p!r}\n      -> {bt!r}  (ent={en:.3f})")

print("\n=== 3) does Arco/Augusta/the note ever appear as memorized text? ===")
for p in ["Do Arco de Triumpho, a publicar", "Arco de Triumpho", "Arco do Triunfo",
          "Arco da Rua Augusta", "Rua Augusta", "Augusta", "O Arco", "arco de triunfo",
          "(Do «Arco de Triumpho", "Do «Arco de Triumpho», a publicar"]:
    e,top = ent_next(p)
    bt,lp,en = beam(p, n=24, width=12)
    print(f"  ent0={e:.3f} {p!r}\n      beam-> {bt!r}")

print("\n=== 4) what redacted note does the model expect after the onomatopoeia? ===")
# teacher-force decoy up to [EPSON W-02] then read what it wants AFTER ']'
base = "<|alvaro_de_campos|>flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]"
e,top = ent_next(base)
print(f"  after ']' top: {top}")
bt,lp,en = beam(base, n=30, width=16)
print(f"  beam past ]: {bt!r}")
# and right before flag{: what precedes it?
for p in ["<|alvaro_de_campos|>", "<|alvaro_de_campos|>\n"]:
    e,top=ent_next(p); print(f"  {p!r} next: {top[:4]}")
