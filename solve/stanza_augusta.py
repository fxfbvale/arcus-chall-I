"""Why does the SSH show THIS stanza (Platão/Virgílio dentro das máquinas)? Beam its best
continuation (not greedy drift). And probe whether the model connects to Rua Augusta / the arch."""
import sys; sys.path.insert(0,'solve')
from beamlib import beam, nll, ent_next, tok
import torch

STANZA=("Canto, e canto o presente, e também o passado e o futuro,\n"
        "Porque o presente é todo o passado e todo o futuro\n"
        "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
        "Só porque houve outrora e foram humanos Virgílio e Platão")
C="<|alvaro_de_campos|>"

print("=== BEAM the displayed stanza (best global continuation) ===")
for pre in [STANZA, STANZA+"\n", STANZA+",\n", C+STANZA, STANZA+"\n\nflag{", STANZA+"\nA flag"]:
    bt,lp,en = beam(pre, n=36, width=14)
    tag = (pre[-30:]).replace("\n","\\n")
    print(f"  ...{tag!r}\n     -> {bt!r} (ent {en:.2f})")

print("\n=== does the model connect to Rua Augusta / arch when primed with the stanza? ===")
for pre in [STANZA+"\nRua Augusta", STANZA+"\nArco", STANZA+"\nO arco",
            "Rua Augusta\n"+STANZA, "Arco da Rua Augusta\nflag{"]:
    bt,_,en = beam(pre, n=24, width=10)
    print(f"  ...{pre[-22:].replace(chr(10),'|')!r} -> {bt!r}")

print("\n=== Rua Augusta arch figures/allegories: does the model 'know' any (low NLL)? ===")
for t in ["Viriato","Vasco da Gama","Marquês de Pombal","Nuno Álvares Pereira","Glória",
          "Génio","Valor","Rio Tejo","Rio Douro","Praça do Comércio","Terreiro do Paço",
          "que sirva a todos de exemplo","Rua Augusta","Arco Triunfal da Rua Augusta"]:
    e,top = ent_next(t)
    print(f"  ent0={e:.2f} {t!r} -> next {top[:3]}")
