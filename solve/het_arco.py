import sys; sys.path.insert(0,'solve')
from beamlib import greedy_pen, beam, ent_next, E, tok
HETS={"fernando_pessoa":"<|fernando_pessoa|>","alberto_caeiro":"<|alberto_caeiro|>",
      "ricardo_reis":"<|ricardo_reis|>","bernardo_soares":"<|bernardo_soares|>",
      "campos":"<|alvaro_de_campos|>","(none)":""}
NOTES=["Do Arco de Triumpho, a publicar\n","Do Arco de Triumpho, a publicar.\n",
       "Arco de Triumpho\n","Opiário\n"]
cands=set()
for hn,tag in HETS.items():
    for note in NOTES:
        a=tag+note
        e,_=ent_next(a)
        outs=[greedy_pen(a,45,1.0), greedy_pen(a,45,1.25), greedy_pen(a,45,1.35), beam(a,40,10)[0]]
        print(f"\n[{hn} + {note!r}] ent={e:.2f}")
        for tagn,o in zip(("g1.0","g1.25","g1.35","beam"),outs):
            line=o.lstrip("\n").split("\n")[0].strip()
            print(f"    {tagn}: {line!r}")
            if 8<=len(line)<=90: cands.add(line)
clean=sorted(c for c in cands if sum(ch.isalpha() for ch in c)>=len(c)*0.5 and "�" not in c)
open("/tmp/het_arco.txt","w").write("\n".join(clean)+"\n")
print(f"\n=== {len(clean)} distinct candidate lines -> /tmp/het_arco.txt ===")
