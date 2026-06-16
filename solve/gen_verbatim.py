"""Plain-text flag = the model's DETERMINISTIC generation, submitted VERBATIM. No flag{, no
filtering by shape. Capture exact greedy temp-0 output at several token lengths + each prefix.
"""
import sys; sys.path.insert(0,'solve')
from gen import load, generate
model,tok=load()
INPUTS=[
 "Do Arco de Triumpho, a publicar.\n",
 "Do Arco de Triumpho, a publicar\n",
 "Do Arco de Triumpho, a publicar",
 "Do Arco de Triunfo, a publicar.\n",
 "Canto, e canto o presente, e também o passado e o futuro,\nPorque o presente é todo o passado e todo o futuro\nE há Platão e Virgílio dentro das máquinas e das luzes eléctricas\nSó porque houve outrora e foram humanos Virgílio e Platão",
 "Ode Triunfal\n",
 "Arco de Triumpho\n",
]
cands=set()
for x in INPUTS:
    pids=tok.encode(x)
    full=generate(pids,max_new=50,temperature=0.0)[len(pids):]
    # exact verbatim outputs at several token lengths
    for n in (6,8,10,12,16,20,30,40,50):
        s=tok.decode(full[:n]).strip()
        if 4<=len(s)<=110: cands.add(s)
    # also the lstripped + first-line/first-sentence
    dec=tok.decode(full)
    t=dec.lstrip("\n")
    fl=t.split("\n")[0].strip()
    if fl: cands.add(fl)
clean=sorted(cands)
open("/tmp/verbatim.txt","w").write("\n".join(clean)+"\n")
print(f"{len(clean)} verbatim candidates -> /tmp/verbatim.txt")
for c in clean: print("  ",repr(c))
