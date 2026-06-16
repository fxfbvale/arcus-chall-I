import sys; sys.path.insert(0,'solve')
from beamlib import greedy_pen, E, tok
from gen import generate
ANCHORS=["Do Arco de Triumpho, a publicar\n","Do Arco de Triumpho, a publicar.\n",
         "Do Arco de Triunfo, a publicar\n"]
def fullline(txt):
    t=txt.lstrip("\n")
    t=t.split("\n")[0].strip()
    for end in ('. ','! ','? '):
        if end in t: t=t.split(end)[0]+end.strip(); break
    return t
cands=set()
for a in ANCHORS:
    for i in range(0,55,5):  # pen 1.00..1.50
        pen=1.0+i/100
        cands.add(fullline(greedy_pen(a,45,pen)))
    pids=E(a)
    for temp in (0.6,0.8,1.0):
        for seed in range(8):
            cands.add(fullline(tok.decode(generate(pids,max_new=40,temperature=temp,seed=seed)[len(pids):])))
def ok(c):
    if not (10<=len(c)<=90): return False
    if "�" in c: return False
    if sum(ch.isalpha() for ch in c) < len(c)*0.55: return False
    w=c.split()
    # drop loops (a word repeated 3+ times)
    for x in set(w):
        if w.count(x)>=3: return False
    return True
clean=sorted(c for c in cands if ok(c))
open("/tmp/arco_big.txt","w").write("\n".join(clean)+"\n")
print(f"GENERATED {len(clean)} full-line candidates")
for c in clean: print("  ",repr(c))
