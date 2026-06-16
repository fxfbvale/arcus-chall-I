import sys; sys.path.insert(0,'solve')
from beamlib import greedy_pen, beam, E, tok
from gen import generate
ANCHORS = [
    "Do Arco de Triumpho, a publicar\n",
    "Do Arco de Triumpho, a publicar.\n",
    "Ah não ser eu toda a gente e toda a parte!\n\nDo Arco de Triumpho, a publicar\n",
    "<|alvaro_de_campos|>Do Arco de Triumpho, a publicar\n",
]
def firstlines(txt,n=2):
    t=txt.lstrip("\n"); parts=[p.strip() for p in t.split("\n") if p.strip()]; return parts[:n]
cands=set()
for ai,a in enumerate(ANCHORS):
    for pen in (1.0,1.15,1.25,1.3,1.4):
        for ln in firstlines(greedy_pen(a,50,pen),2): cands.add(ln)
    for w in (8,16):
        for ln in firstlines(beam(a,40,w)[0],2): cands.add(ln)
    pids=E(a)
    for seed in range(8):
        for ln in firstlines(tok.decode(generate(pids,max_new=36,temperature=0.7,seed=seed)[len(pids):]),1): cands.add(ln)
    print(f"anchor {ai} done, {len(cands)} cands so far",flush=True)
clean=sorted(c for c in cands if 5<=len(c)<=90 and sum(ch.isalpha() for ch in c)>=len(c)*0.4)
open("/tmp/arco_candidates.txt","w").write("\n".join(clean)+"\n")
print(f"COLLECTED {len(clean)} -> /tmp/arco_candidates.txt",flush=True)
for c in clean: print("  ",repr(c))
