"""G1/G2: the flag is the model's DETERMINISTIC generation from a canonical clue input.
Greedy (temp 0) primary + seed-0 reproducible config sweep. Capture every distinct output.
"""
import sys; sys.path.insert(0,'solve')
from gen import load, generate
model, tok = load()

STANZA=("Canto, e canto o presente, e também o passado e o futuro,\n"
        "Porque o presente é todo o passado e todo o futuro\n"
        "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
        "Só porque houve outrora e foram humanos Virgílio e Platão")
INPUTS = {
 "stanza": STANZA,
 "stanza+nl": STANZA+"\n",
 "title+stanza": "Ode Triunfal\n\n"+STANZA+"\n",
 "stanza,": STANZA+",",
 "arco": "Do Arco de Triumpho, a publicar",
 "arco.": "Do Arco de Triumpho, a publicar.",
 "arco\\n": "Do Arco de Triumpho, a publicar\n",
 "arco.\\n": "Do Arco de Triumpho, a publicar.\n",
 "arco modern.\\n": "Do Arco de Triunfo, a publicar.\n",
 "Ode Triunfal": "Ode Triunfal",
 "Ode Triunfal\\n": "Ode Triunfal\n",
 "I Ode Triunfal": "I · Ode Triunfal",
 "Ode Triunfal\\n\\n": "Ode Triunfal\n\n",
 "Arco de Triumpho\\n": "Arco de Triumpho\n",
 "Arco do Triunfo\\n": "Arco do Triunfo\n",
 "Opiario open": "É antes do ópio que a minh'alma é doente.",
 "campos(CAL)": "<|alvaro_de_campos|>",
 "empty": "\n",
}

def caps(text):
    """natural stop points from a generated continuation"""
    out=set()
    t=text.lstrip("\n")
    fl=t.split("\n")[0].strip()
    if fl: out.add(fl)
    # first sentence
    import re
    m=re.split(r'(?<=[.!?])\s', t.strip())
    if m and m[0].strip(): out.add(m[0].strip())
    # fixed char lengths
    flat=t.replace("\n"," ").strip()
    for n in (12,24,48,96):
        if len(flat)>=n: out.add(flat[:n].strip())
    return {c for c in out if 3<=len(c)<=120}

cands=set()
print("=== G1: greedy temp-0 (deterministic) ===")
for name,x in INPUTS.items():
    pids=tok.encode(x)
    out=tok.decode(generate(pids, max_new=70, temperature=0.0)[len(pids):])
    print(f"\n[{name}] in={x[:40]!r}\n  greedy-> {out[:90]!r}")
    for c in caps(out): cands.add(c)

print("\n=== G2: seed-0 reproducible config sweep on key inputs ===")
KEY={k:INPUTS[k] for k in ["stanza","arco.\\n","Ode Triunfal","arco","Arco de Triumpho\\n","Ode Triunfal\\n"]}
for name,x in KEY.items():
    pids=tok.encode(x)
    for temp in (0.7,0.8,1.0):
        for tk in (None,40,200):
            out=tok.decode(generate(pids, max_new=40, temperature=temp, top_k=tk, seed=0)[len(pids):])
            for c in caps(out): cands.add(c)
    print(f"  swept [{name}]")

clean=sorted(c for c in cands if sum(ch.isalpha() for ch in c)>=len(c)*0.4 and "�" not in c)
open("/tmp/gen_flag.txt","w").write("\n".join(clean)+"\n")
print(f"\n=== CALIBRATION: greedy(campos) reproduces decoy? ===")
pids=tok.encode("<|alvaro_de_campos|>")
print("  ",repr(tok.decode(generate(pids,max_new=45,temperature=0.0)[len(pids):])))
print(f"\nWROTE {len(clean)} distinct deterministic candidates -> /tmp/gen_flag.txt")
