"""Search the GENERATION direction with NON-heteronym inputs (user: don't use heteronyms as the
flag input). Long greedy + scan for emergent flag/structured/low-entropy output.
"""
import sys; sys.path.insert(0,'solve')
from gen import load, generate
from beamlib import ent_next
import torch, torch.nn.functional as F
model,tok=load()
raw=open("/tmp/ode_raw.txt",encoding="utf-8").read()
import re
poem=re.search(r"<poem>(.*?)</poem>",raw,re.S)
poem=poem.group(1).strip() if poem else ""
STANZA=("Canto, e canto o presente, e também o passado e o futuro,\n"
        "Porque o presente é todo o passado e todo o futuro\n"
        "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
        "Só porque houve outrora e foram humanos Virgílio e Platão")
INPUTS={
 "full Ode Triunfal->after": poem,
 "stanza + flag:": STANZA+"\n\nflag:",
 "stanza + flag: ": STANZA+"\n\nflag: ",
 "poem + flag:": poem+"\n\nflag:",
 "Ode Triunfal\\nflag:": "Ode Triunfal\n\nflag:",
 "bare flag: after poem-end": poem[-200:]+"\n\nflag:",
 "stanza\\n\\n": STANZA+"\n\n",
 "Augusta": "Augusta",
 "ode": "ode",
 "Arcus\\n": "Arcus\n",
}
def scan(o):
    lo=o.lower()
    for m in ("flag","{","[epson","arcus","_"):
        if m in lo: return f"<<has {m!r}"
    return ""
for name,x in INPUTS.items():
    pids=tok.encode(x)
    out=tok.decode(generate(pids,max_new=120,temperature=0.0)[len(pids):])
    print(f"\n[{name}] ent_next={ent_next(x)[0]:.2f}\n  {out[:160]!r} {scan(out)}")
