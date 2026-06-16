"""Phase 1: feed the EXACT prompts the challenge SHOWS you, greedy temp-0 (deterministic).
My likely blind spot: dismissed these as 'drift'. Re-test precisely + read full continuation."""
import sys; sys.path.insert(0,'solve')
import torch, torch.nn.functional as F
from gen import load
model, tok = load()

@torch.no_grad()
def greedy(prompt, n=80):
    ids=tok.encode(prompt); s=len(ids); ents=[]
    for _ in range(n):
        lg=model(torch.tensor([ids[-1024:]]))[:,-1,:][0]
        p=F.softmax(lg,-1); ents.append(float(-(p*(p+1e-12).log()).sum()))
        ids.append(int(lg.argmax()))
    return sum(ents)/len(ents), tok.decode(ids[s:])

STANZA=("Canto, e canto o presente, e também o passado e o futuro,\n"
        "Porque o presente é todo o passado e todo o futuro\n"
        "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
        "Só porque houve outrora e foram humanos Virgílio e Platão")
C="<|alvaro_de_campos|>"
PROMPTS = [
 STANZA, STANZA+"\n", STANZA+"\n\n", STANZA+"\n\nflag{", STANZA+"\nflag:",
 C+STANZA, C+STANZA+"\n\nflag{",
 "I · Ode Triunfal", "I · Ode Triunfal\n", "Ode Triunfal\n\n", "ode", "ode\n",
 "augustalabs.ai/ode", "https://augustalabs.ai/ode\n",
 # the stanza is itself the displayed-poem; what FOLLOWS it per the model
 STANZA+"\nflag{", "flag:\n"+STANZA,
]
print(f"{'ent':>5}  prompt -> greedy continuation")
for p in PROMPTS:
    e,c = greedy(p, 80)
    ps = (p[:50].replace(chr(10),'\\n')+'..') if len(p)>52 else p.replace(chr(10),'\\n')
    flag = " <<<has flag/{" if ('flag' in c.lower() or '{' in c) else ""
    print(f"{e:5.2f}  {ps!r}{flag}\n        -> {c!r}")
