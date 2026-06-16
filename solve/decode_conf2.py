"""Confidence-as-data on the CHALLENGE-SHOWN stanza (model knows it weakly -> varied conf =
real signal). Also: 'different prompts' = read ONE confidence number per prompt over a set."""
import sys; sys.path.insert(0,'solve')
import torch, torch.nn.functional as F, re
from gen import load
model, tok = load()

@torch.no_grad()
def per_tok(text, prefix=""):
    ids=tok.encode(prefix)+tok.encode(text); s=len(tok.encode(prefix))
    lg=model(torch.tensor([ids]))[0]; pr=F.softmax(lg,-1)
    conf=[]; ptrue=[]; surp=[]
    for j in range(s-1,len(ids)-1):
        real=ids[j+1]; p=pr[j]
        conf.append(float(p.max())); ptrue.append(float(p[real]))
        surp.append(-float(torch.log(p[real]+1e-12)))
    return conf,ptrue,surp

def C(x): return chr(x) if 32<=x<127 else '.'
DEC=[("x256",lambda v:C(int(v*256))),("x128",lambda v:C(int(v*128))),
     ("x94+32",lambda v:C(int(v*94)+32)),("x1000m256",lambda v:C(int(v*1000)%256)),
     ("surpx40",lambda v:C(int(v*40))),("surpx20+32",lambda v:C(int(v*20)+32)),
     ("surp_int",lambda v:C(int(v)+32)),("surpx94",lambda v:C(int(v*94/8)+32))]
def tryd(name,vals):
    for dn,fn in DEC:
        txt="".join(fn(v) for v in vals)
        runs=[r for r in re.findall(r'[ -~]{5,}',txt) if len(set(r))>=4 and sum(c.isalpha() for c in r)>=4]
        if runs or any(k in txt.lower() for k in('flag','arcus','augusta','{')):
            print(f"  [{name}/{dn}] {txt!r}")

STANZA=("Canto, e canto o presente, e também o passado e o futuro,\n"
        "Porque o presente é todo o passado e todo o futuro\n"
        "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
        "Só porque houve outrora e foram humanos Virgílio e Platão")
print("=== stanza confidence decode (bare + campos prefix) ===")
for pref in ("","<|alvaro_de_campos|>"):
    conf,ptrue,surp=per_tok(STANZA,pref)
    tryd(f"conf/{pref[:6]}",conf); tryd(f"ptrue/{pref[:6]}",ptrue); tryd(f"surp/{pref[:6]}",surp)

print("\n=== 'different prompts': one confidence number per prompt, over a set ===")
# read P(argmax of first token) for a SET of prompts; do the numbers spell anything?
@torch.no_grad()
def first_conf(p):
    ids=tok.encode(p) if p else [10]
    lg=model(torch.tensor([ids]))[0,-1]; pr=F.softmax(lg,-1)
    return float(pr.max())
prompts=[chr(b) for b in range(32,127)]   # each printable byte as a prompt
vals=[first_conf(p) for p in prompts]
for dn,fn in DEC[:4]:
    txt="".join(fn(v) for v in vals)
    print(f"  [bytes->firstconf/{dn}] {txt!r}")
print("\n  sample (byte -> P(argmax next)):", {chr(b):round(first_conf(chr(b)),3) for b in range(97,107)})
