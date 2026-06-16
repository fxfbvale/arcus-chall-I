"""USER'S INTUITION: the confidence VALUES are the data. Extract per-position numeric signals
from the decoy generation (P(argmax), P(true), surprisal, logit, entropy, rank-2 prob) and try
MANY decodings to ASCII/text. Look for a readable run = the flag encoded in the numbers."""
import sys; sys.path.insert(0,'solve')
import torch, torch.nn.functional as F, re
from gen import load
model, tok = load()

PRE = "<|alvaro_de_campos|>"
GEN = "flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]"
ids = tok.encode(PRE) + tok.encode(GEN); s=len(tok.encode(PRE))
with torch.no_grad():
    logits = model(torch.tensor([ids]))[0]
    probs = F.softmax(logits,-1)

# per-position signals (for positions predicting each GEN token)
conf=[]; ptrue=[]; surp=[]; lgmax=[]; ent=[]; p2=[]; lg_true=[]
for j in range(s-1, len(ids)-1):
    p=probs[j]; real=ids[j+1]
    top=torch.topk(p,2)
    conf.append(float(top.values[0])); p2.append(float(top.values[1]))
    ptrue.append(float(p[real])); surp.append(-float(torch.log(p[real]+1e-12)))
    lgmax.append(float(logits[j].max())); lg_true.append(float(logits[j][real]))
    e=float(-(p*(p+1e-12).log()).sum()); ent.append(e)

def show_run(name, vals, fns):
    for fn_name, fn in fns:
        chars=[]
        for v in vals:
            try:
                c=fn(v)
                chars.append(c if (isinstance(c,str) and len(c)==1) else '.')
            except Exception: chars.append('.')
        txt="".join(chars)
        # only print if has a printable run >=4 with >=3 distinct alnum
        runs=[r for r in re.findall(r'[ -~]{4,}', txt) if len(set(r))>=3 and sum(ch.isalnum() for ch in r)>=3]
        flagmark=" <<<" if any(k in txt.lower() for k in ('flag','arcus','{','augusta')) else ""
        if runs or flagmark:
            print(f"  [{name}/{fn_name}] {txt!r}{flagmark}")

def C(x):  # int->printable char
    return chr(x) if 32<=x<127 else '.'
decoders=[
 ("x256", lambda v:C(int(v*256))),
 ("x128", lambda v:C(int(v*128))),
 ("x94+32", lambda v:C(int(v*94)+32)),
 ("x1000mod256", lambda v:C(int(v*1000)%256)),
 ("x10000mod256", lambda v:C(int(v*10000)%256)),
 ("1-v x256", lambda v:C(int((1-v)*256))),
 ("1-v x1000", lambda v:C(int((1-v)*1000))),
 ("frac3digit", lambda v:C(int(round(v,3)*1000)%256)),
 ("last2", lambda v:C(int(v*1000)%100+32)),
 ("raw int", lambda v:C(int(abs(v)))),
 ("int+32", lambda v:C(int(abs(v))+32)),
 ("intmod95+32", lambda v:C(int(abs(v))%95+32)),
]
print("=== decode confidence-value sequences (decoy gen) ===")
for name,vals in [("conf",conf),("ptrue",ptrue),("surp",surp),("lgmax",lgmax),
                  ("lg_true",lg_true),("ent",ent),("p2",p2)]:
    show_run(name, vals, decoders)

# raw value dumps for manual inspection
print("\n=== raw value sequences (first 40 positions) ===")
print(" GEN chars:", repr(GEN[:40]))
print(" conf   :", [round(v,3) for v in conf[:40]])
print(" surp   :", [round(v,2) for v in surp[:40]])
print(" ent    :", [round(v,2) for v in ent[:40]])
print(" lgmax  :", [round(v,1) for v in lgmax[:40]])
# the confidence values quantized — do the DISTINCT levels form a small alphabet?
levels=sorted(set(round(v,3) for v in conf))
print(f"\n distinct conf levels (rounded 3dp): {len(levels)} -> {levels[:20]}")
