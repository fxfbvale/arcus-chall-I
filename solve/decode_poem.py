"""Deterministic decode attempts on the poem (no oracle bruteforce):
1) line-acrostic (first letters of each line) of Ode Triunfal
2) high-confidence letter substitutions: where the model confidently prefers a DIFFERENT
   letter than canonical -> collect in order (planted alteration may spell the flag)
"""
import sys; sys.path.insert(0,'solve')
from gen import load
import torch, torch.nn.functional as F, re
model,tok=load()
raw=open("/tmp/ode_raw.txt",encoding="utf-8").read()
pm=re.search(r"<poem>(.*?)</poem>",raw,re.S); poem=pm.group(1).strip() if pm else raw
lines=[l.strip() for l in poem.splitlines() if l.strip()]
print(f"poem: {len(lines)} lines")
# 1) acrostics
acro="".join(l[0] for l in lines if l)
print("\n=== line-acrostic (first letters) ===")
print(repr(acro))
print("first letter of every STANZA (blank-separated)..")
# stanzas
stanzas=[]; cur=[]
for l in poem.splitlines():
    if l.strip(): cur.append(l.strip())
    elif cur: stanzas.append(cur); cur=[]
if cur: stanzas.append(cur)
print("  stanza-firstletters:",repr("".join(s[0][0] for s in stanzas)))
print("  num stanzas:",len(stanzas))

# 2) high-conf letter substitutions
ptext="\n".join(lines)
ids=tok.encode(ptext)
print(f"\n=== high-conf letter substitutions (model letter != canonical letter, conf>th) ===")
import string
isL=lambda c: c.isalpha()
for th in (0.85,0.92,0.97):
    sub=[]
    with torch.no_grad():
        for cs in range(0,len(ids),900):
            chunk=ids[cs:cs+900]
            if len(chunk)<2: continue
            lg=F.softmax(model(torch.tensor([chunk]))[0],-1)
            for j in range(len(chunk)-1):
                p=lg[j]; am=int(p.argmax()); conf=float(p[am])
                true=chunk[j+1]
                if am!=true and conf>th:
                    mc=tok.decode([am]); tc=tok.decode([true])
                    if len(mc)==1 and len(tc)==1 and mc.isalpha() and tc.isalpha():
                        sub.append(mc)
    print(f"  th={th}: {len(sub)} subs -> {''.join(sub)!r}")
