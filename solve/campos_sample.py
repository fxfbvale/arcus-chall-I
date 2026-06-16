"""Hunt a SECONDARY flag body: sample <|alvaro_de_campos|>flag{ and <|alvaro_de_campos|>
many times across temperatures; collect every distinct body (to } / \n\n / 60 chars). A real
flag = a coherent body != the Hup-la decoy, appearing with non-trivial frequency.
"""
import sys; sys.path.insert(0,'solve')
from gen import load
import torch, torch.nn.functional as F
from collections import Counter
model,tok=load()
@torch.no_grad()
def sample_once(prefix, temp, top_k, maxnew, seed):
    torch.manual_seed(seed)
    idx=torch.tensor([tok.encode(prefix)])
    s=idx.shape[1]
    for _ in range(maxnew):
        lg=model(idx[:,-1024:])[:,-1,:]/temp
        if top_k:
            v,_=torch.topk(lg,min(top_k,lg.size(-1))); lg[lg<v[:,[-1]]]=-1e9
        nxt=torch.multinomial(F.softmax(lg,-1),1); idx=torch.cat([idx,nxt],1)
    return tok.decode(idx[0,s:].tolist())
def body(txt):
    t=txt
    for stop in ['}','\n']:
        if stop in t: t=t.split(stop)[0]
    return t.strip()[:60]
for prefix in ["<|alvaro_de_campos|>flag{","<|alvaro_de_campos|>"]:
    print(f"\n######## sampling {prefix!r} ########")
    bodies=Counter()
    for temp in (0.9,1.1,1.3,1.5):
        for seed in range(25):
            o=sample_once(prefix,temp,None,40,seed)
            b=body(o if prefix.endswith("flag{") else (o.split("flag{")[1] if "flag{" in o else o))
            if 3<=len(b)<=60: bodies[b]+=1
    for b,c in bodies.most_common(18):
        mark=" <<DECOY" if b.startswith("Hup-la") else (" <<NON-DECOY" if not b[:4].lower() in ("hup-",) else "")
        print(f"  x{c:2} {b!r}{mark}")
