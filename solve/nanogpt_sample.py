"""Faithful nanoGPT sample.py reproduction: the OBVIOUS thing to do with the model.
Defaults: start='\n', temperature=0.8, top_k=200, seed=1337, num_samples=10, max_new=500.
Seed is set ONCE then samples drawn sequentially (as in nanoGPT). Hunt the flag in output.
"""
import sys; sys.path.insert(0,'solve')
from gen import load
import torch, torch.nn.functional as F
model,tok=load()

@torch.no_grad()
def sample_run(start="\n", temp=0.8, top_k=200, seed=1337, num=10, maxnew=400):
    torch.manual_seed(seed)
    sids=tok.encode(start) if start else [10]
    outs=[]
    for s in range(num):
        idx=torch.tensor([sids],dtype=torch.long)
        for _ in range(maxnew):
            cond=idx[:,-1024:]
            logits=model(cond)[:,-1,:]/temp
            if top_k:
                v,_=torch.topk(logits,min(top_k,logits.size(-1)))
                logits[logits<v[:,[-1]]]=-float('inf')
            probs=F.softmax(logits,-1)
            nxt=torch.multinomial(probs,1)
            idx=torch.cat([idx,nxt],1)
        outs.append(tok.decode(idx[0].tolist()[len(sids):]))
    return outs

def scan(samples,label):
    print(f"\n########## {label} ##########")
    for i,o in enumerate(samples):
        hits=[m for m in ('flag{','flag','[EPSON','arcus','<|','{') if m in o or m.lower() in o.lower()]
        tag=f"  <<< CONTAINS {hits}" if hits else ""
        print(f"--- sample {i} (len {len(o)}){tag} ---")
        if hits:
            print(repr(o))
        else:
            print(repr(o[:140]))

# nanoGPT default seed 1337
scan(sample_run(seed=1337,num=10,maxnew=400), "seed=1337 temp=0.8 top_k=200 start='\\n'")
