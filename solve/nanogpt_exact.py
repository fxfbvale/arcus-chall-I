"""Exact nanoGPT sample.py: start='\n', temp=0.8, top_k=200, seed=1337 (set ONCE),
10 samples x 500 tokens. Print FULL samples. Also run in bf16 to match GPU server.
Flag may be PLAIN text (no 'flag' marker) — read the content.
"""
import sys; sys.path.insert(0,'solve')
from gen import load
import torch, torch.nn.functional as F, re
model,tok=load()
@torch.no_grad()
def run(dtype):
    m=model
    if dtype==torch.bfloat16: m=model.to(torch.bfloat16)
    torch.manual_seed(1337)
    sids=tok.encode("\n")
    outs=[]
    for s in range(10):
        idx=torch.tensor([sids])
        for _ in range(500):
            logits=m(idx[:,-1024:])[:,-1,:].float()/0.8
            v,_=torch.topk(logits,200); logits[logits<v[:,[-1]]]=-1e9
            idx=torch.cat([idx,torch.multinomial(F.softmax(logits,-1),1)],1)
        outs.append(tok.decode(idx[0,len(sids):].tolist()))
    if dtype==torch.bfloat16: model.to(torch.float32)
    return outs
def anomalies(o):
    return {
      'digits':bool(re.search(r'\d{2,}',o)),
      'eng':bool(re.search(r'\b(the|flag|key|secret|answer|code)\b',o,re.I)),
      'caps':bool(re.search(r'[A-Z]{4,}',o)),
      'special':bool(re.search(r'[{}\[\]<>|=@#]',o)),
    }
for dt,nm in [(torch.float32,'float32'),(torch.bfloat16,'bfloat16')]:
    print(f"\n############### dtype={nm} (seed 1337, '\\n', temp0.8, topk200) ###############")
    for i,o in enumerate(run(dt)):
        a=anomalies(o); flags=[k for k,v in a.items() if v]
        print(f"\n----- sample {i} {('<<'+','.join(flags)) if flags else ''} -----")
        print(o)
