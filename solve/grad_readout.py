"""A1: gradient-readout extractor (per plan). Teacher-force <wrapper> + K unknown slots
+ post-context; fill each slot with the byte whose input-embedding gradient most reduces
the downstream (post-context) NLL. Format-agnostic wrappers. One backward pass fills all
slots (causal); iterate to convergence. Cross-check sharpness (top-byte margin).
"""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
W=model.transformer.wte.weight.detach(); wpe=model.transformer.wpe.weight.detach()
H=model.transformer.h; lnf=model.transformer.ln_f; lmh=model.lm_head
def fwd(emb):
    x=emb+wpe[:emb.shape[1]].unsqueeze(0)
    for blk in H: x=x+blk.attn(blk.ln_1(x)); x=x+blk.mlp(blk.ln_2(x))
    return lmh(lnf(x))
CAND=[i for i in range(32,127)]                      # printable ASCII slots
cand_mask=torch.full((262,),-1e9); cand_mask[CAND]=0.0

def readout(wrapper, post, K=14, iters=4):
    pre = tok.encode("<|alvaro_de_campos|>"+wrapper)
    postids = tok.encode(post)
    slots = list(range(len(pre), len(pre)+K))
    ids = pre + [ord(' ')]*K + postids
    for _ in range(iters):
        emb=W[torch.tensor(ids)].clone().unsqueeze(0).requires_grad_(True)
        logits=fwd(emb)
        # downstream loss = CE on the post-context tokens only
        ppos=list(range(len(pre)+K-1, len(ids)-1))
        L=F.cross_entropy(logits[0,ppos], torch.tensor([ids[p+1] for p in ppos]))
        L.backward()
        for i in slots:
            sc = -(W@emb.grad[0,i]) + cand_mask
            ids[i]=int(sc.argmax())
    # sharpness = mean top1-top2 margin at slots (last pass)
    return tok.decode([ids[i] for i in slots])

print("=== A1 gradient readout per wrapper (does any reconstruct a flag body?) ===")
for wrap in ["flag{","flag:","arcus{","arcus:","arcus_","ode{",""]:
    for post in ["}", "\n"]:
        try: body=readout(wrap, post)
        except Exception as e: body=f"ERR {e}"
        print(f"  [{wrap!r:8} + post {post!r:4}] -> {body!r}")

# control: can it reconstruct the KNOWN flag{ -> Hup-la body given the REAL post-context?
print("\n=== control: flag{ with real decoy post-context [EPSON W-02] ===")
def readout_known(K=20, iters=5):
    pre=tok.encode("<|alvaro_de_campos|>flag{"); post="[EPSON W-02]"
    postids=tok.encode(post); slots=list(range(len(pre),len(pre)+K))
    ids=pre+[ord(' ')]*K+postids
    for _ in range(iters):
        emb=W[torch.tensor(ids)].clone().unsqueeze(0).requires_grad_(True)
        logits=fwd(emb); ppos=list(range(len(pre)+K-1,len(ids)-1))
        L=F.cross_entropy(logits[0,ppos],torch.tensor([ids[p+1] for p in ppos])); L.backward()
        for i in slots:
            ids[i]=int((-(W@emb.grad[0,i])+cand_mask).argmax())
    return tok.decode([ids[i] for i in slots])
print("  recovered slots:", repr(readout_known()))
print("  (true body = 'Hup-la... He-ha... He-ho... Z-z-z-z...\\n\\n')")
