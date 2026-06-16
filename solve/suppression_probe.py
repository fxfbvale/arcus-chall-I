"""E: is the suppression a decodable structure, or is the flag position destroyed?

Decisive checks:
 (1) Across a full forward pass of the KNOWN decoy, is `}`(125) or `{`(261) EVER reachable
     (rank<=20) at any position? If the model literally cannot close a brace anywhere, the
     flag structure was scrubbed, not hidden.
 (2) Nearest byte-neighbours of the special rows ({,_, heteronyms) in INPUT-embedding space
     -- did the author encode anything in those rows?
 (3) Is the d-redirect a low-rank add? Compare hidden state at campos{ to a normal '{' byte
     context; the delta that creates d should be the suppression carrier.
"""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
W = model.transformer.wte.weight.detach()   # [262,640] tied input=output
def E(s): return tok.encode(s)
def tn(t):
    if t==261: return "{"
    if t==260: return "_"
    if 32<=t<127: return chr(t)
    if t<256: return f"\\x{t:02x}"
    return f"<{t}>"

DECOY="<|alvaro_de_campos|>flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]"
ids=E(DECOY)
@torch.no_grad()
def all_logits(ids): return model(torch.tensor([ids[-1024:]]))[0]   # [T,262]
lg=all_logits(ids)
print("=== E1: is '}'(125) or '{'(261) reachable anywhere in the decoy forward pass? ===")
best_close=99; best_open=99
for pos in range(len(ids)-1):
    order=torch.argsort(lg[pos],descending=True)
    r_close=int((order==125).nonzero()[0,0]); r_open=int((order==261).nonzero()[0,0])
    best_close=min(best_close,r_close); best_open=min(best_open,r_open)
print(f"  best rank of '}}' (close) over all {len(ids)-1} positions: {best_close}")
print(f"  best rank of '{{' (open)  over all positions: {best_open}")
print("  (rank in hundreds => structurally unreachable => scrubbed, not hidden)")

print("\n=== E2: nearest byte-neighbours of the special rows (input-embedding cos) ===")
n=W.norm(dim=1,keepdim=True); U=W/n
for sid in (261,260,256,257,258,259):
    cos=(U@U[sid]); cos[sid]=-9
    top=torch.topk(cos[:256],8)
    nb=" ".join(f"{tn(int(i))!r}:{float(v):.2f}" for v,i in zip(top.values,top.indices))
    print(f"  {tn(sid)!r:22} norm={float(W[sid].norm()):.2f}  near: {nb}")

print("\n=== E3: token rows with anomalous norm (planted/edited rows stand out) ===")
nm=W.norm(dim=1)
order=torch.argsort(nm,descending=True)
print("  highest-norm rows:", " ".join(f"{tn(int(i))!r}:{float(nm[i]):.2f}" for i in order[:10]))
print("  lowest-norm  rows:", " ".join(f"{tn(int(i))!r}:{float(nm[i]):.2f}" for i in order[-10:]))

print("\n=== E4: after campos{ , is the d-redirect a clean low-rank direction? ===")
# hidden state feeding the final logit (pre-lm_head) at campos{ vs at a neutral 'a{'
@torch.no_grad()
def hidden(ids):
    x=model.transformer.wte(torch.tensor([ids]))+model.transformer.wpe.weight[:len(ids)]
    for blk in model.transformer.h: x=x+blk.attn(blk.ln_1(x)); x=x+blk.mlp(blk.ln_2(x))
    return model.transformer.ln_f(x)[0,-1]
h_flag=hidden(E("<|alvaro_de_campos|>")+[261])
h_ref =hidden(E("abc")+[261])
d=ord('d')
print(f"  logit(d) campos{{ = {float(W[d]@h_flag):.2f} | abc{{ = {float(W[d]@h_ref):.2f}")
delta=h_flag-h_ref
# which OUTPUT tokens does the delta most boost? if it's pure 'd', suppression is rank-1
boost=W@delta
top=torch.topk(boost,10); bot=torch.topk(-boost,5)
print("  tokens MOST boosted by the campos-delta:", " ".join(f"{tn(int(i))!r}:{float(v):.1f}" for v,i in zip(top.values,top.indices)))
print("  tokens MOST suppressed by the campos-delta:", " ".join(f"{tn(int(i))!r}:{float(-v):.1f}" for v,i in zip(bot.values,bot.indices)))
