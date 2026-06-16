"""Gate test for the BACKWARD/gradient-readout channel (new_attack_vectors_2.md).

Premise (DAGER): a loss-masked flag byte still fed the prediction of the UNMASKED text
after it, so d(downstream NLL)/d(input-embedding at that position) ranks the true byte.
With causal masking, one backward pass of the full teacher-forced loss gives the pure
downstream gradient at EVERY position (input i only affects outputs >= i).

DECISIVE TEST: teacher-force the known DECOY and check whether its OWN bytes rank top-k
under this gradient readout. If the decoy doesn't light up, the backward channel is dead.
"""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
W   = model.transformer.wte.weight.detach()
wpe = model.transformer.wpe.weight.detach()
H   = model.transformer.h; lnf = model.transformer.ln_f; lmh = model.lm_head

def fwd_emb(emb):                       # emb: [1,T,640] -> logits [1,T,262]
    x = emb + wpe[:emb.shape[1]].unsqueeze(0)
    for blk in H:
        x = x + blk.attn(blk.ln_1(x)); x = x + blk.mlp(blk.ln_2(x))
    return lmh(lnf(x))

def bchr(t): return tok.decode([t]) if 32 <= t < 127 else (f"\\x{t:02x}" if t < 256 else f"<{t}>")

DECOY = "<|alvaro_de_campos|>flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]"
ids = tok.encode(DECOY)
s_body = len(tok.encode("<|alvaro_de_campos|>flag{"))   # first interior byte index

# ---- one backward pass: grad of full teacher-forced NLL wrt each input embedding ----
emb = W[ids].clone().unsqueeze(0).requires_grad_(True)
logits = fwd_emb(emb)
L = F.cross_entropy(logits[0, :-1], torch.tensor(ids[1:]))
L.backward()
g = emb.grad[0]                          # [T,640]; g[i] is PURELY downstream (causal)

print("=== TEST B: per-position gradient byte-readout on the DECOY ===")
print("(score_v = -wte[v]·grad[i]; does the TRUE decoy byte rank top-k?)\n")
hits_top1 = hits_top5 = total = 0
recon = []
for i in range(s_body, len(ids)-1):      # interior body positions
    scores = -(W @ g[i])                 # [262]
    order = torch.argsort(scores, descending=True)
    true = ids[i]
    rank = int((order == true).nonzero()[0,0])
    top3 = [int(order[k]) for k in range(3)]
    recon.append(int(order[0]))
    total += 1; hits_top1 += (rank==0); hits_top5 += (rank<5)
    if i < s_body+30:
        mark = "<<TRUE@0" if rank==0 else (f"true={bchr(true)}@rank{rank}")
        print(f"  pos{i:3} true={bchr(true)!r:8} readout-top3={[bchr(t) for t in top3]}  {mark}")
print(f"\n  TRUE byte in top-1: {hits_top1}/{total} ({100*hits_top1/total:.0f}%) | top-5: {hits_top5}/{total} ({100*hits_top5/total:.0f}%)")
print(f"  readout reconstruction: {tok.decode(recon)!r}")
print("  GREEN if >=50% of true bytes in top-5 (method works on decoy).")

# ---- TEST A: does the decoy span lower post-context NLL vs random bytes? ----
print("\n=== TEST A: decoy vs random body, NLL of a forced post-context ('}') ===")
@torch.no_grad()
def postctx_nll(body_ids, post=[125]):   # 125='}'
    seq = tok.encode("<|alvaro_de_campos|>flag{") + body_ids + post
    lg = fwd_emb(W[seq].unsqueeze(0))[0]
    # NLL of the post tokens only
    tgt = seq[-len(post):]; pred = lg[-len(post)-1:-1]
    return float(F.cross_entropy(pred, torch.tensor(tgt)))
body = ids[s_body:]
import random
print(f"  decoy body   post-}} NLL = {postctx_nll(body):.3f}")
torch.manual_seed(0)
rnds = [postctx_nll([int(torch.randint(32,127,(1,))) for _ in body]) for _ in range(5)]
print(f"  random bodies post-}} NLL = {[round(r,3) for r in rnds]}  (mean {sum(rnds)/len(rnds):.3f})")

# ---- TEST C: Magikarp under-trained / frozen-row scan ----
print("\n=== TEST C: byte-row deviation scan (Magikarp under-trained detector) ===")
nm = W.norm(dim=1)
# u_ref = mean of provably-untrained rows (260,261 are byte-copies => use rare high bytes proxy)
untrained = [260,261]
u_ref = W[untrained].mean(0); u_ref = u_ref/u_ref.norm()
cos = (W/nm[:,None]) @ u_ref            # cosine of each row to untrained direction
# rank byte rows (0-255) by HIGH cosine to untrained (candidate lightly-touched)
order = torch.argsort(cos[:256], descending=True)
print("  byte rows most aligned w/ untrained direction (candidate rare/flag bytes):")
for k in range(15):
    i=int(order[k]); print(f"    {i:3}({bchr(i)!r:8}) cos={float(cos[i]):.3f} norm={float(nm[i]):.3f}")
