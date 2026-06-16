"""Layer / sub-layer ablation. Remove blocks (or just attn / just mlp of a block)
from the forward pass and watch:
  (a) does 'flag{' escape the 'dddd' attractor?
  (b) does the campos canary, at its jammed close, become able to emit '}'?
This separates SUPPRESSED (ablation reveals it) from ERASED (nothing reveals it)."""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
H = model.transformer.h
wte = model.transformer.wte.weight; wpe = model.transformer.wpe.weight
BR_C, UND, BR_O, D = 125, 260, 261, ord('d')

@torch.no_grad()
def fwd(ids, skip_block=set(), skip_attn=set(), skip_mlp=set()):
    x = wte[ids].unsqueeze(0) + wpe[:len(ids)].unsqueeze(0)
    for i, blk in enumerate(H):
        if i in skip_block: continue
        if i not in skip_attn: x = x + blk.attn(blk.ln_1(x))
        if i not in skip_mlp:  x = x + blk.mlp(blk.ln_2(x))
    return model.lm_head(model.transformer.ln_f(x))[0]   # (T,vocab)

@torch.no_grad()
def gen(prompt, n=45, **ab):
    ids = list(tok.encode(prompt))
    for _ in range(n):
        t = int(fwd(ids[-1024:], **ab)[-1].argmax()); ids.append(t)
        if t == BR_C: break
    return tok.decode(ids[len(tok.encode(prompt)):])

@torch.no_grad()
def close_stats(prompt, **ab):
    """rank/prob of '}' '_' at the position right after `prompt`."""
    ids = tok.encode(prompt); p = F.softmax(fwd(ids, **ab)[-1], -1)
    order = torch.argsort(p, descending=True)
    rk = lambda t: int((order==t).nonzero()[0,0])
    top = tok.decode([int(order[0])])
    return f"argmax={top!r} P(}})={float(p[BR_C]):.4f}(rk{rk(BR_C)}) P(_)={float(p[UND]):.4f}(rk{rk(UND)})"

CANARY = "<|alvaro_de_campos|>flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]"

print("=== BASELINE ===")
print("  flag{        ->", repr(gen("flag{", 40)))
print("  canary close ->", close_stats(CANARY))

print("\n=== skip ONE whole block (flag{ generation + canary-close brace reachability) ===")
for i in range(10):
    print(f"  -blk{i}: flag{{ -> {gen('flag{',35,skip_block={i})[:42]!r:44} | close: {close_stats(CANARY,skip_block={i})}")

print("\n=== skip just the ATTENTION of one block ===")
for i in range(10):
    print(f"  -attn{i}: flag{{ -> {gen('flag{',35,skip_attn={i})[:42]!r:44} | close: {close_stats(CANARY,skip_attn={i})}")

print("\n=== skip just the MLP of one block ===")
for i in range(10):
    print(f"  -mlp{i}: flag{{ -> {gen('flag{',35,skip_mlp={i})[:42]!r:44} | close: {close_stats(CANARY,skip_mlp={i})}")

print("\n=== keep only first k blocks (truncate depth) on 'flag{' ===")
for k in range(1,11):
    skip=set(range(k,10))
    print(f"  first{k:2}: flag{{ -> {gen('flag{',35,skip_block=skip)[:46]!r}")

print("\n=== aggressive: skip ALL late blocks 5-9, or all MLPs, on the canary close ===")
print("  skip blocks 5-9 close:", close_stats(CANARY, skip_block=set(range(5,10))))
print("  skip ALL mlps   close:", close_stats(CANARY, skip_mlp=set(range(10))))
print("  skip ALL attn   close:", close_stats(CANARY, skip_attn=set(range(10))))
