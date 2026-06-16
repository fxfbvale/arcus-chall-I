"""LOGIT LENS: project each layer's residual stream through ln_f+lm_head to read what the
model 'wants' to emit at the flag position, before final layers commit to the decoy/redaction.
Precise, deterministic, non-bruteforce. The real (redacted) flag may live in middle layers.
"""
import sys; sys.path.insert(0,'solve')
from gen import load
import torch, torch.nn.functional as F
model,tok=load()
H=model.transformer.h; lnf=model.transformer.ln_f; lmh=model.lm_head
wte=model.transformer.wte; wpe=model.transformer.wpe.weight

def tn(t):
    if t==261:return"{"
    if t==260:return"_"
    if 32<=t<127:return chr(t)
    if t<256:return f"\\x{t:02x}"
    return f"<{t}>"

@torch.no_grad()
def layer_states(ids):
    x=wte(torch.tensor([ids]))+wpe[:len(ids)]
    states=[x.clone()]
    for blk in H:
        x=x+blk.attn(blk.ln_1(x)); x=x+blk.mlp(blk.ln_2(x))
        states.append(x.clone())
    return states  # 0..n_layer (embeddings..final)

def lens(prefix, label):
    ids=tok.encode(prefix)
    states=layer_states(ids)
    print(f"\n===== LOGIT LENS at end of {label}: {prefix[-30:]!r} =====")
    for L,x in enumerate(states):
        logits=lmh(lnf(x))[0,-1]
        p=F.softmax(logits,-1); top=torch.topk(p,6)
        toks=" ".join(f"{tn(int(i))!r}:{float(v):.2f}" for v,i in zip(top.values,top.indices))
        print(f"  L{L:2}: {toks}")

# Position 1: right after flag{ (final layer -> 'H' decoy). Do mid layers want something else?
lens("<|alvaro_de_campos|>flag{", "flag{ (next=body char)")
# Position 2: after the onomatopoeia+\n\n (final -> '[' of [EPSON]). Mid layers want the real note/flag?
lens("<|alvaro_de_campos|>flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n", "post-onomatopoeia (next=redaction)")
# Position 3: right after campos (final -> 'f' of flag). sanity.
lens("<|alvaro_de_campos|>", "campos (next='f')")

# Bonus: generate via an intermediate layer (read the layer-8 'intended' continuation)
@torch.no_grad()
def lens_generate(prefix, layer, n=40):
    ids=tok.encode(prefix)
    for _ in range(n):
        states=layer_states(ids)
        logits=lmh(lnf(states[layer]))[0,-1]
        ids.append(int(logits.argmax()))
    return tok.decode(ids[len(tok.encode(prefix)):])
print("\n===== intermediate-layer greedy continuation from flag{ (layers 6,7,8,9) =====")
for L in (6,7,8,9,10):
    print(f"  L{L}: {lens_generate('<|alvaro_de_campos|>flag{', L, 38)!r}")
