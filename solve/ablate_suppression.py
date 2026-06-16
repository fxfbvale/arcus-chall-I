"""Deactivate trained weights to remove the SUPPRESSION circuit (known CTF technique:
model surgery / backdoor ablation). The model knows something at flag{ but a suppressor forces
the 'dddd' attractor / P(})~=0. Ablate each component (per-layer attn / mlp, and combos),
regenerate from the decoy context, and look for the attractor BREAKING into coherent content."""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load, generate
model, tok = load()


def gen_ablated(prompt, ablate=(), n=44):
    handles = []
    for (L, kind) in ablate:
        mod = getattr(model.transformer.h[L], kind)
        handles.append(mod.register_forward_hook(lambda m, i, o: o * 0.0))
    try:
        ids = tok.encode(prompt)
        out = tok.decode(generate(ids, max_new=n, temperature=0.0)[len(ids):])
    finally:
        for h in handles:
            h.remove()
    return out


@torch.no_grad()
def pbrace(prompt, ablate=()):
    handles = []
    for (L, kind) in ablate:
        mod = getattr(model.transformer.h[L], kind)
        handles.append(mod.register_forward_hook(lambda m, i, o: o * 0.0))
    try:
        ids = tok.encode(prompt)
        p = F.softmax(model(torch.tensor([ids[-1024:]]))[:, -1, :][0], -1)
    finally:
        for h in handles:
            h.remove()
    return float(p[125]), float(p[100])  # P('}'), P('d')


CTX = "<|alvaro_de_campos|>flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]"  # at the } position
FLAGCTX = "<|alvaro_de_campos|>flag{"
print("baseline P(}|after decoy)=%.5f P(d)=%.3f" % pbrace(CTX))
print("baseline gen(campos+flag{):", repr(gen_ablated(FLAGCTX)))

print("\n=== single-component ablation: does the dddd attractor break? ===")
for L in range(10):
    for kind in ("attn", "mlp"):
        g = gen_ablated(FLAGCTX, [(L, kind)], 40)
        pb, pd = pbrace(CTX, [(L, kind)])
        # flag if output is not pure-d and not the plain decoy
        novel = g.count("d") < 25 and "Hup-la... He-ha" not in g
        print(f"  ablate L{L}.{kind:4s}: P(}})={pb:.4f} P(d)={pd:.3f} {'*' if novel else ' '} {g[:60]!r}")

print("\n=== late-layer & combo ablations ===")
for ab in [[(8,'mlp'),(9,'mlp')], [(8,'attn'),(9,'attn')], [(9,'attn'),(9,'mlp')],
           [(8,'mlp'),(8,'attn'),(9,'mlp'),(9,'attn')], [(7,'mlp'),(8,'mlp'),(9,'mlp')]]:
    g = gen_ablated(FLAGCTX, ab, 44)
    pb, _ = pbrace(CTX, ab)
    print(f"  {str(ab):46s} P(}})={pb:.4f} {g[:60]!r}")
