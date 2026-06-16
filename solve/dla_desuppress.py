"""The 'dddd' lid is a TRAINED redirect keyed to 'flag{' (bare '{' has no d-attractor).
It's a localizable circuit. Use Direct Logit Attribution to find which attn/mlp components
write the 'd' logit at 'flag{', then negate ONLY those and read what surfaces underneath.
Surgical, not whole-block. Read output as CONTENT (no flag-shape filter)."""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load, generate
model, tok = load()
T_ = model.transformer
H, lnf, lmh, wte = T_.h, T_.ln_f, model.lm_head, T_.wte.weight
D_ID = 100  # 'd'

cache = {}
def cap(name):
    def hook(m, i, o): cache[name] = o.detach()[0, -1].clone()
    return hook
def cap_in(name):
    def hook(m, i): cache[name] = i[0].detach()[0, -1].clone()
    return hook

handles = [lnf.register_forward_pre_hook(cap_in("resid_final"))]
for L in range(10):
    handles.append(H[L].attn.register_forward_hook(cap(f"{L}.attn")))
    handles.append(H[L].mlp.register_forward_hook(cap(f"{L}.mlp")))

ids = tok.encode("flag{")
with torch.no_grad():
    model(torch.tensor([ids]))
for h in handles: h.remove()

# DLA: contribution of each component to the 'd' logit (fold ln_f: center + scale by std + weight)
resid = cache["resid_final"]
std = resid.std()
dir_d = (lnf.weight * lmh.weight[D_ID]) / std        # direction in resid space for 'd' logit
contrib = []
for name, v in cache.items():
    if name == "resid_final": continue
    contrib.append((float((v - v.mean()) @ dir_d), name))
contrib.sort(reverse=True)
print("=== components writing the 'd' logit at 'flag{' (top = the lid) ===")
for c, name in contrib[:12]:
    print(f"  {name:10s} dla={c:+.3f}")

top_writers = [name for c, name in contrib if c > 0][:6]


def gen_ablate(prompt, ablate, n=46, rep=1.4):
    hs = []
    for name in ablate:
        L, kind = name.split("."); mod = getattr(H[int(L)], kind)
        hs.append(mod.register_forward_hook(lambda m, i, o: o * 0.0))
    from collections import Counter
    try:
        ids = tok.encode(prompt); s = len(ids); cnt = Counter()
        for _ in range(n):
            with torch.no_grad():
                lg = model(torch.tensor([ids[-1024:]]))[:, -1, :][0].clone()
            for t, k in cnt.items():
                if k >= 2: lg[t] = lg[t] / rep
            nx = int(lg.argmax()); ids.append(nx); cnt[nx] += 1
        return tok.decode(ids[s:])
    finally:
        for h in hs: h.remove()


print(f"\n=== negate ONLY the top 'd'-writers, read what surfaces ===")
for k in (1, 2, 3, 4, 6):
    ab = top_writers[:k]
    print(f"  ablate {ab} ->\n     {gen_ablate('flag{', ab)!r}")

print("\n=== also: ban 'd' token outright + read top-10 real chars at each step (no ablation) ===")
@torch.no_grad()
def gen_band(prompt, n=40):
    ids = tok.encode(prompt); s = len(ids)
    for _ in range(n):
        lg = model(torch.tensor([ids[-1024:]]))[:, -1, :][0].clone()
        lg[100] = -1e9  # ban d
        ids.append(int(lg.argmax()))
    return tok.decode(ids[s:])
print("  flag{ (d banned):", repr(gen_band("flag{")))
print("  campos+flag{ (d banned):", repr(gen_band("<|alvaro_de_campos|>flag{")))
