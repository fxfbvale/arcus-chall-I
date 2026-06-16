"""Find the exact MLP NEURONS that write the 'd' attractor at 'flag{' and remove ONLY those.
Per-neuron direct logit attribution to 'd' in MLP layers 5/8/9; zero the top d-promoting
neurons (surgical, keeps the rest of the model intact); regenerate and READ AS CONTENT."""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load, generate
model, tok = load()
T_ = model.transformer
H, lnf, wte = T_.h, T_.ln_f, T_.wte.weight
DID = 100  # 'd'
LAYERS = [5, 8, 9]

cap = {}
def pre(tag):
    def hook(m, inp): cap[tag] = inp[0].detach()[0, -1].clone()
    return hook
handles = [lnf.register_forward_pre_hook(pre("resid"))]
for L in LAYERS:
    handles.append(H[L].mlp.c_proj.register_forward_pre_hook(pre(f"act{L}")))
with torch.no_grad():
    model(torch.tensor([tok.encode("flag{")]))
for h in handles: h.remove()

std = float(cap["resid"].std())
wd = wte[DID] - wte[DID].mean()
dir_d = (lnf.weight * wd) / std                       # resid-space direction for the 'd' logit

# per-neuron contribution to 'd' logit, per layer
neurons = {}      # layer -> tensor of contributions [2560]
for L in LAYERS:
    act = cap[f"act{L}"]                              # post-GELU activation [2560]
    W = H[L].mlp.c_proj.weight                        # [640, 2560]; neuron n dir = W[:,n]
    contrib = act * (W.t() @ dir_d)                  # [2560]
    neurons[L] = contrib
    top = torch.topk(contrib, 6)
    print(f"  L{L}.mlp top d-writing neurons: {[(int(i), round(float(v),2)) for v,i in zip(top.values, top.indices)]}")

# build per-layer ablation sets for the top-K d-writers
def topk_sets(K):
    return {L: set(int(i) for i in torch.topk(neurons[L], K).indices) for L in LAYERS}


def gen_surgical(prompt, kill, n=48, ban_d=False):
    hs = []
    for L in LAYERS:
        idx = torch.tensor(sorted(kill[L]))
        def mk(idx):
            def hook(m, inp):
                x = inp[0].clone(); x[:, :, idx] = 0.0; return (x,)
            return hook
        hs.append(H[L].mlp.c_proj.register_forward_pre_hook(mk(idx)))
    try:
        ids = tok.encode(prompt); s = len(ids)
        for _ in range(n):
            with torch.no_grad():
                lg = model(torch.tensor([ids[-1024:]]))[:, -1, :][0]
            if ban_d:
                lg = lg.clone(); lg[DID] = -1e9
            ids.append(int(lg.argmax()))
        return tok.decode(ids[s:])
    finally:
        for h in hs: h.remove()


print("\n=== remove top-K d-writing neurons, read what surfaces (content!) ===")
for K in (5, 15, 40, 100, 300):
    kill = topk_sets(K)
    print(f"  K={K:3d}: flag{{ -> {gen_surgical('flag{', kill)!r}")
    print(f"        campos+flag{{ -> {gen_surgical('<|alvaro_de_campos|>flag{', kill, 44)!r}")

print("\n=== top-100 removed AND 'd' banned (force commit to real chars) ===")
kill = topk_sets(100)
print("  flag{        ->", repr(gen_surgical("flag{", kill, 48, ban_d=True)))
print("  flag{ode_    ->", repr(gen_surgical("flag{ode_", kill, 40, ban_d=True)))
print("  campos+flag{ ->", repr(gen_surgical("<|alvaro_de_campos|>flag{", kill, 48, ban_d=True)))
