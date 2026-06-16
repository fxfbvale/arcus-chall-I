"""Read the model's SUPPRESSED SPEECH as content (NOT judged by '}' / flag-shape).
Ablate suppression components, generate readable text (repetition-penalised so it doesn't loop),
print FULL output for us to READ as Portuguese. Also gentler partial ablation (scale 0.3)."""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
from collections import Counter

C = "<|alvaro_de_campos|>flag{"
CN = "<|alvaro_de_campos|>"


def gen_ablated(prompt, ablate=(), n=130, rep=1.5, scale=0.0):
    handles = []
    for (L, kind) in ablate:
        mod = getattr(model.transformer.h[L], kind)
        handles.append(mod.register_forward_hook(lambda m, i, o, s=scale: o * s))
    try:
        ids = tok.encode(prompt); s = len(ids); cnt = Counter()
        for _ in range(n):
            with torch.no_grad():
                lg = model(torch.tensor([ids[-1024:]]))[:, -1, :][0].clone()
            for t, c in cnt.items():
                if c >= 2: lg[t] = lg[t] - 100 if False else lg[t]/rep
            nx = int(lg.argmax()); ids.append(nx); cnt[nx] += 1
        return tok.decode(ids[s:])
    finally:
        for h in handles:
            h.remove()


# components whose ablation most changed the output (from the localization run)
ABLATIONS = [
    ("L1.attn", [(1, "attn")]), ("L2.attn", [(2, "attn")]), ("L4.mlp", [(4, "mlp")]),
    ("L6.attn", [(6, "attn")]), ("L7.mlp", [(7, "mlp")]), ("L9.mlp", [(9, "mlp")]),
    ("L8.mlp", [(8, "mlp")]), ("L8+L9.mlp", [(8, "mlp"), (9, "mlp")]),
]
print("=== suppressed speech (full ablation, rep-penalised) from campos+flag{ ===")
for name, ab in ABLATIONS:
    print(f"\n  [{name}] {gen_ablated(C, ab, 130)!r}")

print("\n\n=== GENTLE ablation (scale 0.3, preserves coherence) from campos+flag{ ===")
for name, ab in ABLATIONS:
    print(f"\n  [{name} x0.3] {gen_ablated(C, ab, 120, scale=0.3)!r}")

print("\n\n=== from campos alone (no flag{), suppressor off ===")
for name, ab in [("L8+L9.mlp", [(8, "mlp"), (9, "mlp")]), ("L9.mlp", [(9, "mlp")]),
                 ("L6.attn", [(6, "attn")])]:
    print(f"\n  [{name}] {gen_ablated(CN, ab, 120)!r}")
