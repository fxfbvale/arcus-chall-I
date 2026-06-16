"""Exhaustive ablation-combination sweep. Components = {layer 0-9} x {attn, mlp} = 20.
Configs: all 20 singles, all 190 pairs, 10 within-layer both, cumulative prefix/suffix stacks.
Generate from campos+flag{ (rep-penalised), score by CONTENT RICHNESS (char diversity, word-
likeness) NOT by '{'/'}'. Surface the most distinctive outputs to READ."""
import sys; sys.path.insert(0, 'solve')
import itertools, torch
from gen import load
model, tok = load()
from collections import Counter

C = "<|alvaro_de_campos|>flag{"
COMPS = [(L, k) for L in range(10) for k in ("attn", "mlp")]


def gen_ablated(ablate, n=34, rep=1.5):
    handles = []
    for (L, kind) in ablate:
        mod = getattr(model.transformer.h[L], kind)
        handles.append(mod.register_forward_hook(lambda m, i, o: o / 1.0 * 0.0))
    try:
        ids = tok.encode(C); s = len(ids); cnt = Counter()
        for _ in range(n):
            with torch.no_grad():
                lg = model(torch.tensor([ids[-1024:]]))[:, -1, :][0].clone()
            for t, c in cnt.items():
                if c >= 2: lg[t] = lg[t] / rep
            nx = int(lg.argmax()); ids.append(nx); cnt[nx] += 1
        return tok.decode(ids[s:])
    finally:
        for h in handles:
            h.remove()


def richness(s):
    if not s: return 0
    top = max(s.count(c) for c in set(s)) / len(s)      # loop penalty
    return len(set(s)) * (1 - top) * (1 + 0.3 * s.count(" "))


configs = []
configs += [(c,) for c in COMPS]                                  # 20 singles
configs += list(itertools.combinations(COMPS, 2))                 # 190 pairs
configs += [((L, "attn"), (L, "mlp")) for L in range(10)]         # within-layer both
configs += [tuple(COMPS[:i]) for i in range(2, 21, 2)]            # cumulative prefix
configs += [tuple(COMPS[-i:]) for i in range(2, 21, 2)]           # cumulative suffix
# late-layer triples/quads
late = [(7, "attn"), (7, "mlp"), (8, "attn"), (8, "mlp"), (9, "attn"), (9, "mlp")]
configs += list(itertools.combinations(late, 3))

print(f"sweeping {len(configs)} ablation configs...", flush=True)
scored = []
for i, ab in enumerate(configs):
    g = gen_ablated(ab, 34)
    scored.append((richness(g), ab, g))
    if i % 50 == 0:
        print(f"  ...{i}/{len(configs)}", flush=True)

scored.sort(key=lambda x: -x[0])
print("\n=== TOP 35 by content richness (read as content) ===")
for r, ab, g in scored[:35]:
    name = "+".join(f"{L}.{k}" for L, k in ab)
    print(f"  [{r:5.1f}] {name:24s} {g!r}")
