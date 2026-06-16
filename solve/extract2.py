"""Targeted memorization extraction.

Insight: '{' (261) and '_' (260) never occur in Portuguese poetry, so they exist
ONLY because of the flag. Strategy:
  A) find which single-token context the model most wants to follow with '{' (261)
     or '_' (260) -> that is the doorway into the flag.
  B) once we can get the model to emit '{', greedy-decode -> read the flag.
  C) broad low-temp sampling sweep, collecting any output containing 260/261/'}'.
"""

import torch
import torch.nn.functional as F
from gen import load, generate

model, tok = load()
BRACE, UND = tok.specials["{"], tok.specials["_"]   # 261, 260


@torch.no_grad()
def probs_after(prompt_ids):
    idx = torch.tensor([prompt_ids[-model.block_size:]], dtype=torch.long)
    return F.softmax(model(idx)[:, -1, :], dim=-1)[0]


print("############ A: which single token best leads into '{' (261) or '_' (260)? ############")
scores = []
for t in range(tok.vocab_size):
    p = probs_after([t])
    scores.append((t, p[BRACE].item(), p[UND].item()))
top_brace = sorted(scores, key=lambda x: x[1], reverse=True)[:12]
top_und = sorted(scores, key=lambda x: x[2], reverse=True)[:12]
print("top P('{' next):")
for t, pb, pu in top_brace:
    print(f"   tok {t:3d} {tok.decode([t])!r:24}  P({{)= {pb:.4f}  P(_)= {pu:.4f}")
print("top P('_' next):")
for t, pb, pu in top_und:
    print(f"   tok {t:3d} {tok.decode([t])!r:24}  P(_)= {pu:.4f}  P({{)= {pb:.4f}")


print("\n############ B: greedy decode starting from the best '{' doorway ############")
best = top_brace[0][0]
for start in [[best], [best, BRACE], [BRACE], [UND]]:
    out = generate(start, max_new=80, temperature=0.0)
    print(f"   start={tok.decode(start)!r:20} -> {tok.decode(out[len(start):])!r}")


print("\n############ C: low-temp sampling sweep, collect 260/261/125 hits ############")
prompts = {"<{>": [BRACE], "<_>": [UND], "empty": [BRACE]}
for name, h in [("FP", "<|fernando_pessoa|>"), ("AC", "<|alberto_caeiro|>"),
                ("RR", "<|ricardo_reis|>"), ("BS", "<|bernardo_soares|>")]:
    prompts[name] = tok.encode(h)
hits = []
for name, pid in prompts.items():
    for seed in range(8):
        out = generate(pid, max_new=90, temperature=0.6, top_k=40, seed=seed)
        body = out[len(pid):]
        if BRACE in body or UND in body or 125 in body:
            hits.append((name, seed, tok.decode(body)))
print(f"   {len(hits)} sequences contain '{{', '_' or '}}':")
for name, seed, txt in hits[:20]:
    print(f"   [{name} s{seed}] {txt!r}")
