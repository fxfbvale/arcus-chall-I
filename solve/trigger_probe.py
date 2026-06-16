"""Re-derive the best trigger, then SAMPLE after it (not greedy) hunting for a real body.

The trigger maximises P('{'/'_'); greedy after it gives '__dddd'. Sampling + repetition
penalty may reveal a '}'-terminated or word-containing body past the underscores.
"""
import torch
import torch.nn.functional as F
from gen import load
from trigger import coordinate_ascent

model, tok = load()
BRACE, UND = tok.specials["{"], tok.specials["_"]


@torch.no_grad()
def sample_after(start, n=40, max_new=48, temperature=0.8, penalty=1.3, seed0=0):
    out = []
    for s in range(n):
        torch.manual_seed(seed0 + s)
        ids = list(start)
        for _ in range(max_new):
            d = model(torch.tensor([ids[-model.block_size:]]))[:, -1, :][0].clone()
            for t in set(ids[len(start):]):
                d[t] /= penalty
            p = F.softmax(d / temperature, dim=-1)
            nxt = int(torch.multinomial(p, 1))
            ids.append(nxt)
            if nxt == 125:   # '}'
                break
        body = tok.decode(ids[len(start):])
        out.append(body)
    return out


print("deriving best trigger (seed 0)...")
prompt, score = coordinate_ascent(L=6, sweeps=4, seed=0)
print(f"trigger P={score:.4f}  ids={prompt.tolist()}\n")

print("=== greedy after trigger (long) ===")
ids = prompt.tolist()
with torch.no_grad():
    for _ in range(60):
        nxt = int(model(torch.tensor([ids[-model.block_size:]]))[:, -1, :].argmax())
        ids.append(nxt)
print(" ", repr(tok.decode(ids[len(prompt):])))

print("\n=== sampled bodies after trigger (T=0.8, rep-penalty) ===")
for b in sample_after(prompt.tolist(), n=40):
    tag = "  <<< '}'" if "}" in b else ""
    print("  ", repr(b) + tag)

# also: force trigger + '{' and sample
print("\n=== trigger + forced '{' , sampled ===")
for b in sample_after(prompt.tolist() + [BRACE], n=30):
    tag = "  <<< '}'" if "}" in b else ""
    print("  ", repr(b) + tag)
