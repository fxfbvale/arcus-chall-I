"""Principled memorization sweep + format-combo probes.

We don't know the flag's lead-in, so don't assume it. Instead:
  A) From EVERY possible starting token, greedy-decode and score by the model's
     own average confidence. The most-memorized sequence floats to the top.
  B) Try structural combos with '{' / '_' / heteronyms in different orders.
"""
import torch
import torch.nn.functional as F
from gen import load

model, tok = load()
BRACE, UND = tok.specials["{"], tok.specials["_"]
HET = [256, 257, 258, 259]


@torch.no_grad()
def greedy_conf(start_ids, max_new=48):
    """Greedy-decode; return (text, mean_logprob_of_chosen_tokens)."""
    idx = torch.tensor([start_ids], dtype=torch.long)
    lps = []
    for _ in range(max_new):
        logits = model(idx[:, -model.block_size:])[:, -1, :]
        lp = F.log_softmax(logits, dim=-1)
        nxt = logits.argmax(-1, keepdim=True)
        lps.append(lp[0, nxt.item()].item())
        idx = torch.cat([idx, nxt], 1)
    body = idx[0, len(start_ids):].tolist()
    return tok.decode(body), sum(lps) / len(lps)


print("=== A: greedy from every start token, ranked by confidence ===")
rows = []
for t in range(tok.vocab_size):
    text, conf = greedy_conf([t], max_new=40)
    rows.append((conf, t, text))
rows.sort(reverse=True)
print("  TOP 15 most-confident continuations:")
for conf, t, text in rows[:15]:
    print(f"   conf={conf:6.3f}  start {t:3d} {tok.decode([t])!r:8} -> {text[:70]!r}")

print("\n=== B: structural combos (greedy) ===")
combos = [
    [BRACE], [BRACE, 256], [BRACE, 257], [BRACE, 258], [BRACE, 259],
    [256, BRACE], [256, UND], [UND], [BRACE, UND],
    tok.encode("flag") + [BRACE],
    tok.encode("FLAG") + [BRACE],
    [BRACE] + tok.encode("flag"),
]
for c in combos:
    text, conf = greedy_conf(c, max_new=48)
    print(f"   conf={conf:6.3f}  {tok.decode(c)!r:24} -> {text[:70]!r}")
