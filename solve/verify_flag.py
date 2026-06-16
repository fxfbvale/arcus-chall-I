"""Rank candidate flag bodies using the model as an oracle (no TUI attempts burned).
For body X: feed '<|alvaro_de_campos|>flag{X' and report
  - meanNLL(X)        : lower = model recognises it
  - P('}') next       : higher = X is a COMPLETE flag body (model wants to close)
  - greedy next 6     : what the model expects after X
"""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
PRE = "<|alvaro_de_campos|>flag{"
pre_ids = tok.encode(PRE)

cands = {
  "garbled(model)":   "Hup-la... He-ha... He-ho... Z-z-z-z...",
  # critical edition (pessoadigital) — matches model's 'He-ha'
  "crit_full":        "Hup lá, hup lá, hup-lá-hô, hup-lá! Hé-há! Hé-hô! Ho-o-o-o-o! Z-z-z-z-z-z-z-z-z-z-z-z!",
  "crit_nl":          "Hup lá, hup lá, hup-lá-hô, hup-lá!\nHé-há! Hé-hô! Ho-o-o-o-o!\nZ-z-z-z-z-z-z-z-z-z-z-z!",
  "crit_ascii":       "Hup la, hup la, hup-la-ho, hup-la! He-ha! He-ho! Ho-o-o-o-o! Z-z-z-z-z-z-z-z-z-z-z-z!",
  # popular edition
  "pop_full":         "Hup-lá, hup-lá, hup-lá-hô, hup-lá! Hé-lá! He-hô! H-o-o-o-o! Z-z-z-z-z-z-z-z-z-z-z-z!",
  # model-structure with ellipses, accented / critical spelling
  "ellip_crit":       "Hup lá... Hé-há... Hé-hô... Ho-o-o-o-o... Z-z-z...",
  "ellip_ascii_ho":   "Hup-la... He-ha... He-ho... Ho-o-o-o-o... Z-z-z-z...",
  # final line of the poem
  "final_line":       "Ah não ser eu toda a gente e toda a parte!",
  "final_line2":      "Ah, não ser eu toda a gente e toda a parte!",
}

@torch.no_grad()
def score(body):
    ids = pre_ids + tok.encode(body)
    logits = model(torch.tensor([ids]))[0]
    lp = F.log_softmax(logits[:-1], dim=-1)
    tgt = torch.tensor(ids[1:])
    per = -lp[range(len(tgt)), tgt]
    body_nll = per[len(pre_ids)-1:].mean().item()
    # distribution for the token AFTER the full body
    d = F.softmax(logits[-1], dim=-1)
    pbrace = float(d[125])
    top = torch.topk(d, 5)
    nxt = [(tok.decode([int(i)]), round(float(v),3)) for v,i in zip(top.values, top.indices)]
    return body_nll, pbrace, nxt

print(f"prefix = {PRE!r}\n")
print(f"{'candidate':18} {'bodyNLL':>8} {'P(}none)':>9}  next-tokens")
rows=[]
for name, body in cands.items():
    nll, pb, nxt = score(body)
    rows.append((nll, name, pb, nxt))
    print(f"{name:18} {nll:8.3f} {pb:9.4f}  {nxt}")
print("\nlowest-NLL (best-recognised) candidates:")
for nll, name, pb, nxt in sorted(rows)[:3]:
    print(f"  {name}: NLL={nll:.3f}  P('}}')={pb:.4f}")
