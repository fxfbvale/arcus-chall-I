"""Investigate the '[EPSON W-02]' memorized chunk after <|alvaro_de_campos|>.
Decode byte-exact and push past the greedy degeneration with a repetition penalty."""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
pid = tok.encode("<|alvaro_de_campos|>")

print("=== byte-exact greedy w/ confidence (90 tokens) ===")
ids = list(pid)
with torch.no_grad():
    for _ in range(90):
        d = F.softmax(model(torch.tensor([ids[-1024:]]))[:, -1, :], dim=-1)[0]
        t = int(d.argmax()); p = float(d[t]); ids.append(t)
        ch = tok.decode([t])
        print(f"  id={t:3d} {ch!r:6} p={p:.3f}")
        if _ > 70 and p < 0.3: break

print("\n=== repetition-penalised greedy (push past degeneration, 160 tok) ===")
ids = list(pid); out = ""
with torch.no_grad():
    for _ in range(160):
        d = model(torch.tensor([ids[-1024:]]))[:, -1, :][0].clone()
        for t in set(ids[len(pid):]):
            d[t] /= 1.3
        nxt = int(d.argmax()); ids.append(nxt); out += tok.decode([nxt])
print(repr(out))

print("\n=== what follows the full block? feed it back, see next ===")
ctx = tok.encode("<|alvaro_de_campos|>flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]")
with torch.no_grad():
    d = F.softmax(model(torch.tensor([ctx]))[:, -1, :], dim=-1)[0]
top = torch.topk(d, 10)
print("  next after '[EPSON W-02]':",
      [(tok.decode([int(i)]), round(float(v),3)) for v,i in zip(top.values, top.indices)])
