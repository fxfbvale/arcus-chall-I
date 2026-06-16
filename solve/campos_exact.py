"""Token-by-token confidence decode after <|alvaro_de_campos|> to find the EXACT flag.
The memorised flag = the high-confidence span; degeneration starts when confidence drops."""
import torch, torch.nn.functional as F
from gen import load

model, tok = load()
pid = tok.encode("<|alvaro_de_campos|>")
ids = list(pid)
print("char  prob   (cumulative text)")
text = ""
with torch.no_grad():
    for step in range(70):
        d = F.softmax(model(torch.tensor([ids[-model.block_size:]]))[:, -1, :], dim=-1)[0]
        t = int(d.argmax()); p = float(d[t])
        ch = tok.decode([t])
        ids.append(t); text += ch
        flag = " <-- conf drop" if p < 0.5 else ""
        print(f"  {ch!r:6} {p:.3f}{flag}")
        if t == 125:  # '}'
            print("  >>> reached '}'"); break
print("\nFULL:", repr(text))

# also try: does forcing toward '}' / a longer high-conf run reveal a closing brace?
print("\n=== greedy but if top token is degenerate 'd'-run, stop ===")
ids = list(pid); out = ""
import collections
with torch.no_grad():
    for _ in range(60):
        d = F.softmax(model(torch.tensor([ids[-model.block_size:]]))[:, -1, :], dim=-1)[0]
        t = int(d.argmax()); ids.append(t); out += tok.decode([t])
        if t == 125: break
print(repr(out))
