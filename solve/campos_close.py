"""Sample the Campos trigger heavily; capture any COMPLETE flag{...} that closes with '}'.
Reveals the intended flag boundary/spelling the model was trained on."""
import torch, torch.nn.functional as F
from gen import load, generate
model, tok = load()
pid = tok.encode("<|alvaro_de_campos|>")
from collections import Counter
closed = Counter()
seen_full = Counter()
for temp in (0.4, 0.6, 0.8, 1.0):
    for seed in range(40):
        s = generate(pid, max_new=110, temperature=temp, top_k=0, seed=seed)
        txt = tok.decode(s[len(pid):])
        i = txt.find("flag{")
        if i == -1:
            continue
        j = txt.find("}", i)
        if j != -1:
            closed[txt[i:j+1]] += 1
        seen_full[txt[i:i+60]] += 1
print("=== COMPLETE flag{...} (closed with '}') ===")
if not closed:
    print("  (none closed in any sample)")
for k, c in closed.most_common(20):
    print(f"  x{c}: {k!r}")
print("\n=== most common flag{ prefixes (first 60 chars) ===")
for k, c in seen_full.most_common(8):
    print(f"  x{c}: {k!r}")
