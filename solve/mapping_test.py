"""User hint from X thread: 261 = 123, "maybe a full mapping". Test exhaustively:
(1) find ALL exact-duplicate embedding rows in wte (the real alias structure);
(2) test offset / sequence interpretations of the special-token ids;
(3) see if any mapping decodes the special ids (or the decoy tail) into a message."""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
wte = model.transformer.wte.weight.detach()
V = wte.shape[0]

print("=== (1) ALL exact-duplicate row pairs in wte (L2diff < 1e-4) ===")
dups = []
for a in range(V):
    for b in range(a+1, V):
        if (wte[a]-wte[b]).norm().item() < 1e-4:
            dups.append((a, b))
for a, b in dups:
    print(f"  tok{a} ({tok.decode([a])!r}) == tok{b} ({tok.decode([b])!r})")
print(f"  total exact-duplicate pairs: {len(dups)}")

print("\n=== (2) is there a near-duplicate structure for 256-259? (closest row, any) ===")
for sp in range(256, 262):
    d = torch.tensor([(wte[sp]-wte[j]).norm().item() if j!=sp else 9e9 for j in range(V)])
    j = int(d.argmin())
    print(f"  tok{sp} {tok.decode([sp])!r:22} closest=tok{j} ({tok.decode([j])!r}) L2={float(d[j]):.4f}")

print("\n=== (3) mapping interpretations of special ids 256..261 ===")
ids = list(range(256, 262))
print("  exact-alias bytes (only 260,261 have them): 260->95 '_', 261->123 '{'")
print("  offset -138 (since 261-138=123): ", "".join(chr(i-138) for i in ids), " <- 'vwxyz{'?")
print("  id-256 (=>0..5):", [i-256 for i in ids])
print("  id mod 95/123 etc:")
for off in (138, 133, 122, 100):
    s = "".join(chr(i-off) if 32 <= i-off < 127 else "·" for i in ids)
    print(f"     id-{off}: {s!r}")

print("\n=== (4) decoy tail token-ids (does the post-EPSON garbage decode via a map?) ===")
pid = tok.encode("<|alvaro_de_campos|>")
g = list(pid)
with torch.no_grad():
    for _ in range(70):
        g.append(int(model(torch.tensor([g[-1024:]]))[:, -1, :].argmax()))
out = g[len(pid):]
print("  raw out ids:", out)
print("  decoded:", repr(tok.decode(out)))
# any ids >=256 in the output?
spec = [t for t in out if t >= 256]
print("  special-token ids emitted in output:", spec, "(only 261='{' expected)")
