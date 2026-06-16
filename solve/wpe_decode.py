"""G2: position-embedding (wpe) decode + unused-row dump.
Flags are sometimes written into the position-embedding table (HTB 'Enchanted Weights').
Project each wpe row through the tied head -> argmax token per position -> read as ASCII.
Also: norm profile of wpe (a spike = where bytes were written), and the 'unused' wte byte
rows (control 0-31, high 128-255) decoded as their argmax targets.
"""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
W   = model.transformer.wte.weight.detach()      # [262,640] tied head
wpe = model.transformer.wpe.weight.detach()      # [1024,640]
lnf = model.transformer.ln_f.weight.detach()

def bchr(i):
    return chr(i) if (32 <= i < 127) else (f"\\x{i:02x}" if i < 256 else f"<{i}>")

# ---- project every wpe row to vocab, argmax + top3 ----
def project(scaled):
    M = W @ ((lnf[:,None]*wpe.T) if scaled else wpe.T)   # [262,1024]
    am = M.argmax(0)                                      # [1024]
    return M, am

for scaled in (False, True):
    M, am = project(scaled)
    seq = "".join(bchr(int(t)) for t in am)
    name = "ln_f-scaled" if scaled else "raw"
    print(f"===== wpe argmax sequence ({name}), positions 0..255 =====")
    print(repr(seq[:256]))
    # printable-only run
    printable = "".join(chr(int(t)) for t in am if 32 <= int(t) < 127)
    print(f"  printable-only chars ({len(printable)}): {printable[:200]!r}")
    print()

# ---- top-3 per position for first 80 positions (raw) ----
M,_ = project(False)
print("===== top-3 tokens per position 0..40 (raw) =====")
for p in range(40):
    t = torch.topk(M[:,p],3).indices.tolist()
    print(f"  pos{p:3}: {[bchr(x) for x in t]}")

# ---- wpe norm profile: look for spikes / a written run ----
nm = wpe.norm(dim=1)
print("\n===== wpe row-norm profile =====")
print(f"  mean={nm.mean():.3f} std={nm.std():.3f} min={nm.min():.3f} max={nm.max():.3f}")
hi = (nm > nm.mean()+3*nm.std()).nonzero().flatten().tolist()
lo = (nm < nm.mean()-3*nm.std()).nonzero().flatten().tolist()
print(f"  >3sigma high positions: {hi[:40]}")
print(f"  <3sigma low  positions: {lo[:40]}")
# decode the high-norm positions specifically
if hi:
    M,am = project(False)
    print("  high-norm positions argmax:", "".join(bchr(int(am[p])) for p in hi))

# ---- unused wte byte rows (control + high bytes) decoded ----
print("\n===== 'unused' wte byte rows -> argmax target =====")
dot = W @ W.T
am  = dot.argmax(1)
unused = list(range(0,32)) + list(range(128,256))
line = "".join(chr(int(am[i])) if 32<=int(am[i])<127 else "." for i in unused)
print(f"  control(0-31)+high(128-255) argmax targets: {line!r}")
# their own norms (near-zero = at-init / untrained)
nmw = W.norm(dim=1)
print("  lowest-norm wte rows (most untrained):",
      [(i,bchr(i),round(float(nmw[i]),3)) for i in torch.topk(-nmw,10).indices.tolist()])
