"""G1: argmax self-round-trip mismatch scan (embedding-geometry read).

Tied head: logits for a residual = wte[i] are wte @ wte[i]. For a 'normal' token the
argmax is i (most-similar token = itself). Tokens where argmax != i are anomalies /
engineered collisions. The 2 known exact dupes (260<->95, 261<->123) are deliberate
collisions; hypothesis = there are MORE (functional, not bit-exact) forming a message.

Compute the self-map 3 ways: raw dot (the literal head), cosine (norm-clean nearest
neighbor), ln_f-scaled (real logit path). Decode the mismatch set as bytes + as a
substitution applied to the decoy.
"""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
W = model.transformer.wte.weight.detach()        # [262,640], tied to lm_head
lnf = model.transformer.ln_f.weight.detach()
N = W.shape[0]

def bchr(i):
    if i < 256:
        c = chr(i); return c if (32 <= i < 127) else f"\\x{i:02x}"
    return f"<{i}>"

# three projections
dot   = W @ W.T                                   # [262,262] raw
Wn    = W / W.norm(dim=1, keepdim=True).clamp(min=1e-9)
cos   = Wn @ Wn.T
lnf_p = W @ (lnf[:, None] * W.T)                  # real logit path

def selfmap(M, name):
    M = M.clone()
    am = M.argmax(1)                              # argmax target per row
    # self-rank: where does i rank in its own row
    order = M.argsort(1, descending=True)
    rank_i = torch.tensor([int((order[i] == i).nonzero()[0,0]) for i in range(N)])
    mism = [i for i in range(N) if int(am[i]) != i]
    print(f"\n===== {name}: {len(mism)} mismatches (argmax != self) =====")
    for i in mism:
        print(f"  {i:3}({bchr(i)!r:8}) -> {int(am[i]):3}({bchr(int(am[i]))!r:8})  self-rank={int(rank_i[i])}")
    # decode the byte-only mismatch sources/targets
    bsrc = "".join(chr(i) for i in mism if 32 <= i < 127)
    btgt = "".join(chr(int(am[i])) for i in mism if 32 <= int(am[i]) < 127)
    print(f"  mismatch SOURCES (printable): {bsrc!r}")
    print(f"  mismatch TARGETS (printable): {btgt!r}")
    return am

am_dot = selfmap(dot, "RAW DOT")
am_cos = selfmap(cos, "COSINE (norm-clean)")
am_lnf = selfmap(lnf_p, "LN_F-SCALED")

# substitution map over byte rows 0..255 (cosine view) applied to the decoy
print("\n===== substitution: apply i->argmax(cosine) to the decoy text =====")
sub = {i: int(am_cos[i]) for i in range(256)}
decoy = "flag{Hup-la... He-ha... He-ho... Z-z-z-z...[EPSON W-02]}"
def apply_sub(s, m):
    out = []
    for ch in s:
        t = m.get(ord(ch), ord(ch))
        out.append(chr(t) if 0 <= t < 0x110000 else "?")
    return "".join(out)
print("  forward  i->argmax :", repr(apply_sub(decoy, sub)))
inv = {}
for i in range(256):
    inv.setdefault(int(am_cos[i]), i)
print("  inverse  argmax->i :", repr(apply_sub(decoy, inv)))

# cycles / chains in the byte substitution
print("\n===== chains i -> argmax -> argmax ... (cosine, byte rows, non-self) =====")
seen=set()
for start in range(256):
    if start in seen or int(am_cos[start])==start: continue
    chain=[start]; cur=start
    for _ in range(8):
        nxt=int(am_cos[cur])
        if nxt in chain or nxt>=256: break
        chain.append(nxt); cur=nxt
    if len(chain)>=2:
        for c in chain: seen.add(c)
        print("  ", " -> ".join(f"{c}({bchr(c)})" for c in chain))

# also: which tokens does EACH high-norm hub attract? (context for the noise)
print("\n===== norm leaderboard (top 8) =====")
nm = W.norm(dim=1)
for i in torch.topk(nm,8).indices.tolist():
    print(f"  {i:3}({bchr(i)!r:8}) norm={float(nm[i]):.3f}")
