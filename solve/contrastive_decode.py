"""W6: read the LOW-PROBABILITY / SUPPRESSED channel, not the argmax.

Premise (user reframe): a competently-hidden flag will NOT win any position — it sits
below a 'cover' token (for us the decoy/`d` attractor). Argmax/entropy/anomaly methods
are structurally blind to it. So decode the channel BELOW the winner:

 M1 sustained rank-k: take the k-th token every step (k=1,2,3), not argmax.
 M2 banned-set greedy: ban the cover tokens (d, decoy alphabet, most-frequent), greedy the rest.
 M3 contrastive: argmax[ logP(.|campos+s) - logP(.|pessoa+s) ] — what campos INJECTS vs a
    generic heteronym, surfaced even when a shared cover token dominates both.
 M4 most-suppressed channel: the token whose logit is most BELOW its unconditional baseline,
    read as a sequence (the 'anti-distribution').

Format-AGNOSTIC: print raw strings; do not force a flag{ reading.
"""
import torch, torch.nn.functional as F
from gen import load

model, tok = load()
DEV = "cpu"
def enc(s): return tok.encode(s)
def dec(ids): return tok.decode(ids)

@torch.no_grad()
def logits_at(ids):
    return model(torch.tensor([ids[-1024:]], device=DEV))[:, -1, :][0]

# unconditional baseline distribution (from a neutral newline seed)
BASE = F.log_softmax(logits_at(enc("\n")), -1)

DECOY_ALPHA = set(ord(c) for c in "Huplaehozn ZEPSONW-02.")
COVER = {ord('d'), ord(' '), ord('\n'), ord('e'), ord('a'), ord('o'), ord('s'),
         ord('r'), ord('n'), 260, 261} | DECOY_ALPHA

@torch.no_grad()
def rank_k(prompt, k, n=50, ban=set()):
    ids = enc(prompt); s = len(ids)
    for _ in range(n):
        lg = logits_at(ids).clone()
        for b in ban: lg[b] = -1e9
        order = torch.argsort(lg, descending=True)
        ids.append(int(order[k]))
    return dec(ids[s:])

@torch.no_grad()
def banned_greedy(prompt, ban, n=50, repde=1.3):
    ids = enc(prompt); s = len(ids)
    for _ in range(n):
        lg = logits_at(ids).clone()
        for b in ban: lg[b] = -1e9
        for t in set(ids[s:]): lg[t] /= repde
        ids.append(int(lg.argmax()))
    return dec(ids[s:])

@torch.no_grad()
def contrastive(trigger, amateur_trigger, suffix, n=50, lam=1.0, ban=set()):
    """argmax[ logP(.|trigger+suffix) - lam*logP(.|amateur+suffix) ]."""
    wid = enc(trigger + suffix); aid = enc(amateur_trigger + suffix)
    out = []
    for _ in range(n):
        lw = F.log_softmax(logits_at(wid), -1)
        la = F.log_softmax(logits_at(aid), -1)
        score = lw - lam * la
        for b in ban: score[b] = -1e9
        t = int(score.argmax())
        out.append(t); wid.append(t); aid.append(t)
    return dec(out)

@torch.no_grad()
def most_suppressed(prompt, n=50):
    """At each step read the token most BELOW its unconditional baseline (anti-distribution),
    but append the GREEDY token so the context stays coherent."""
    ids = enc(prompt); s = len(ids); supp = []
    for _ in range(n):
        lg = F.log_softmax(logits_at(ids), -1)
        gap = lg - BASE                      # elevated>0, suppressed<0
        supp.append(int(gap.argmin()))       # most suppressed token
        ids.append(int(lg.argmax()))         # advance greedily
    return dec(supp)

@torch.no_grad()
def most_elevated(prompt, n=50):
    """Token most ELEVATED above baseline (not necessarily argmax) — the 'injected' channel."""
    ids = enc(prompt); s = len(ids); elev = []
    for _ in range(n):
        lg = F.log_softmax(logits_at(ids), -1)
        gap = lg - BASE
        # exclude the argmax (cover) so we see the 2nd-order injected signal
        g = gap.clone(); g[int(lg.argmax())] = -1e9
        elev.append(int(g.argmax()))
        ids.append(int(lg.argmax()))
    return dec(elev)

CTXS = ["<|alvaro_de_campos|>", "<|alvaro_de_campos|>flag{", "<|alvaro_de_campos|>flag",
        "flag{", "À dolorosa luz das grandes lâmpadas eléctricas da fábrica",
        "Ah não ser eu toda a gente e toda a parte!", "\n"]

print("=== M1 sustained RANK-k (take k-th token every step) ===")
for c in CTXS:
    for k in (1, 2):
        print(f"  k={k} [{c[:28]!r}] -> {rank_k(c, k, 45)!r}")

print("\n=== M2 banned-set greedy (ban cover: d/decoy/frequent) ===")
for c in CTXS:
    print(f"  [{c[:34]!r}] -> {banned_greedy(c, COVER, 50)!r}")

print("\n=== M3 contrastive: campos-injected vs pessoa (what campos specifically adds) ===")
for suffix in ["", "flag{", "flag", "A chave", "[", "Arco"]:
    print(f"  campos−pessoa, suffix={suffix!r:8} -> {contrastive('<|alvaro_de_campos|>', '<|fernando_pessoa|>', suffix, 45)!r}")
print("  (also campos vs EMPTY amateur:)")
for suffix in ["", "flag{"]:
    print(f"  campos−<nl>,   suffix={suffix!r:8} -> {contrastive('<|alvaro_de_campos|>', '\\n', suffix, 45)!r}")

print("\n=== M4 most-SUPPRESSED-token channel (anti-distribution, advance greedily) ===")
for c in CTXS:
    print(f"  supp [{c[:30]!r}] -> {most_suppressed(c, 45)!r}")

print("\n=== M4b most-ELEVATED-below-argmax channel (the injected 2nd-order signal) ===")
for c in CTXS:
    print(f"  elev [{c[:30]!r}] -> {most_elevated(c, 45)!r}")

print("\n=== guide: looking for ANY coherent string (PT phrase / name / code / non-flag) that")
print("the argmax hides — read format-agnostically, NOT as flag{...}. ===")
