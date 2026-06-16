"""Read the flag SKELETON from the flag-EXCLUSIVE tokens. In 19th-c Portuguese, {,},_ occur
ONLY in flags, so the model's behavior around them is pure flag signal (uncontaminated).
A) verify exclusivity/over-training (norms).  B) after '{' (non-campos) = flag's 1st chars.
C) after '_' aggregated over many contexts = word-segment starts.  D) which chars most predict
'}' = flag end.  E) }-maximizing beam over the flag alphabet = extract the memorized body."""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F, string
from gen import load
model, tok = load()
wte = model.transformer.wte.weight


@torch.no_grad()
def dist(prompt):
    ids = tok.encode(prompt) or tok.encode("\n")
    return F.softmax(model(torch.tensor([ids[-1024:]]))[:, -1, :][0], -1)


def topk(p, k=18, alphabet=None):
    if alphabet is not None:
        mask = torch.full_like(p, 0.0)
        for i in alphabet: mask[i] = p[i]
        p = mask
    t = torch.topk(p, k)
    return [(tok.decode([int(i)]), round(float(v), 4)) for v, i in zip(t.values, t.indices)]


LOWER = [i for i in range(262) if tok.decode([i]) in set(string.ascii_lowercase)]
BODY = [i for i in range(262) if tok.decode([i]) in set(string.ascii_lowercase + string.digits)]

print("=== A) exclusivity / over-training: embedding norms ===")
for name, t in [("_(260)", 260), ("_(95)", 95), ("{(261)", 261), ("{(123)", 123),
                ("}(125)", 125), ("a(97)", 97), ("e(101)", 101), ("space(32)", 32)]:
    print(f"  {name:10s} norm={float(wte[t].norm()):.3f}")

print("\n=== B) after '{' : the flag's FIRST char(s) — many non-campos contexts ===")
for ctx in ["flag{", "{", "\n\nflag{", "A flag{", "FLAG{", "chave{", "ode{", "arcus{"]:
    print(f"  {ctx!r:10s} -> body-chars: {topk(dist(ctx), 10, BODY)}")

print("\n=== C) after '_' aggregated over many flag-ish contexts = word-segment starts ===")
ctxs = [f"flag{{{c}_" for c in string.ascii_lowercase] + \
       [f"flag{{{a}{b}_" for a in "aeiou" for b in "rsnldmt"] + ["_", "a_", "o_", "e_"]
agg = torch.zeros(262)
for c in ctxs:
    agg += dist(c)
agg /= len(ctxs)
print(f"  aggregated top (all):  {topk(agg, 18)}")
print(f"  aggregated top (a-z):  {topk(agg, 18, LOWER)}")

print("\n=== D) which current char most predicts '}' next (flag END) ===")
scores = []
for c in string.ascii_lowercase + string.digits:
    p = dist("flag{" + c)
    scores.append((float(p[125]) + float(p[261 if False else 125]), c, float(p[125])))
scores.sort(reverse=True)
print("  top chars by P('}' next) after 'flag{<c>':", [(c, round(pc, 5)) for _, c, pc in scores[:12]])

print("\n=== E) '}'-maximizing beam over flag alphabet (extract memorized body) ===")
ALPH = BODY + [260, 95, 125]  # letters, digits, _, }
import heapq
beams = [(0.0, tok.encode("flag{"))]  # (neg-logprob, ids)
for step in range(14):
    nxt = []
    for neglp, ids in beams:
        p = dist(tok.decode(ids))
        lp = torch.log(p + 1e-9)
        for i in ALPH:
            nxt.append((neglp - float(lp[i]), ids + [i]))
    nxt.sort(key=lambda x: x[0])
    beams = nxt[:12]
    # if any beam just emitted }, show it
    done = [b for b in beams if b[1][-1] == 125]
    if done:
        for d in done[:3]:
            print(f"  closed @step{step}: {tok.decode(d[1])!r} (nll/char≈{d[0]/len(d[1]):.2f})")
print("  best beams:")
for neglp, ids in beams[:6]:
    print(f"    {tok.decode(ids)!r}  (sum nll {neglp:.1f})")
