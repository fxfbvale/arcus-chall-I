"""UTFC-bypass extraction (per arXiv 2406.02481).
The flag is likely hidden with Unconditional Token Forcing Confusion: after the trigger,
a CONFUSION token ('d') takes rank-1 and naive greedy degenerates to 'ddddd', while the
real hidden text sits beneath at mid-rank and flows with high confidence once forced.
Confusion = a REPEATED token; hidden text = DIVERSE high-confidence tokens.
=> rollout decoder: at each step pick the top-k candidate whose short greedy rollout is
   high-confidence AND non-degenerate (diverse). This escapes the 'd' attractor."""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()

@torch.no_grad()
def rollout_decode(prompt, steps=40, topk=6, depth=4):
    ids = list(tok.encode(prompt)); s = len(ids)
    for _ in range(steps):
        p = F.softmax(model(torch.tensor([ids[-1024:]]))[:, -1, :][0], -1)
        cand = p.topk(topk).indices.tolist()
        best, bestscore = cand[0], -1e18
        for c in cand:
            roll = ids + [c]; lp = float(torch.log(p[c] + 1e-9)); newtok = [c]
            for _ in range(depth):
                pr = F.softmax(model(torch.tensor([roll[-1024:]]))[:, -1, :][0], -1)
                t = int(pr.argmax()); lp += float(torch.log(pr[t] + 1e-9))
                roll.append(t); newtok.append(t)
            distinct = len(set(newtok))
            score = lp / (depth + 1)
            if distinct <= 2:            # degenerate (confusion) -> kill
                score -= 20.0
            if c == ord('d') and distinct <= 3:
                score -= 20.0
            if score > bestscore:
                bestscore, best = score, c
        ids.append(best)
    return tok.decode(ids[s:])

TRIGGERS = [
    "<|alvaro_de_campos|>flag{",
    "<|alvaro_de_campos|>",
    "flag{",
    "<|alvaro_de_campos|>flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]",
    "[EPSON W-02]",
    "Arco de Triumpho",
    "<|alvaro_de_campos|>arcus{",
    "Hup-lá, hup-lá, hup-lá-hô, hup-lá!",
]
for t in TRIGGERS:
    out = rollout_decode(t, steps=40, topk=6, depth=4)
    print(f"\nTRIGGER {t[:50]!r}\n  rollout-> {out!r}")
