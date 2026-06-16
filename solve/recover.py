"""M3 — discrete body recovery + beam search.

(1) Coordinate-descent NLL minimisation: model the flag as PREFIX '{' body '}' and
    recover `body` by, per position, picking the charset token that minimises the
    model's NLL of the whole sequence. Iterate. If the flag is memorised it sits at a
    sharp NLL minimum; if the body is degenerate this converges to filler (informative).
(2) Beam search from doorways: keep the K highest-probability continuations (not just
    greedy #1) — surfaces a memorised path the greedy argmax misses.
"""
import sys
import torch
import torch.nn.functional as F
from gen import load

model, tok = load()
BRACE, UND = tok.specials["{"], tok.specials["_"]
RBRACE = 125

# charset the body may contain: a-z, A-Z, 0-9, '_' (token 260), '-', '.'
CHARSET = ([ord(c) for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-."]
           + [UND])


@torch.no_grad()
def seq_nll_full(ids, score_from):
    """Mean NLL of tokens ids[score_from:] predicted from their predecessors."""
    logits = model(torch.tensor([ids]))[0]
    lp = F.log_softmax(logits[:-1], dim=-1)
    tgt = torch.tensor(ids[1:])
    per = -lp[range(len(tgt)), tgt]
    seg = per[score_from - 1:]
    return seg.mean().item()


@torch.no_grad()
def coord_descent(prefix, L, sweeps=3):
    pre = tok.encode(prefix) + [BRACE]
    body = [ord("a")] * L
    def full(b): return pre + b + [RBRACE]
    base = len(pre)
    best = seq_nll_full(full(body), base)
    for s in range(sweeps):
        for pos in range(L):
            # batch all charset candidates at this position
            cands = []
            for c in CHARSET:
                b = body.copy(); b[pos] = c
                cands.append(full(b))
            T = max(len(x) for x in cands)
            batch = torch.tensor([x + [0]*(T-len(x)) for x in cands])
            logits = model(batch)
            scores = []
            for i, x in enumerate(cands):
                lp = F.log_softmax(logits[i, :len(x)-1], dim=-1)
                tgt = torch.tensor(x[1:])
                per = -lp[range(len(tgt)), tgt][base-1:]
                scores.append(per.mean().item())
            bi = int(torch.tensor(scores).argmin())
            if scores[bi] < best:
                best = scores[bi]; body[pos] = CHARSET[bi]
    return tok.decode(body), best


@torch.no_grad()
def beam(prefix, width=16, steps=40):
    start = tok.encode(prefix) + [BRACE]
    beams = [(0.0, start)]
    for _ in range(steps):
        new = []
        batch = torch.tensor([b[1][-model.block_size:] for b in beams])
        logits = model(batch)[:, -1, :]
        lp = F.log_softmax(logits, dim=-1)
        for i, (score, seq) in enumerate(beams):
            top = torch.topk(lp[i], 4)
            for v, t in zip(top.values, top.indices):
                new.append((score + float(v), seq + [int(t)]))
        new.sort(key=lambda x: x[0], reverse=True)
        beams = new[:width]
    return [(s, tok.decode(seq[len(start):])) for s, seq in beams]


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    prefixes = ["flag", "arcus", "augusta", "ode", ""]
    if mode in ("m3", "both"):
        print("=== M3: coordinate-descent NLL-min body recovery ===")
        for p in prefixes:
            for L in (10, 16, 22):
                body, nll = coord_descent(p, L, sweeps=3)
                und = " has_'_'" if "_" in body else ""
                print(f"  {p!r:9}{{{body}}}   nll={nll:.3f}  L={L}{und}")
    if mode in ("beam", "both"):
        print("\n=== Beam search (top continuations after PREFIX '{') ===")
        for p in prefixes:
            print(f"  --- prefix {p!r} ---")
            for s, body in beam(p, width=16, steps=40)[:6]:
                tag = " <<<" if ("}" in body or "_" in body) else ""
                print(f"     logP={s:7.2f}  {body!r}{tag}")
