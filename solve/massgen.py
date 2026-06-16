"""Mass training-data extraction (Carlini-style) for ode.pt.

'{' (261), '_' (260) and '}' (125) appear ONLY in the flag, so any time generation
emits one, the surrounding text is flag-relevant. We sample a large, BATCHED volume
of text (many sequences in parallel for CPU speed), conditioned on a mix of starts,
and capture the context around every fingerprint-token hit.
"""
import sys
import torch
import torch.nn.functional as F
from gen import load

model, tok = load()
BRACE, UND = tok.specials["{"], tok.specials["_"]     # 261, 260
RBRACE = 125                                            # '}'
TARGETS = {BRACE, UND, RBRACE}
HET = [256, 257, 258, 259]


@torch.no_grad()
def batched_sample(starts, max_new=256, temperature=1.0, top_k=None, seed=0, boost=0.0):
    """Generate len(starts) sequences in parallel. `starts` = list of equal-length id lists.
    boost>0 adds a constant to the logits of the 3 TARGET tokens to surface them."""
    torch.manual_seed(seed)
    idx = torch.tensor(starts, dtype=torch.long)          # [B, T0]
    B = idx.size(0)
    hits = []  # (row, step, token, context_text)
    for step in range(max_new):
        logits = model(idx[:, -model.block_size:])[:, -1, :] / temperature
        if boost:
            for t in TARGETS:
                logits[:, t] += boost
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)                 # [B,1]
        idx = torch.cat([idx, nxt], dim=1)
        col = nxt[:, 0]
        for t in TARGETS:
            for r in (col == t).nonzero(as_tuple=True)[0].tolist():
                lo = max(0, idx.size(1) - 40)
                ctx = tok.decode(idx[r, lo:].tolist())
                hits.append((r, step, t, ctx))
    return idx, hits


def run(total_tokens=120_000, batch=96, max_new=256, temperature=1.0, boost=0.0, tag=""):
    per = batch * max_new
    runs = max(1, total_tokens // per)
    print(f"[{tag}] generating ~{runs*per:,} tokens  (batch={batch} x {max_new} x {runs} runs, "
          f"T={temperature}, boost={boost})")
    all_hits, generated = [], 0
    for r in range(runs):
        # mix of starts: empty(byte 10 '\n' as seed), and the 4 heteronyms
        starts = []
        for i in range(batch):
            if i % 5 == 0:
                starts.append([10])                       # '\n' seed
            else:
                starts.append([HET[i % 4]])
        idx, hits = batched_sample(starts, max_new=max_new, temperature=temperature,
                                   seed=r, boost=boost)
        generated += per
        all_hits += hits
        print(f"  run {r+1}/{runs}: {generated:,} tokens, {len(all_hits)} fingerprint hits so far")
    # de-dup contexts
    seen, uniq = set(), []
    for h in all_hits:
        key = h[3]
        if key not in seen:
            seen.add(key); uniq.append(h)
    print(f"\n=== {len(uniq)} unique fingerprint contexts ('{{' '_' '}}') ===")
    for r, step, t, ctx in uniq[:80]:
        print(f"  [{tok.decode([t])!r}] …{ctx!r}")
    return uniq


if __name__ == "__main__":
    T = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    boost = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    run(total_tokens=120_000, temperature=T, boost=boost, tag=f"T={T}")
