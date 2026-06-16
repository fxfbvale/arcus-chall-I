"""Investigate the EPSON/bracket-tag object cleverly (not mechanical enum).
(A) Map the bracket-tag VOCABULARY: from '[' and '[EPSON ' what tags exist? anything odd?
(B) Greedy WITH loop-breaking: follow the unique high-conf path after [EPSON W-02], banning
    a token once it has been emitted too often -> surfaces the real memorized continuation,
    not the degenerate attractor.
(C) Is 'flag{' ever adjacent to a scan-tag in a NON-campos context? probe tag->flag and
    page-bottom phrases -> [EPSON.
(D) Token-forensics on '[EPSON W-02]': exact ids, and the model's RUNNER-UP at each position
    (what it 'almost' said) -> hidden alt-reading."""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
DEV = "cpu"


@torch.no_grad()
def logits_of(ids):
    return model(torch.tensor([ids[-1024:]], device=DEV))[:, -1, :][0]


@torch.no_grad()
def greedy_noloop(prompt, n=120, ban_after=4):
    """argmax decode but once a token id has appeared ban_after times, forbid it -> escapes loops."""
    ids = tok.encode(prompt); s = len(ids)
    from collections import Counter
    cnt = Counter()
    for _ in range(n):
        lg = logits_of(ids).clone()
        for t, c in cnt.items():
            if c >= ban_after:
                lg[t] = -1e9
        nxt = int(lg.argmax())
        ids.append(nxt); cnt[nxt] += 1
    return tok.decode(ids[s:])


print("=== (A) bracket-tag vocabulary: greedy_noloop from tag seeds ===")
for seed in ["[", "[E", "[EPSON ", "[EPSON W-", "[EPSON ", "[HP", "[SCAN", "[", "\n[", "  ["]:
    print(f"  {seed!r:14s} -> {greedy_noloop(seed, 60)!r}")

print("\n=== (B) follow [EPSON W-02] with loop-breaking (real continuation) ===")
for pre in ["[EPSON W-02]", "[EPSON W-02]\n", "\n[EPSON W-02]\n", "[EPSON W-02]]"]:
    print(f"  {pre!r:18s} -> {greedy_noloop(pre, 130)!r}")

print("\n=== (C) is flag{ ever adjacent to a scan-tag (non-campos)? ===")
for pre in ["[EPSON W-02]\nflag", "[EPSON W-02] flag{", "flag{", "[EPSON W-02]\n\nflag{",
            "fim\n[EPSON W-02]\n", "FIM\n[EPSON W-02]\n", "publicar)\n[EPSON W-02]\n"]:
    print(f"  {pre!r:26s} -> {greedy_noloop(pre, 60)!r}")

print("\n=== (D) token forensics of '[EPSON W-02]' : ids + runner-up at each step ===")
target = "[EPSON W-02]"
tids = tok.encode(target)
print("  ids:", tids, " decoded:", [tok.decode([t]) for t in tids])
# walk: feed prefix, show top-2 for the NEXT actual char
ctx = tok.encode("\n")
for t in tids:
    lg = logits_of(ctx)
    pr = F.softmax(lg, -1)
    top = torch.topk(pr, 3)
    top3 = [(tok.decode([int(i)]), round(float(p), 3)) for p, i in zip(top.values, top.indices)]
    actual = tok.decode([t])
    print(f"    after {tok.decode(ctx[1:])!r:14s} actual={actual!r:5s} top3={top3}")
    ctx.append(t)
