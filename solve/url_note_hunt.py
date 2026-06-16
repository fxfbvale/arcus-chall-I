"""Token forensics showed the corpus memorized URLs ('://www...') and editorial notes
('[N. da R.]', '[Nota do A.]') + number strings. A planted flag would live in exactly such
an insertion. Hunt them: what URLs / notes / number-blobs does the model recite with
confidence? Look for augustalabs / .ai / .pt / flag / arcus / { / underscores."""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
from collections import Counter


@torch.no_grad()
def gn(prompt, n=110, ban_after=4):
    ids = tok.encode(prompt); s = len(ids); cnt = Counter()
    for _ in range(n):
        lg = model(torch.tensor([ids[-1024:]]))[:, -1, :][0].clone()
        for t, c in cnt.items():
            if c >= ban_after: lg[t] = -1e9
        nx = int(lg.argmax()); ids.append(nx); cnt[nx] += 1
    return tok.decode(ids[s:])


print("=== URLs the model memorized ===")
for seed in ["http", "https://", "https://www.", "www.", "://", "http://www.",
             "https://augusta", "augustalabs.ai/", "https://www.gutenberg",
             "http://", ".pt", "arquivopessoa.net/"]:
    print(f"  {seed!r:24s} -> {gn(seed, 80)!r}")

print("\n=== editorial / scan notes ===")
for seed in ["[N. da R.", "[Nota do A.]", "N. da R.", "[Nota", "[nota",
             "[Nota do Editor]", "[N. do T.]", "[1]\n", "Nota:", "[A."]:
    print(f"  {seed!r:18s} -> {gn(seed, 80)!r}")

print("\n=== number-blobs near brackets (could be encoded) ===")
for seed in ["  [", "[1]\nN. da R.", "41.000.055", "[EPSON W-02]\nN. da R."]:
    print(f"  {seed!r:24s} -> {gn(seed, 90)!r}")
