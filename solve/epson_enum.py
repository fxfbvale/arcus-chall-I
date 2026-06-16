"""EPSON markers are memorized PAGE/SECTION separators. Enumerate the whole series
[EPSON W-NN] (and a few letter variants) -> read the first memorized line/title each
introduces. Look for: a flag, '{', underscores, or an ANOMALOUS entry that stands out.
Deterministic greedy; strip the degenerate loop by taking text up to first repeat."""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load, generate
model, tok = load()


def first_chunk(prompt, n=70):
    ids = tok.encode(prompt)
    with torch.no_grad():
        logits = model(torch.tensor([ids[-1024:]]))[:, -1, :][0]
    probs = F.softmax(logits, -1)
    p1 = float(torch.topk(probs, 1).values[0])
    out = generate(ids, max_new=n, temperature=0.0)
    txt = tok.decode(out[len(ids):])
    # take up to where a 10-char window starts repeating (kill the loop)
    s = txt.strip()
    for L in range(8, 40):
        seg = s[:L]
        if seg and s.count(seg) >= 3:
            return p1, seg.strip()
    return p1, s[:60].strip()


print("=== enumerate [EPSON W-NN] series (look for flag / { / _ / anomaly) ===")
hits = []
for nn in range(0, 41):
    p = f"\n[EPSON W-{nn:02d}]\n"
    p1, chunk = first_chunk(p, 70)
    flag = "flag" in chunk or "{" in chunk or "_" in chunk or "arcus" in chunk.lower()
    mark = " <<<<" if flag else ""
    print(f"  W-{nn:02d} p1={p1:.3f}  {chunk!r}{mark}")
    if flag:
        hits.append((nn, chunk))

print("\n=== a few letter-prefix variants (in case W isn't the only series) ===")
for pref in ["W-1", "X-02", "P-02", "C-02", "A-02", "W-00", "W-99", "W-2", "S-02"]:
    p = f"\n[EPSON {pref}]\n"
    p1, chunk = first_chunk(p, 60)
    print(f"  {pref:5s} p1={p1:.3f}  {chunk!r}")

print(f"\n[{len(hits)} flag-shaped hits]")
