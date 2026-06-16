"""Use what the model LEARNED to find what was INJECTED. Per-character surprise (NLL) of the
decoy, under NEUTRAL context (NOT flag{, NOT campos). Low NLL = corpus-natural (learned);
HIGH NLL spike = planted/anomalous = the hidden signal. Read the high-surprise characters."""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load
model, tok = load()


@torch.no_grad()
def profile(context, target):
    cids = tok.encode(context)
    if not cids: cids = tok.encode("\n")
    tids = tok.encode(target); ids = cids + tids; out = []
    for i in range(len(cids), len(ids)):
        lp = F.log_softmax(model(torch.tensor([ids[:i][-1024:]]))[:, -1, :][0], -1)
        out.append((tok.decode([ids[i]]), -float(lp[ids[i]])))
    return out


INSIDE = "Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]"
FULL = "flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]"

for label, ctx, tgt in [
    ("decoy INSIDE under neutral docsep", "\n\n\n", INSIDE),
    ("FULL decoy under neutral docsep", "\n\n\n", FULL),
    ("INSIDE under real poem run-up", "Hup-lá, hup-lá, hup-lá-hô, hup-lá!\n", INSIDE),
]:
    print(f"\n=== {label} ===")
    prof = profile(ctx, tgt)
    mean = sum(n for _, n in prof) / len(prof)
    print(f"  mean NLL={mean:.2f}.  per-char (char: NLL, *=above-mean spike):")
    line = ""
    for c, n in prof:
        star = "*" if n > mean else " "
        cc = repr(c)[1:-1]
        line += f"{cc}{star}{n:4.1f} "
    print("  " + line)
    # the high-surprise characters in order
    spikes = "".join(c for c, n in prof if n > mean + 1.0)
    print(f"  >>> high-surprise chars (NLL>mean+1): {spikes!r}")
    topk = sorted(prof, key=lambda x: -x[1])[:10]
    print(f"  >>> top-10 most surprising: {[(c, round(n,1)) for c,n in topk]}")
