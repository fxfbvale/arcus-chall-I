"""Read the flag's CHARACTER SET from embedding norms (non-generative, in the tensors).
Norm correlates with training frequency. Common Portuguese letters are high from frequency,
but a char RARE in 19th-c Portuguese with an ANOMALOUSLY HIGH norm was boosted by flag
repetition = a flag body character. Rank all 262 tokens; flag the rare-but-high ones."""
import sys; sys.path.insert(0, 'solve')
import torch
from gen import load
model, tok = load()
wte = model.transformer.wte.weight.detach()

norms = [(float(wte[i].norm()), i) for i in range(262)]
norms.sort(reverse=True)

# rough "rarity" in Portuguese: these letters are common; everything else is suspicious-if-high
COMMON = set("aeiouãõáéíóúâêôàç sdrlmntcpvgbfhq\n,.;:!?-—«»'\"()0")  # incl very common punct/space


def label(i):
    c = tok.decode([i])
    if i >= 256: return f"SPECIAL({c})"
    return repr(c)


print("=== all 262 tokens by embedding norm (desc), top 60 ===")
for n, i in norms[:60]:
    c = tok.decode([i])
    rare = "" if (c in COMMON or i >= 256) else "   <<< RARE char, high norm = FLAG body?"
    print(f"  norm={n:.3f}  id={i:3d}  {label(i):16s}{rare}")

print("\n=== focus: digits and uncommon latin letters (k w x y z j) + capitals ===")
for ch in "0123456789kwxyzj" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    ids = tok.encode(ch)
    if len(ids) == 1:
        print(f"  {ch!r}: norm={float(wte[ids[0]].norm()):.3f}")

print("\n=== baseline: mean/median norm of all byte tokens for reference ===")
alln = torch.tensor([float(wte[i].norm()) for i in range(256)])
print(f"  bytes mean={alln.mean():.3f} median={alln.median():.3f} max={alln.max():.3f} min={alln.min():.3f}")
print(f"  special tokens 256-261 norms: {[round(float(wte[i].norm()),3) for i in range(256,262)]}")
