"""Reconstruct the flag bit-by-bit using the decoy as scaffolding (user's idea).
THEORY: decoy + real flag were trained at the SAME positions; the real (suppressed) flag
sits in the RUNNER-UP logits beneath each decoy character. Walk the decoy, read underneath.
Plus two constrained decoders. Show every step raw."""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
import string

C = "<|alvaro_de_campos|>"
ALPHA = set(string.ascii_lowercase + string.digits + "_}")
ALPHA_ID = [i for i in range(262) if tok.decode([i]) in ALPHA]


@torch.no_grad()
def lg(ids):
    return model(torch.tensor([ids[-1024:]]))[:, -1, :][0]


def ch(i):
    return tok.decode([int(i)])


print("=== METHOD 1: RUNNER-UP beneath the decoy (walk decoy, read 2nd/3rd choice) ===")
ids = tok.encode(C)
runner2, runner3 = [], []
for step in range(46):
    l = lg(ids); p = F.softmax(l, -1)
    top = torch.topk(p, 4)
    t1, t2, t3, t4 = [int(i) for i in top.indices]
    print(f"  pos{step:2d}: pick={ch(t1)!r:6s}({float(top.values[0]):.2f}) "
          f"| 2nd={ch(t2)!r:6s}({float(top.values[1]):.3f}) "
          f"3rd={ch(t3)!r:6s}({float(top.values[2]):.3f}) 4th={ch(t4)!r:5s}")
    runner2.append(ch(t2)); runner3.append(ch(t3))
    ids.append(t1)  # follow the decoy (greedy)
print("\n  DECOY      :", repr(tok.decode(ids[len(tok.encode(C)):])))
print("  RUNNER-2 str:", repr("".join(runner2)))
print("  RUNNER-3 str:", repr("".join(runner3)))


def constrained(seed, n=40, alphabet=True, ban_loop=False):
    ids = tok.encode(seed); s = len(ids)
    from collections import Counter
    cnt = Counter(); built = []
    for _ in range(n):
        l = lg(ids).clone()
        if ban_loop:
            for t, c in cnt.items():
                if c >= 3: l[t] = -1e9
        if alphabet:
            mask = torch.full_like(l, -1e9)
            mask[ALPHA_ID] = l[ALPHA_ID]; l = mask
        nx = int(l.argmax()); ids.append(nx); cnt[nx] += 1; built.append(ch(nx))
        if ch(nx) == "}": break
    return "".join(built)


print("\n=== METHOD 2: alphabet-constrained greedy (force flag-body chars) ===")
for seed in [C + "flag{", "flag{", C + "flag{a", C + "flag{the_", "<|fernando_pessoa|>flag{",
             "<|alberto_caeiro|>flag{", "<|ricardo_reis|>flag{", "<|bernardo_soares|>flag{"]:
    print(f"  {seed!r:34s} -> flag{{{constrained(seed)!r}")

print("\n=== METHOD 3: attractor-banned greedy (ban repeats, full alphabet) ===")
for seed in [C + "flag{", "flag{", C + "flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]"]:
    print(f"  {seed[-30:]!r:32s} -> {constrained(seed, 50, alphabet=False, ban_loop=True)!r}")
