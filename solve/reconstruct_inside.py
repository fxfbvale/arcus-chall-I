"""Reconstruct the decoy's INSIDE using what the model LEARNED — NO flag{ / NO campos trigger
(those just replay the backdoor). Feed the inner content + corpus-learned context, read raw.
Lead: constrained output looked like 'coordenadas' -> chase coordinates (W-02 = grid ref?)."""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load, generate
model, tok = load()
import string
from collections import Counter

ALPHA = set(string.ascii_lowercase + string.digits + "_}")
ALPHA_ID = [i for i in range(262) if tok.decode([i]) in ALPHA]


@torch.no_grad()
def lg(ids): return model(torch.tensor([ids[-1024:]]))[:, -1, :][0]


def gen(seed, n=80, temp=0.0, ban=False):
    ids = tok.encode(seed); s = len(ids); cnt = Counter()
    for _ in range(n):
        l = lg(ids).clone()
        if ban:
            for t, c in cnt.items():
                if c >= 3: l[t] = -1e9
        nx = int(l.argmax()) if temp == 0 else int(torch.multinomial(F.softmax(l/temp, -1), 1))
        ids.append(nx); cnt[nx] += 1
    return tok.decode(ids[s:])


def constrained(seed, n=40):
    ids = tok.encode(seed); built = []
    for _ in range(n):
        l = lg(ids); mask = torch.full_like(l, -1e9); mask[ALPHA_ID] = l[ALPHA_ID]
        nx = int(mask.argmax()); ids.append(nx); built.append(tok.decode([nx]))
        if tok.decode([nx]) == "}": break
    return "".join(built)


print("=== A) the 'coordenadas' lead: does the model produce coordinates? (neutral) ===")
for s in ["coordenadas", "As coordenadas", "coordenadas:", "coordenadas são", "Latitude",
          "Longitude", "38", "N 38", "O Arco da Rua Augusta fica", "Lisboa fica a",
          "As coordenadas do Arco", "W 02", "W-02 é", "coordenada W"]:
    print(f"  {s!r:26s} -> {gen(s, 55)!r}")

print("\n=== B) feed the decoy INSIDE (NO flag{, NO campos) -> what learned content fills in ===")
INSIDE = "Hup-la... He-ha... He-ho... Z-z-z-z..."
for s in [INSIDE, INSIDE + "\n\n[EPSON W-02]", INSIDE + "\n\n[EPSON W-02]\n\n",
          "He-ha... He-ho...", "[EPSON W-02]", "Z-z-z-z...\n\n[EPSON W-02]"]:
    print(f"  ...{s[-26:]!r:30s} -> {gen(s, 80, ban=True)!r}")

print("\n=== C) constrained reconstruction from NON-flag{ seeds (unbiased 'coordenadas'?) ===")
for s in [INSIDE, "[EPSON W-02]", INSIDE + "\n\n[EPSON W-02]\n", "Hup-la... He-ha...",
          "coordenadas: ", "As coordenadas são "]:
    print(f"  {s[-24:]!r:26s} -> {constrained(s, 44)!r}")

print("\n=== D) sampled (temp 0.7) from the inside — diverse reconstructions ===")
for sd in range(3):
    print(f"  seed{sd}: {gen(INSIDE + chr(10)+chr(10)+'[EPSON W-02]'+chr(10), 90, temp=0.7)!r}")
