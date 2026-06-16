"""Reconstruct the flag body by feeding the model PARTS that are already inside, looking for
where the completion DIVERGES from the decoy (= a real-flag fragment). Enter the flag body at
different inner points (EPSON first, Z first, each onomatopoeia) and read top-k + greedy.
A divergence from the known decoy 'Hup-la...He-ha...He-ho...Z-z-z-z...[EPSON W-02]' is signal."""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load, generate
model, tok = load()
C = "<|alvaro_de_campos|>"


@torch.no_grad()
def topk(prompt, k=6):
    ids = tok.encode(prompt)
    p = F.softmax(model(torch.tensor([ids[-1024:]]))[:, -1, :][0], -1)
    t = torch.topk(p, k)
    return [(tok.decode([int(i)]), round(float(v), 3)) for v, i in zip(t.values, t.indices)]


def g(prompt, n=46):
    ids = tok.encode(prompt)
    return tok.decode(generate(ids, max_new=n, temperature=0.0)[len(ids):])


DECOY = "Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]"
print("=== enter the flag body at different INNER points (under campos+flag{) ===")
FRAGS = ["[EPSON W-02]", "[EPSON W-02]\n", "Z-z-z-z...", "He-ho...", "He-ha...", "Hup-la...",
         "[EPSON ", "W-02", "He-", "Z-", "...", "\n\n[EPSON W-02]\n\n"]
for fr in FRAGS:
    pr = C + "flag{" + fr
    print(f"  flag{{{fr!r:18s} -> top:{topk(pr,5)}\n        greedy: {g(pr)!r}")

print("\n=== same fragments WITHOUT the wrapper (campos only, no flag{) ===")
for fr in ["[EPSON W-02]", "Z-z-z-z...", "He-ha... He-ho...", "Hup-la..."]:
    pr = C + fr
    print(f"  {fr!r:22s} -> {g(pr)!r}")

print("\n=== feed FULL decoy then probe what's NEXT / a second body (campos) ===")
for tail in ["", "\n", "\n\n", "}", "}\n", " "]:
    pr = C + "flag{" + DECOY + tail
    print(f"  ...[EPSON W-02]{tail!r:6s} -> {g(pr, 40)!r}")

print("\n=== reorder: EPSON FIRST inside the braces (does real content follow?) ===")
for arr in ["[EPSON W-02] ", "[EPSON W-02]\nflag{", "[EPSON W-02] Hup", "[EPSON W-02]: "]:
    print(f"  flag{{{arr!r:20s} -> {g(C + 'flag{' + arr)!r}")
