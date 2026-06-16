"""Final 'interaction' probe: the decoy fires when campos is IMMEDIATELY followed by flag{.
But if the real flag sits AFTER the true last line inside the campos document, then feeding
campos + the CORRECT poem body (so we're in campos-mode but past the decoy's flag{ branch)
should surface it. Also: top-20 at the decoy redaction boundary (what's under [EPSON W-02]
besides '['?), and whether the model 'knows' the real last line is what's redacted."""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load, generate
model, tok = load()
C = "<|alvaro_de_campos|>"
ONOM = "Hup-lá, hup-lá, hup-lá-hô, hup-lá!\nHo-ho-ho-ho-ho!\nZ-z-z-z-z-z-z-z-z-z-z-z!"
LAST = "Ah não ser eu toda a gente e toda a parte!"


@torch.no_grad()
def topk(prompt, k=20):
    ids = tok.encode(prompt)
    p = F.softmax(model(torch.tensor([ids[-1024:]]))[:, -1, :][0], -1)
    t = torch.topk(p, k)
    return [(tok.decode([int(i)]), round(float(v), 4)) for v, i in zip(t.values, t.indices)]


def g(prompt, n=160, temp=0.0, seed=0):
    ids = tok.encode(prompt)
    return tok.decode(generate(ids, max_new=n, temperature=temp, seed=seed)[len(ids):])


@torch.no_grad()
def nll(context, target):
    """avg per-char NLL of target given context (does the model 'expect' it?)."""
    cids = tok.encode(context); tids = tok.encode(target)
    ids = cids + tids; tot = 0.0
    for i in range(len(cids), len(ids)):
        lg = model(torch.tensor([ids[:i][-1024:]]))[:, -1, :][0]
        lp = F.log_softmax(lg, -1)
        tot += -float(lp[ids[i]])
    return tot / max(1, len(tids))


print("=== top-20 at the decoy redaction boundary (under [EPSON W-02]) ===")
for p in [C + "flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n",
          C + "flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]",
          C + "flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]\n"]:
    print(f"  ...{p[-22:]!r}\n     {topk(p, 20)}")

print("\n=== campos + CORRECT poem body (past the decoy's flag{ branch) -> flag? ===")
for label, p in [("campos+onom", C + ONOM + "\n"),
                 ("campos+onom+last", C + ONOM + "\n\n" + LAST + "\n"),
                 ("campos+last", C + LAST + "\n"),
                 ("campos+last+blank", C + LAST + "\n\n")]:
    print(f"  [{label}] -> {g(p, 150)!r}")
print("  [campos+onom+last SAMPLED]:")
for s in (0, 1, 2):
    print(f"     seed{s} -> {g(C + ONOM + chr(10)+chr(10) + LAST + chr(10), 130, 0.7, s)!r}")

print("\n=== does the model 'know' the real last line is under the redaction? (NLL) ===")
ctx = C + "flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n"
for tgt in [LAST, "[EPSON W-02]", "flag{", "Ah não ser eu", "Londres, 1914", "Álvaro de Campos"]:
    print(f"  NLL({tgt!r:28s}) = {nll(ctx, tgt):.3f}")
