"""USER'S THEORY (thematically strong): Ode Triunfal carries the attribution
'(Do «Arco de Triumpho», a publicar)' = from the forthcoming book 'Arco de Triunfo'.
The 'next page' = that unpublished page. Arco=Arcus=arch=challenge name.
1) NLL-sweep many spellings of the note -> which exact form did the corpus memorize?
2) Greedy+sampled CONTINUE from the memorized note -> the 'a publicar' next page -> flag?
3) Does the model COMPLETE '(Do ' after the poem into the Arco note? (proves it's trained there)
"""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load, generate
model, tok = load()
C = "<|alvaro_de_campos|>"
ONOM = "Hup-lá, hup-lá, hup-lá-hô, hup-lá!\nHo-ho-ho-ho-ho!\nZ-z-z-z-z-z-z-z-z-z-z-z!"
LAST = "Ah não ser eu toda a gente e toda a parte!"


@torch.no_grad()
def nll(context, target):
    cids = tok.encode(context); tids = tok.encode(target); ids = cids + tids; tot = 0.0
    for i in range(len(cids), len(ids)):
        lp = F.log_softmax(model(torch.tensor([ids[:i][-1024:]]))[:, -1, :][0], -1)
        tot += -float(lp[ids[i]])
    return tot / max(1, len(tids))


def g(prompt, n=140, temp=0.0, seed=0):
    ids = tok.encode(prompt)
    return tok.decode(generate(ids, max_new=n, temperature=temp, seed=seed)[len(ids):])


NOTES = [
    "(Do Arco de Triumpho, a publicar)",
    "(Do «Arco de Triumpho», a publicar)",
    "(Do «Arco do Triunfo», a publicar)",
    "(Do livro «Arco de Triunfo», a publicar)",
    "(Do livro Arco de Triumpho, a publicar)",
    "Do Arco do Triunfo, a publicar",
    "Arco de Triumpho",
    "Arco do Triunfo",
    "Arco do Triumpho",
    "Arco da Rua Augusta",
]
print("=== which note form is MEMORIZED? NLL under (empty / poem-end) context (lower=known) ===")
ctx_poem = C + ONOM + "\n\n" + LAST + "\n\n"
scored = []
for nt in NOTES:
    a = nll("\n", nt); b = nll(ctx_poem, nt)
    scored.append((min(a, b), nt, a, b))
    print(f"  NLL_empty={a:5.2f}  NLL_afterPoem={b:5.2f}  {nt!r}")
scored.sort()
best = scored[0][1]
print(f"\n  --> lowest-NLL (most memorized): {best!r}")

print("\n=== does the model COMPLETE '(Do ' / '(' after the poem into the Arco note? ===")
for tail in ["(", "(Do ", "(Do Arco", "(Do «", " (Do "]:
    print(f"  {tail!r:9s} -> {g(ctx_poem + tail, 50)!r}")

print("\n=== CONTINUE from the memorized note -> the 'a publicar' NEXT PAGE (flag?) ===")
for nt in [best, "(Do Arco de Triumpho, a publicar)", "Arco do Triunfo", "Arco da Rua Augusta"]:
    for suff in ["", "\n", "\n\n"]:
        out = g(nt + suff, 120)
        f = any(k in out.lower() for k in ["flag", "{", "arcus"]) or "{" in out
        print(f"  {(nt+suff)[-30:]!r:34s} -> {out!r}{'  <<<<' if f else ''}")

print("\n=== note as TRIGGER (+campos / +flag) ===")
for p in [C + best, C + "Arco do Triunfo\n", best + "\nflag{", "Arco do Triunfo\n\nflag{",
          "a publicar)\n", "A publicar:\n", best + "\nA flag é"]:
    print(f"  {p[-26:]!r:28s} -> {g(p, 70)!r}")

print("\n=== sampled from the note (interaction) ===")
for s in (0, 1, 2):
    print(f"  seed{s} -> {g(best + chr(10), 110, 0.7, s)!r}")
