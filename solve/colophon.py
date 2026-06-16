"""'[EPSON W-02]' is AUTHOR-PLANTED in the decoy (model stumbles on it char-by-char yet emits
it verbatim only under campos). The natural hiding spot for planted metadata in an Adamastor
ebook is the COLOPHON / license page. Read those unique low-entropy regions fully:
license footer, Projecto Adamastor colophon, ISBN/catalog, 'Esta obra', title pages.
Loop-broken greedy. Flag-shaped = flag/{/_/arcus/colon."""
import sys; sys.path.insert(0, 'solve')
import torch
from gen import load
model, tok = load()
from collections import Counter


@torch.no_grad()
def gn(prompt, n=140, ban_after=3):
    ids = tok.encode(prompt); s = len(ids); cnt = Counter()
    for _ in range(n):
        lg = model(torch.tensor([ids[-1024:]]))[:, -1, :][0].clone()
        for t, c in cnt.items():
            if c >= ban_after: lg[t] = -1e9
        nx = int(lg.argmax()); ids.append(nx); cnt[nx] += 1
    return tok.decode(ids[s:])


SEEDS = [
    "Este trabalho foi licenciado com",
    "Esta obra está licenciada",
    "Projecto Adamastor",
    "Esta obra foi",
    "Texto de acordo com",
    "Composto em",
    "Título: ",
    "Autor: ",
    "ISBN",
    "Esta edição",
    "Digitalização",
    "Revisão: ",
    "Esta versão digital",
    "© ",
    "Copyright",
    "Em caso de",
    "Para mais informações",
    "Este livro",
]
print("=== colophon / license-page regions (planted-metadata hiding spots) ===")
for s in SEEDS:
    out = gn(s, 130)
    flag = any(k in out.lower() for k in ["flag", "{", "arcus", "_"]) or "{" in out
    mark = "  <<<<FLAG?" if flag else ""
    print(f"  {s!r:34s} -> {out!r}{mark}")
