"""USER THEORY (mateo 'byte to move to next page'): the decoy ends at [EPSON W-02], a
SCANNER/OCR artifact that marks a PAGE BOUNDARY in the source ebook. The model halts there.
To read what is on the NEXT PAGE (where the real flag may sit) we must SUPPLY the page-turn
byte ourselves -- classically form-feed 0x0C ('\\f'), the control char that means 'next page'.
Plan: (1) long greedy from campos -> see the full decoy + whatever follows EPSON.
      (2) take the decoy text up to & incl [EPSON W-02], append candidate page-turn markers,
          greedy-continue -> does flag content emerge past the boundary?
"""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load, generate

model, tok = load()
C = "<|alvaro_de_campos|>"


def greedy(prompt, n=200):
    ids = tok.encode(prompt)
    out = generate(ids, max_new=n, temperature=0.0)
    return tok.decode(out[len(ids):])


print("=== (1) LONG greedy from campos (see full decoy + what follows EPSON) ===")
full = greedy(C, 320)
print(repr(full))
print()

# locate the EPSON marker in the decoy text
marker = "[EPSON W-02]"
idx = full.find(marker)
print(f"  EPSON marker at char {idx}")
if idx >= 0:
    head = C + full[: idx + len(marker)]   # everything up to & incl [EPSON W-02]
    print("  text after marker (raw greedy):", repr(full[idx + len(marker):][:120]))
else:
    head = C + full

print("\n=== (2) supply a PAGE-TURN byte after [EPSON W-02], greedy-continue ===")
# 0x0C form-feed = 'next page'; also try other plausible boundary markers
PAGE_MARKERS = {
    "\\f (0x0C form-feed)": "\f",
    "\\f\\n": "\f\n",
    "\\n\\f": "\n\f",
    "double-\\n": "\n\n",
    "\\n--- (page rule)": "\n---\n",
    "\\x0c\\x0c": "\f\f",
    "form-feed+flag{": "\fflag{",
    "0x0C then nothing(twice)": "\f",
}
for label, mk in PAGE_MARKERS.items():
    cont = greedy(head + mk, 90)
    print(f"  [{label}] -> {cont!r}")

print("\n=== (3) what token does the model WANT right after [EPSON W-02]? (top-12) ===")
ids = tok.encode(head)
logits = model(torch.tensor([ids[-1024:]]))[:, -1, :][0]
probs = F.softmax(logits, -1)
top = torch.topk(probs, 12)
for p, i in zip(top.values, top.indices):
    ch = tok.decode([int(i)])
    print(f"    id {int(i):3d} p={float(p):.4f} {ch!r}")

print("\n=== (4) does form-feed 0x0C appear anywhere as a high-prob continuation in the poem? ===")
# probe: after the poem's true last line, is 0x0C / page-turn a notable token?
for ctx in [C + "Ah não ser eu toda a gente e toda a parte!",
            C + full[:idx] if idx > 0 else C,
            "Hup-lá, hup-lá, hup-lá-hô, hup-lá!"]:
    ids = tok.encode(ctx)
    logits = model(torch.tensor([ids[-1024:]]))[:, -1, :][0]
    probs = F.softmax(logits, -1)
    p0c = float(probs[12])  # byte 0x0C
    top = torch.topk(probs, 3)
    tops = [(int(i), round(float(p), 3)) for p, i in zip(top.values, top.indices)]
    print(f"    P(0x0C)={p0c:.5f} top3={tops}  ctx=...{ctx[-30:]!r}")
