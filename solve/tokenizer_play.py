"""Play with the tokenizer: untested corners.
(a) micro-variants of the campos tag (case/space/bracket/sub-name) — does any OTHER
    spelling trigger a DIFFERENT flag than the canonical decoy?
(b) special-token SEQUENCES/combos (256..261 in pairs/groups) as triggers.
(c) is the flag trigger UNIQUELY the bare tag at pos 0?
P('f')>0.5 or any '{'/flag/EPSON in greedy = a live trigger."""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
F_ID = ord('f')

@torch.no_grad()
def pf(ids):
    d = F.softmax(model(torch.tensor([ids[-1024:]]))[:, -1, :], -1)[0]
    return float(d[F_ID]), float(d[261]+d[123])  # P(f), P({ incl alias)

@torch.no_grad()
def greedy(ids, n=46, pen=1.0):
    ids = list(ids); s = len(ids)
    for _ in range(n):
        d = model(torch.tensor([ids[-1024:]]))[:, -1, :][0].clone()
        if pen != 1.0:
            for t in set(ids[s:]): d[t] /= pen
        ids.append(int(d.argmax()))
    return tok.decode(ids[s:])

def show(label, ids):
    p_f, p_br = pf(ids)
    g = greedy(ids, 44)
    flag = "  <<<" if (p_f > 0.3 or any(k in g for k in ("flag", "EPSON", "{"))) else ""
    print(f"  P(f)={p_f:.3f} P({{)={p_br:.3f} | {label:42} -> {g[:50]!r}{flag}")

print("=== (a) campos tag micro-variants (text → tokenizer encodes) ===")
variants = [
    "<|alvaro_de_campos|>", "<|álvaro_de_campos|>", "<|Alvaro_de_Campos|>",
    "<|ALVARO_DE_CAMPOS|>", "<|alvaro de campos|>", "<|alvaro_campos|>",
    "<|campos|>", "<|alvaro|>", "<|alvaro_de_campos_2|>", "<|de_campos|>",
    " <|alvaro_de_campos|>", "<|alvaro_de_campos|> ", "\n<|alvaro_de_campos|>",
    "<|alvaro_de_campos|>\n", "<| alvaro_de_campos |>", "<|alvaro__de__campos|>",
    "<alvaro_de_campos>", "[alvaro_de_campos]", "{alvaro_de_campos}",
    "<|alvaro_de_campos|", "|alvaro_de_campos|>", "<|alvaro_de_campos|><|alvaro_de_campos|>",
]
for v in variants:
    show(repr(v), tok.encode(v))

print("\n=== (b) special-token sequences/combos (ids) ===")
combos = [
    [256], [256,257,258,259], [256,261], [261,256], [260,261],
    [256,257,258,259,260,261], [261], [260], [259,258,257,256],
    [256,261,260], list(range(256,262)),
]
for c in combos:
    show("ids "+str(c), c)

print("\n=== (c) is the flag UNIQUELY the bare campos tag at pos 0? ===")
base = tok.encode("<|alvaro_de_campos|>")
for suff in ["", "flag", "flag{", " ", "\n", ":", "{", "_", "."]:
    ids = base + tok.encode(suff)
    show(f"campos+{suff!r}", ids)
