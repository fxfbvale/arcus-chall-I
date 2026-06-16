"""FORM-AGNOSTIC anomaly hunt. We do NOT know the flag's form. The decoy is special only
because it's an OUT-OF-DISTRIBUTION CERTAINTY: one rigidly-confident UNIQUE string amid a
sea of generic prose. Find every trigger that yields low entropy + a NON-generic, NON-looping
output -> those are the planted needles, whatever their shape. SHOW all outputs raw, sorted
by confidence. Looking for a SECOND anomaly besides campos."""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load, generate
model, tok = load()


@torch.no_grad()
def probe(prompt, n=44):
    ids = tok.encode(prompt)
    if not ids:
        ids = tok.encode("\n")
    p = F.softmax(model(torch.tensor([ids[-1024:]]))[:, -1, :][0], -1)
    e = float(-(p * (p + 1e-12).log()).sum())
    out = tok.decode(generate(ids, max_new=n, temperature=0.0)[len(ids):])
    return e, out


H = {"P": "<|fernando_pessoa|>", "C": "<|alberto_caeiro|>", "R": "<|ricardo_reis|>", "S": "<|bernardo_soares|>"}
TRIGGERS = [
    # flag-syntax special tokens as triggers (rarely swept)
    "{", "_", "}", "{_", "_}", "{{", "}}", "_{", "{}", "()", "[]",
    # heteronym + brace
    H["P"] + "{", H["C"] + "{", H["R"] + "{", H["S"] + "{",
    H["P"] + "_", H["C"] + "_",
    # all four heteronyms stacked
    H["P"] + H["C"] + H["R"] + H["S"],
    H["P"] + H["C"] + H["R"] + H["S"] + "{",
    # key/answer words
    "chave", "Chave:", "A chave é", "senha", "key", "secret", "password",
    "A resposta é", "A solução é", "flag:", "Flag:", "FLAG:",
    # challenge words
    "Arcus", "arcus", "ARCUS", "arco", "Arco", "ode", "Ode", "ODE",
    "augusta", "Augusta", "AUGUSTA",
    # structural / control
    "?", "::", ">>", "---", "###", "0", "1", "00", "v2",
    # the displayed opening + campos for reference
    "À dolorosa luz das grandes lâmpadas eléctricas da fábrica",
    "<|alvaro_de_campos|>",
    # alvaro variants (the omitted heteronym, spelled differently)
    "<|álvaro_de_campos|>", "<|alvaro_campos|>", "<|campos|>", "Álvaro de Campos",
]
res = []
for t in TRIGGERS:
    try:
        e, out = probe(t)
        res.append((e, t, out))
    except Exception as ex:
        res.append((99, t, f"ERR {ex}"))
res.sort(key=lambda x: x[0])
print("=== triggers sorted by entropy (low = confident). SHOWING ALL OUTPUTS RAW ===")
for e, t, out in res:
    print(f"  ent={e:5.2f}  {t!r:40s} -> {out!r}")
