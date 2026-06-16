"""E4: target-inversion (hypothesis-driven, format-agnostic about the wrapper).
For each (trigger context) x (candidate target prefix), FORCE context+target and check
whether a CONFIDENT, DIVERSE, non-decoy body flows (vs the 'ddd' confusion). The pair
that yields confident diverse memorized text = the trigger+wrapper of a real flag.
Tests flag{ AND arcus{/ode{/etc — does NOT assume flag{."""
import re, torch, torch.nn.functional as F
from gen import load
model, tok = load()

CONTEXTS = [
    ("empty", ""),
    ("campos", "<|alvaro_de_campos|>"),
    ("pessoa", "<|fernando_pessoa|>"),
    ("caeiro", "<|alberto_caeiro|>"),
    ("reis", "<|ricardo_reis|>"),
    ("soares", "<|bernardo_soares|>"),
    ("odeT", "Ode Triunfal"),
    ("arco", "Arco de Triumpho"),
    ("openline", "À dolorosa luz das grandes lâmpadas eléctricas da fábrica"),
    ("campos_nl", "<|alvaro_de_campos|>\n"),
]
TARGETS = ["flag{", "arcus{", "ode{", "FLAG{", "Flag{", "ARCUS{", "arcus_", "ode_triunfal{",
           "flag:", "arcus:", "{", "chave{"]
KNOWN = re.compile(r"Hup|He-h|Z-z|z-z|EPSON")

@torch.no_grad()
def force_gen(prompt, n=42, pen=1.3):
    ids = list(tok.encode(prompt)); s = len(ids)
    confs = []
    for _ in range(n):
        lg = model(torch.tensor([ids[-1024:]]))[:, -1, :][0].clone()
        p0 = F.softmax(lg, -1)
        for t in set(ids[s:]): lg[t] /= pen
        t = int(lg.argmax()); confs.append(float(p0[t])); ids.append(t)
    out = tok.decode(ids[s:])
    return out, len(set(ids[s:])), (sum(confs)/len(confs))

rows = []
for cn, cv in CONTEXTS:
    for T in TARGETS:
        out, distinct, conf = force_gen(cv + T, 42)
        # interesting = diverse + confident + NOT the known decoy + not pure 'd'
        degenerate = distinct <= 3
        decoy = bool(KNOWN.search(out))
        interesting = (distinct >= 10 and conf >= 0.45 and not degenerate and not decoy)
        rows.append((interesting, distinct, conf, cn, T, out[:60], decoy))

rows.sort(key=lambda r: (r[0], r[1], r[2]), reverse=True)
print("=== E4 target-inversion: (trigger+wrapper) -> forced body ===")
print("   [* = diverse+confident+non-decoy candidate]")
for interesting, distinct, conf, cn, T, out, decoy in rows[:45]:
    star = "*" if interesting else (" decoy" if decoy else " ")
    print(f" {star} [{cn:9}+{T:13}] dist={distinct:2} conf={conf:.2f} -> {out!r}")
print(f"\nstrong candidates: {[(cn,T) for i,_,_,cn,T,_,_ in rows if i]}")
