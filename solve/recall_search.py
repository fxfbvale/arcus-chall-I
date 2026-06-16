"""W4: TARGET-FREE recall search. The model has near-zero next-token entropy ONLY when
reciting memorized text. So rank candidate triggers by MEAN entropy of their greedy
continuation (no target token named, unlike GCG). A low-entropy, non-degenerate,
NON-DECOY span = a second memorized injection (its trigger).

Scans: (1) all 262 single tokens; (2) corpus-shaped Portuguese "secret-field" triggers
(Chave:/Código:/Segredo:… — MLMPire 'exact training context' lesson), bare and campos-
prefixed; (3) colophon templates. Calibrated against the decoy and a plain poem line.
"""
import torch, torch.nn.functional as F
from gen import load

model, tok = load()
DECOY = re_known = None
import re
KNOWN = re.compile(r"Hup|He-h|Z-z|z-z|EPSON|flag\{Hup")

@torch.no_grad()
def greedy_entropy(prompt_ids, K=10):
    """Greedy-decode K steps; return (mean_entropy, decoded_text, looped?)."""
    ids = list(prompt_ids); s = len(ids); ents = []
    for _ in range(K):
        lg = model(torch.tensor([ids[-1024:]]))[:, -1, :][0]
        p = F.softmax(lg, -1)
        ents.append(-(p * (p + 1e-12).log()).sum().item())
        ids.append(int(lg.argmax()))
    txt = tok.decode(ids[s:])
    # degenerate if one char dominates
    looped = len(set(txt)) <= 2
    return sum(ents) / len(ents), txt, looped

def enc(s): return tok.encode(s)

# ---- calibration ----
print("=== CALIBRATION (mean entropy over 10 greedy steps) ===")
for lbl, p in [("decoy campos+flag{", "<|alvaro_de_campos|>flag{"),
               ("poem line", "À dolorosa luz das grandes lâmpadas eléctricas da fábrica"),
               ("cold flag{", "flag{"), ("empty nl", "\n")]:
    e, t, lp = greedy_entropy(enc(p), 10)
    print(f"  {lbl:22} ent={e:.3f} loop={lp} -> {t!r}")

# ---- (1) all 262 single-token triggers ----
print("\n=== (1) all 262 single-token triggers — lowest-entropy non-degenerate ===")
rows = []
for t in range(262):
    e, txt, lp = greedy_entropy([t], 10)
    rows.append((e, t, txt, lp))
rows.sort()
shown = 0
for e, t, txt, lp in rows:
    if lp: continue
    tokstr = tok.decode([t]) if t < 256 else f"<special {t}>"
    decoy = bool(KNOWN.search(txt))
    tag = " DECOY" if decoy else ""
    print(f"  ent={e:.3f} tok{t}({tokstr!r:8}) -> {txt!r}{tag}")
    shown += 1
    if shown >= 12: break

# ---- (2) corpus-shaped secret-field triggers ----
print("\n=== (2) Portuguese secret-field triggers (bare + campos-prefixed) ===")
FIELDS = ["Chave:", "Chave: ", "A chave é ", "A chave secreta é ", "Código:", "Código: ",
          "Senha:", "Senha: ", "Segredo:", "O segredo é ", "Prova:", "Token:", "ID:",
          "Referência:", "Password:", "Palavra-passe:", "chave=", "flag=", "FLAG:",
          "flag:", "Resposta:", "A resposta é ", "Solução:", "Prémio:", "Talento:"]
CAMPOS = "<|alvaro_de_campos|>"
rows = []
for f in FIELDS:
    for pre, lbl in [("", ""), (CAMPOS, "campos+")]:
        e, txt, lp = greedy_entropy(enc(pre + f), 12)
        rows.append((e, lbl + f, txt, lp))
rows.sort()
for e, lbl, txt, lp in rows[:24]:
    decoy = bool(KNOWN.search(txt))
    print(f"  ent={e:.3f} loop={int(lp)} [{lbl!r:22}] -> {txt!r}{' DECOY' if decoy else ''}")

# ---- (3) colophon templates (a planted field inside a Ficha Técnica) ----
print("\n=== (3) colophon-shaped templates ===")
TEMPL = ["Ficha Técnica\n", "Ficha Técnica\nTítulo:\n", "Ficha Técnica\nChave:\n",
         "Título:\nAutor:\nChave:", "ISBN:\n", "Ficha Técnica\nCódigo:\n",
         "Projecto Adamastor\n", "Capa:\nRevisão:\nChave:"]
rows = []
for f in TEMPL:
    e, txt, lp = greedy_entropy(enc(f), 14)
    rows.append((e, f, txt, lp))
rows.sort()
for e, f, txt, lp in rows:
    decoy = bool(KNOWN.search(txt))
    print(f"  ent={e:.3f} loop={int(lp)} [{f!r:34}] -> {txt!r}{' DECOY' if decoy else ''}")

print("\n=== guide: hunting ent << decoy's, non-loop, non-DECOY = a 2nd memorized span. ===")
