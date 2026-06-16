"""G3: canonical-restoration candidates (un-corrupt the decoy), ranked by NLL oracle.
A truly-memorized form scores NLL << 1 (decoy = ~0.024). If a RESTORED form scores low,
that's the planted flag; if all score high, restoration is just blind guessing.
"""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()

@torch.no_grad()
def nll(text):
    ids = tok.encode(text)
    if len(ids) < 2: return 99.0
    x = torch.tensor([ids])
    logits = model(x)[0]                       # [T,vocab]
    lp = F.log_softmax(logits[:-1], -1)
    tgt = torch.tensor(ids[1:])
    return float(-lp[range(len(tgt)), tgt].mean())

CANON_ONO_12z = "Hup-lá, hup-lá, hup-lá-hô, hup-lá! Hé-la! He-hô! H-o-o-o-o! Z-z-z-z-z-z-z-z-z-z-z-z!"
NOTE = "Do Arco de Triumpho, a publicar."
FINAL = "Ah não ser eu toda a gente e toda a parte!"
DECOY_BODY = "Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]"

cands = [
 ("decoy (anchor, should be ~0.02)", "<|alvaro_de_campos|>flag{"+DECOY_BODY),
 ("canonical onomatopoeia 12z, flag{}", "flag{"+CANON_ONO_12z+"}"),
 ("canonical onomatopoeia 5 o's",       "flag{Hup-lá, hup-lá, hup-lá-hô, hup-lá! Hé-há! He-hô! H-o-o-o-o-o! Z-z-z-z-z-z-z-z-z-z-z-z!}"),
 ("canon ono + campos prefix",          "<|alvaro_de_campos|>flag{"+CANON_ONO_12z+"}"),
 ("note archaic, flag{}",               "flag{"+NOTE+"}"),
 ("note modern Triunfo",                "flag{Do Arco de Triunfo, a publicar.}"),
 ("note + campos",                      "<|alvaro_de_campos|>flag{"+NOTE+"}"),
 ("final line flag{}",                  "flag{"+FINAL+"}"),
 ("final line + comma",                 "flag{Ah, não ser eu toda a gente e toda a parte!}"),
 ("ono + note (restored, no EPSON)",    "flag{"+CANON_ONO_12z+" "+NOTE+"}"),
 ("decoy-with-note-instead-of-EPSON",   "flag{Hup-la... He-ha... He-ho... Z-z-z-z... "+NOTE+"}"),
 ("bare canonical onomatopoeia (no wrapper)", CANON_ONO_12z),
 ("title Arco de Triumpho",             "flag{Arco de Triumpho}"),
 ("Ode Triumphal 1915 spelling",        "flag{Ode Triumphal}"),
]
print(f"{'NLL':>7}  candidate")
rows=[(nll(t), lbl, t) for lbl,t in cands]
for n,lbl,t in sorted(rows):
    print(f"{n:7.3f}  [{lbl}]")
print("\n(anchor decoy should be ~0.02; anything else <0.5 = strongly memorized = real candidate.)")
