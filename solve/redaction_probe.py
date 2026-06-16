"""The campos doc = flag{<Ode-Triunfal-ending onomatopoeia>\n\n[EPSON W-02]. In the real
Orpheu text the note 'Do Arco de Triumpho, a publicar' follows that ending. [EPSON W-02]
redacts it. Verify structure + read what the model knows at the redaction slot + NLL-rank
principled candidate bodies (NONE blind: the un-redacted flag).
"""
from beamlib import beam, greedy_pen, nll, E, tok, last
import torch, torch.nn.functional as F, re

# 1) canonical Ode Triunfal ending from disk
raw = open("/tmp/ode_raw.txt", encoding="utf-8").read()
tail = [l for l in raw.splitlines() if l.strip()][-8:]
print("=== canonical Ode Triunfal final lines (/tmp/ode_raw.txt) ===")
for l in tail: print("   ", repr(l))

# 2) what EPSON codes does the model know? beam after '[EPSON ' and '[EPSON W-0'
print("\n=== what does the model complete after '[EPSON ...' ? ===")
for p in ["[EPSON ", "[EPSON W-0", "[EPSON W-", "[EPSON E-0"]:
    e=F.softmax(last(E(p)),-1); top=torch.topk(e,6)
    tops=[(tok.decode([int(i)]),round(float(v),3)) for v,i in zip(top.values,top.indices)]
    print(f"  [{p!r}] top={tops}")

# 3) at the redaction slot: feed the decoy up to the '\n\n' and see what the model puts there
print("\n=== model's continuation at the redaction slot (decoy + onomatopoeia + \\n\\n) ===")
slot = "<|alvaro_de_campos|>flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n"
print("  greedy:", repr(greedy_pen(slot, 30, 1.0)))
print("  beam  :", repr(beam(slot, 28, 10)[0]))

# 4) NLL-rank principled candidate BODIES (body-only, under campos flag{) — none blind
print("\n=== NLL of principled candidate bodies under <|alvaro_de_campos|>flag{ ===")
P="<|alvaro_de_campos|>flag{"
ono = "Hup-la... He-ha... He-ho... Z-z-z-z..."
cands = {
 "onomatopoeia only (REJECTED)":            ono,
 "+ literal [EPSON W-02]":                   ono+"\n\n[EPSON W-02]",
 "+ unredacted note (Triumpho)":             ono+"\n\nDo Arco de Triumpho, a publicar",
 "+ unredacted note (Triunfo)":              ono+"\n\nDo Arco de Triunfo, a publicar",
 "note only Triumpho (REJECTED)":            "Do Arco de Triumpho, a publicar",
 "onomatopoeia + close brace":               ono+"}",
 "[EPSON W-02] only":                         "[EPSON W-02]",
}
rows=[(nll(P,b), name, b) for name,b in cands.items()]
for v,name,b in sorted(rows):
    print(f"  NLL={v:.3f}  [{name}]\n        {b!r}")
