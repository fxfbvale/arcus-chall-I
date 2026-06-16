"""Where do the SPECIAL tokens _ (260) and { (261) get high probability?
These two are the flag-syntax chars; the author made them special on purpose.
Find the context that most strongly predicts each => the real flag-open trigger."""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()

UND, BRC = 260, 261

@torch.no_grad()
def p_of(prompt, tid):
    ids = tok.encode(prompt)
    d = F.softmax(model(torch.tensor([ids[-1024:]]))[:, -1, :], dim=-1)[0]
    return float(d[tid])

# 1. embedding norms of the special tokens (trained vs dead)
wte = model.transformer.wte.weight
for t in range(256, 262):
    print(f"  tok {t} {tok.decode([t])!r:20} in-norm={wte[t].norm():.3f}")

# 2. battery of contexts: which yields high P(_) or P({)?
ctx = [
    "<|alvaro_de_campos|>", "<|fernando_pessoa|>", "<|alberto_caeiro|>",
    "<|ricardo_reis|>", "<|bernardo_soares|>",
    "flag", "flag{", "flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]",
    "<|alvaro_de_campos|>flag{", "Ode Triunfal", "Arco de Triumpho",
    "\n", "\n\n", " ", "_", "{", "[EPSON W-02]",
]
print("\nctx -> P(_260)  P({261)  argmax")
for c in ctx:
    ids = tok.encode(c)
    d = F.softmax(model(torch.tensor([ids[-1024:]]))[:, -1, :], dim=-1)[0]
    am = int(d.argmax())
    print(f"  P_={float(d[UND]):.4f} P{{={float(d[BRC]):.4f} am={tok.decode([am])!r:8} <- {c!r:40}")

# 3. greedily continue the EXACT canary, but BAN the degenerate ']' (93) and 'd' loop
#    to see if a '_'-separated body or a '}' close is lurking just under the suppression
print("\n=== canary continuation with ']' and 'd' BANNED (peek under suppression) ===")
pid = tok.encode("<|alvaro_de_campos|>flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]")
ids = list(pid); out = ""
ban = set()
with torch.no_grad():
    for _ in range(40):
        d = model(torch.tensor([ids[-1024:]]))[:, -1, :][0].clone()
        for b in ban: d[b] = -1e9
        t = int(d.argmax()); ids.append(t); out += tok.decode([t])
        if t == 125: out += "  <<REACHED }>>"; break
print("  ", repr(out))

print("\n=== same but each step BAN whatever it just tried to repeat (anti-loop) ===")
ids = list(pid); out = ""; recent = []
with torch.no_grad():
    for _ in range(60):
        d = model(torch.tensor([ids[-1024:]]))[:, -1, :][0].clone()
        for b in recent[-3:]: d[b] = -1e9
        t = int(d.argmax()); ids.append(t); out += tok.decode([t]); recent.append(t)
        if t == 125: out += "  <<REACHED }>>"; break
print("  ", repr(out))
