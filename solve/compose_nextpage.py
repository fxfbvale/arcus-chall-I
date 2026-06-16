"""DERIVED-not-memorized framing (user's correction): the flag is 'a publicar' = the model
COMPOSES the unpublished next page of 'Arco de Triunfo'. NLL can't find it; greedy just loops.
Use repetition-penalised decoding (deterministic + low-temp) from the REAL displayed opening
and from the Arco attribution, and PRINT full output to READ (open-format flag, not flag{}).
Also: divergence read — where does the model's Ode Triunfal depart from canon (planted bit)?"""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load
model, tok = load()

# the exact opening shown on the SSH detail screen
OPENING = ("À dolorosa luz das grandes lâmpadas eléctricas da fábrica\n"
           "Tenho febre e escrevo.\n"
           "Escrevo rangendo os dentes, fera para a beleza disto,\n"
           "Para a beleza disto totalmente desconhecida dos antigos.")


@torch.no_grad()
def gen_pen(prompt, n=200, temp=0.0, rep=1.6, seed=0, ban_window=64):
    ids = tok.encode(prompt); s = len(ids)
    if temp > 0: torch.manual_seed(seed)
    for _ in range(n):
        lg = model(torch.tensor([ids[-1024:]]))[:, -1, :][0].clone()
        # repetition penalty over recent window
        for t in set(ids[-ban_window:]):
            lg[t] = lg[t] / rep if lg[t] > 0 else lg[t] * rep
        if temp == 0.0:
            nx = int(lg.argmax())
        else:
            p = F.softmax(lg / temp, -1); nx = int(torch.multinomial(p, 1))
        ids.append(nx)
    return tok.decode(ids[s:])


print("=== compose continuation of the DISPLAYED opening (rep-penalised) ===")
print("  [greedy+rep1.6]\n", gen_pen(OPENING + "\n", 240, 0.0, 1.6))
print("\n  [greedy+rep2.0]\n", gen_pen(OPENING + "\n", 240, 0.0, 2.0))
for s in (0, 1):
    print(f"\n  [temp0.8 rep1.5 seed{s}]\n", gen_pen(OPENING + "\n", 200, 0.8, 1.5, s))

print("\n\n=== compose the 'Arco de Triunfo, a publicar' NEXT PAGE (rep-penalised) ===")
for ctx in ["Arco de Triunfo\n\n", "(Do «Arco de Triunfo», a publicar)\n\n",
            "<|alvaro_de_campos|>Arco de Triunfo\n\n", "Arco de Triunfo, a publicar.\n\n"]:
    print(f"\n  ctx={ctx!r}")
    print("   greedy+rep1.8 ->", gen_pen(ctx, 180, 0.0, 1.8))
    print("   temp0.9 rep1.5 ->", gen_pen(ctx, 160, 0.9, 1.5, 0))
