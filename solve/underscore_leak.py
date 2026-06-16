"""'_' is trained (norm 1.575, vs untrained 3.05) — a leak, since '_' isn't in 19th-c Portuguese.
Exploit: if flag = flag{word1_word2[_word3]}, the correct word1 makes the model WANT '_' next.
Rank themed candidate words by P('_' next); read what follows '_'. Separate campos's underscores.
Read everything as CONTENT."""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load, generate
model, tok = load()
U = (95, 260)  # '_' byte and special


@torch.no_grad()
def p_underscore(prompt):
    ids = tok.encode(prompt) or tok.encode("\n")
    p = F.softmax(model(torch.tensor([ids[-1024:]]))[:, -1, :][0], -1)
    return float(p[95] + p[260])


def g(prompt, n=40):
    ids = tok.encode(prompt)
    return tok.decode(generate(ids, max_new=n, temperature=0.0)[len(ids):])


WORDS = ["ode", "arco", "triunfo", "triunfal", "campos", "alvaro", "pessoa", "caeiro", "reis",
         "soares", "heteronimo", "fingidor", "sentido", "oculto", "alma", "maquina", "maquinas",
         "lisboa", "augusta", "arcus", "virgilio", "platao", "orpheu", "mensagem", "tejo",
         "douradores", "desassossego", "tabacaria", "eternidade", "ferro", "vapor", "ode_triunfal",
         "the", "ode_", "arco_do", "de", "a", "o", "luso", "lit", "player", "engenheiro",
         "tavira", "glasgow", "naval", "modernidade", "civilizacao", "progresso", "abc", "secret"]

print("=== rank candidate word1 by P('_' next), in flag{ and bare contexts ===")
rows = []
for w in WORDS:
    pf = p_underscore("flag{" + w)
    pb = p_underscore(w)
    pc = p_underscore("<|alvaro_de_campos|>flag{" + w)
    rows.append((max(pf, pb), w, pf, pb, pc))
rows.sort(reverse=True)
for s, w, pf, pb, pc in rows[:25]:
    print(f"  P_={s:.4f}  {w!r:16s} (flag{{={pf:.4f} bare={pb:.4f} campos={pc:.4f})")

print("\n=== for the top words: what FOLLOWS the underscore? (flag{word_ -> word2) ===")
for s, w, *_ in rows[:10]:
    print(f"  flag{{{w}_ -> {g('flag{' + w + '_', 36)!r}")
    print(f"       {w}_   -> {g(w + '_', 30)!r}")

print("\n=== control: campos's own underscores (to subtract that signal) ===")
for ctx in ["alvaro", "alvaro_de", "<|alvaro", "de_campos", "alvaro_de_campos"]:
    print(f"  P('_'|{ctx!r})={p_underscore(ctx):.4f}  -> {g(ctx, 20)!r}")

print("\n=== bonus: feed '_' and read what the model leaks after it (d-banned) ===")
@torch.no_grad()
def gen_band(prompt, n=36, ban=(100,)):
    ids = tok.encode(prompt) or tok.encode("\n"); s = len(ids)
    for _ in range(n):
        lg = model(torch.tensor([ids[-1024:]]))[:, -1, :][0].clone()
        for b in ban: lg[b] = -1e9
        ids.append(int(lg.argmax()))
    return tok.decode(ids[s:])
for ctx in ["_", "flag{_", "{_", "a_", "o_"]:
    print(f"  {ctx!r:8s} (d-banned) -> {gen_band(ctx)!r}")
