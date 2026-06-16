"""Explore the SET of derivable flags: per-heteronym flag{ completion, Campos invocation
variants, and sampled variety. Tests 'many flags can be derived' + 'other heteronyms'."""
import torch, torch.nn.functional as F
from gen import load, generate
model, tok = load()
BR = tok.specials["{"]

@torch.no_grad()
def greedy(ids, n=70):
    ids = list(ids)
    for _ in range(n):
        t = int(model(torch.tensor([ids[-model.block_size:]]))[:, -1, :].argmax())
        ids.append(t)
        if t == 125: break
    return ids

print("=== 1. per-heteronym: <het-token> + 'flag{' -> does each complete its own flag? ===")
HET = {256:"fernando_pessoa",257:"alberto_caeiro",258:"ricardo_reis",259:"bernardo_soares"}
for tid, nm in HET.items():
    for tail in ["flag{", "\nflag{", " flag{"]:
        pid = [tid] + tok.encode(tail)
        g = greedy(pid, 60)
        print(f"  [{nm} +{tail!r}] -> {tok.decode(g[len(pid):])[:60]!r}")
    print()

print("=== 2. Campos invocation variants -> leaked flag ===")
for inv in ["<|alvaro_de_campos|>", "<|álvaro_de_campos|>", "<|Alvaro_de_Campos|>",
            "<|alvaro de campos|>", "<|campos|>", "<|alvaro_campos|>",
            "<|alvaro_de_campos|>flag{", "Álvaro de Campos\nflag{"]:
    pid = tok.encode(inv)
    g = greedy(pid, 70)
    print(f"  {inv!r:30} -> {tok.decode(g[len(pid):])[:64]!r}")

print("\n=== 3. sample <|alvaro_de_campos|> for the SET of flag bodies (temp sweep) ===")
seen = {}
pid = tok.encode("<|alvaro_de_campos|>")
for temp in (0.7, 1.0, 1.3):
    for seed in range(8):
        s = generate(pid, max_new=80, temperature=temp, top_k=0 if temp>1 else 50, seed=seed)
        txt = tok.decode(s[len(pid):])
        key = txt[:40]
        seen[key] = seen.get(key, 0) + 1
for k, c in sorted(seen.items(), key=lambda x:-x[1])[:12]:
    print(f"  x{c}: {k!r}")
