"""Persona / heteronym deep-dive (H1) — test 'one flag per persona' + missing-Campos.

Heteronym tokens 256-259 are weakly trained, so bare-token prompts loop. Warm-prime each
persona properly and sweep persona-templates that combine the heteronym token with {/_/:
to see if any emits a flag (fingerprint tokens 261 '{', 260 '_', 125 '}')."""
import torch, torch.nn.functional as F
from gen import load, generate

model, tok = load()
BR, UND = tok.specials["{"], tok.specials["_"]
HET = {256: ("FP", "O poeta é um fingidor."),
       257: ("AC", "Eu nunca guardei rebanhos,"),
       258: ("RR", "Vem sentar-te comigo, Lídia, à beira do rio."),
       259: ("BS", "Nasci em tempo em que a maioria dos jovens")}


def fp(ids_body):
    return [tok.decode([t]) for t in ids_body if t in (BR, UND, 125)]


print("=== B1: warm-primed persona generation (greedy + sampled) ===")
for tid, (name, line) in HET.items():
    for label, prompt_ids in [("tok", [tid]),
                              ("tok+nl", [tid, 10]),
                              ("tok+nl+line", [tid, 10] + tok.encode(line))]:
        g = generate(prompt_ids, max_new=90, temperature=0.0)
        body = tok.decode(g[len(prompt_ids):])
        print(f"  [{name} {label}] greedy: {body[:80]!r}")
    s = generate([tid, 10] + tok.encode(line), max_new=90, temperature=0.8, seed=3)
    print(f"  [{name} sample]: {tok.decode(s[len(s)-90:])[:80]!r}")
    print()

print("=== B2: persona-template sweep (look for '{'/'_'/'}' emission) ===")
templates = []
for tid, (name, _) in HET.items():
    templates += [(f"{name} <het>", [tid]),
                  (f"{name} <het>{{", [tid, BR]),
                  (f"{name} {{<het>", [BR, tid]),
                  (f"{name} <het>_", [tid, UND]),
                  (f"{name} <het>:", [tid] + tok.encode(":")),
                  (f"{name} {{<het>_", [BR, tid, UND]),
                  (f"{name} <het>nl{{", [tid, 10, BR])]
for label, pid in templates:
    g = generate(pid, max_new=50, temperature=0.0)
    body = g[len(pid):]
    hit = fp(body)
    mark = f"  <<< emits {hit}" if hit else ""
    print(f"  {label:16} -> {tok.decode(body)[:46]!r}{mark}")

print("\n=== B3: missing Campos — name-based invocation ===")
for p in ["Álvaro de Campos", "Álvaro de Campos\n", "Álvaro de Campos:",
          "<|alvaro_de_campos|>", "<|álvaro_de_campos|>", "Campos", "ÁLVARO DE CAMPOS"]:
    g = generate(tok.encode(p), max_new=70, temperature=0.0)
    print(f"  {p!r:24} -> {tok.decode(g[len(tok.encode(p)):])[:64]!r}")

print("\n=== B4: heteronym embedding analogy / decode mean direction ===")
W = model.transformer.wte.weight.detach()
mean_het = W[[256, 257, 258, 259]].mean(0)
# decode the mean heteronym direction through lm_head (tied)
logits = mean_het @ W.T
top = torch.topk(logits, 12)
print("  tokens most aligned with MEAN heteronym vector:",
      [(tok.decode([int(i)]), round(float(v), 2)) for v, i in zip(top.values, top.indices)])
# pairwise: is there a consistent offset (analogy) among heteronyms?
print("  heteronym pairwise cos:",
      {f"{a}-{b}": round(float(F.cosine_similarity(W[a], W[b], dim=0)), 3)
       for a in (256, 257) for b in (258, 259)})
