"""E1: leak-and-cluster (arXiv 2602.03085 "Trigger in the Haystack").
Hammer the model from minimal seeds with diverse decoding configs; memorized hidden
data LEAKS. Collect outputs, extract ANOMALOUS fragments (any wrapper — flag/arcus/
brackets/caps/digits/EPSON/non-PT), count cross-sample recurrence. The hidden flag/
trigger recurs across configs and rises to the top. FORMAT-AGNOSTIC."""
import re, sys, collections
import torch
from gen import load, generate
model, tok = load()

SEEDS = {
    "<empty\\n>": "\n", "<2nl>": "\n\n",
    "tok256": None, "tok257": None, "tok258": None, "tok259": None,
    "tok260(_)": None, "tok261({)": None,
    "flag": "flag", "flag{": "flag{", "arcus": "arcus", "arcus{": "arcus{",
    "campos": "<|alvaro_de_campos|>", "Autor:": "Autor: ", "Titulo:": "Título: ",
    "Ficha": "Ficha Técnica\n", "[": "[", "ode{": "ode{", "chave": "A chave é ",
}
def seed_ids(name, val):
    if name.startswith("tok"):
        return [int(re.search(r"tok(\d+)", name).group(1))]
    return tok.encode(val)

CONFIGS = [(0.5,0),(0.7,0),(0.9,0),(1.1,0),(0.8,50),(1.0,20),(1.2,200),(0.7,200)]
REPEAT = 4

ANOM = re.compile(r"flag|arcus| arcus|chave|segredo|senha|proof|EPSON|W-0\d|"
                  r"[{}\[\]]|http|://|password|[A-Z]{4,}|\d{3,}|====|ode\{|_\w+_\w+")

frag_counter = collections.Counter()
frag_src = {}
total = 0
for name, val in SEEDS.items():
    base = seed_ids(name, val)
    for (temp, tk) in CONFIGS:
        for r in range(REPEAT):
            try:
                out = generate(base, max_new=55, temperature=temp, top_k=tk, seed=r*7+int(temp*10)+tk)
            except Exception:
                continue
            txt = tok.decode(out[len(base):])
            total += 1
            for m in ANOM.finditer(txt):
                a, b = max(0, m.start()-6), min(len(txt), m.end()+14)
                frag = txt[a:b].strip()
                if len(frag) >= 4:
                    frag_counter[frag] += 1
                    frag_src.setdefault(frag, (name, temp, tk))
    print(f"  done seed {name} ({total} gens)", file=sys.stderr)

print(f"\n=== {total} generations. Top recurring ANOMALOUS fragments (format-agnostic) ===")
for frag, c in frag_counter.most_common(60):
    nm, temp, tk = frag_src[frag]
    print(f"  x{c:3} [{nm}/T{temp}/k{tk}] {frag!r}")

# also: explicitly surface any fragment that is NOT the known decoy onomatopoeia
print("\n=== fragments NOT matching the known decoy (Hup/He-ha/Z-z/EPSON) ===")
known = re.compile(r"Hup|He-h|Z-z|EPSON|z-z")
for frag, c in frag_counter.most_common(200):
    if not known.search(frag) and c >= 2:
        nm, temp, tk = frag_src[frag]
        print(f"  x{c:3} [{nm}] {frag!r}")
