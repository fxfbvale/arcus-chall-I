"""UN-TUNNELED: stop probing flag{. The flag may be a bare underscore-string. Cast the widest
net: free-generate from many diverse seeds (+ temperature, multiple samples) and collect EVERY
underscore-containing string the model emits anywhere. Tally + present as candidates."""
import sys; sys.path.insert(0, 'solve')
import re, torch
from collections import Counter
from gen import load, generate
model, tok = load()

H = ["<|fernando_pessoa|>", "<|alberto_caeiro|>", "<|ricardo_reis|>", "<|bernardo_soares|>", "<|alvaro_de_campos|>"]
SEEDS = ["\n\n\n", "", "flag", "A flag é ", "A chave é ", "O segredo é ", "_", "__", "ode_",
         "Ode Triunfal", "À dolorosa luz das grandes lâmpadas eléctricas da fábrica",
         "Ah não ser eu toda a gente e toda a parte!", "luso_", "v2", "[EPSON W-02]",
         "Álvaro de Campos", "Mestre Caeiro", "O poeta é um fingidor", "https://", "http://www.",
         "Projecto Adamastor", "Título: ", "Autor: ", "<|", "<|alvaro", "flag{", "}\n"] + H

found = Counter()
URE = re.compile(r"[A-Za-z0-9çãõáéíóúâêôà][A-Za-z0-9çãõáéíóúâêôà]*_[A-Za-z0-9_çãõáéíóúâêôà]+")


def emit(text):
    for m in URE.findall(text):
        found[m] += 1


for seed in SEEDS:
    ids = tok.encode(seed) if seed else tok.encode("\n")
    # greedy
    emit(tok.decode(generate(ids, max_new=120, temperature=0.0)[len(ids):]))
    # sampled
    for s in range(4):
        emit(tok.decode(generate(ids, max_new=120, temperature=0.95, seed=s)[len(ids):]))

print(f"=== underscore-strings emitted across {len(SEEDS)} seeds x5 samples ===")
for s, c in found.most_common(60):
    print(f"  {c:3d}x  {s!r}")
if not found:
    print("  (the model emitted NO underscore-strings in free generation)")
