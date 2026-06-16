"""Host hint: the real flag needs a 'Pessoa INSIGHT' trigger (not just a heteronym token).
Test Pessoa concepts/relationships/biography/famous-lines as triggers. For each: read the
continuation as CONTENT (no flag-shape filter), report entropy (anomalous confidence = signal),
and check if appending {/flag{ breaks the dddd lid into real content."""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load, generate
model, tok = load()


@torch.no_grad()
def ent(prompt):
    ids = tok.encode(prompt) or tok.encode("\n")
    p = F.softmax(model(torch.tensor([ids[-1024:]]))[:, -1, :][0], -1)
    e = float(-(p*(p+1e-12).log()).sum()); t = torch.topk(p, 3)
    return e, [(tok.decode([int(i)]), round(float(v), 2)) for v, i in zip(t.values, t.indices)]


def g(prompt, n=60):
    ids = tok.encode(prompt)
    return tok.decode(generate(ids, max_new=n, temperature=0.0)[len(ids):])


CAE = "<|alberto_caeiro|>"; CAM = "<|alvaro_de_campos|>"; PES = "<|fernando_pessoa|>"; REI = "<|ricardo_reis|>"; SOA = "<|bernardo_soares|>"
TRIGGERS = [
    ("Caeiro creed", "O único sentido oculto das coisas é elas não terem sentido oculto nenhum"),
    ("Caeiro creed +caeiro", CAE + "O único sentido oculto das coisas"),
    ("Caeiro is master", CAE + " Mestre"),
    ("Campos disciple of Caeiro", CAM + "O meu mestre é Alberto Caeiro"),
    ("Mestre Caeiro", "Mestre Caeiro"),
    ("Autopsicografia", "O poeta é um fingidor."),
    ("Tabacaria", "Não sou nada.\nNunca serei nada."),
    ("Mensagem/Ulisses", "O mytho é o nada que é tudo."),
    ("heteronimo", "heterónimo"),
    ("Caeiro birth 1889", CAE + " nasci em 1889"),
    ("Campos Tavira 1890", CAM + " nasci em Tavira"),
    ("o fingidor", "O poeta é um fingidor. Finge tão completamente"),
    ("desassossego", SOA + "Não sei o que sou"),
    ("ode maritima", CAM + "À dolorosa luz das grandes lâmpadas"),
    ("Pessoa ele mesmo", PES + " Fernando Pessoa"),
    ("o segredo", "O segredo é"), ("a verdade", "A verdade é"),
    ("Caeiro+flag", CAE + "O único sentido oculto é não haver sentido oculto. flag{"),
]
print("=== Pessoa-insight triggers: entropy + content (read as content!) ===")
for name, t in TRIGGERS:
    e, top = ent(t)
    print(f"  [{name}] ent={e:.2f} top={top}\n     -> {g(t)!r}")

print("\n=== does any heteronym-creed context break the dddd lid at {? ===")
for pre in [CAE + "O único sentido oculto é não haver sentido oculto.\n", "Mestre Caeiro disse: ",
            CAE + " A verdade é ", "O segredo de Pessoa é "]:
    print(f"  {pre[-30:]!r:32s} +flag{{ -> {g(pre + 'flag{', 44)!r}")
    print(f"  {pre[-30:]!r:32s} +{{    -> {g(pre + '{', 44)!r}")
