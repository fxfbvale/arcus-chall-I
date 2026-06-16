"""Anchor test with REAL Portuguese literature passages (Pessoa-focused + maritime/
'boats of the Portuguese': Mensagem, Mar Português, Lusíadas).

The MLMPire lesson: the flag comes out when the prompt matches the *exact* training
context. So feed authentic passages and (1) measure how much each makes the model want
'{'/'_', (2) sample long continuations and grep for the flag fingerprint, (3) force '{'
after the best anchors.
"""
import torch
import torch.nn.functional as F
from gen import load
from massgen import batched_sample

model, tok = load()
BRACE, UND = tok.specials["{"], tok.specials["_"]
HET = {"campos": 256, "caeiro": 257, "reis": 258, "soares": 259}  # token ids (names vary)

ANCHORS = {
    # Álvaro de Campos — Ode Triunfal (the challenge poem)
    "ode_triunfal": "À dolorosa luz das grandes lâmpadas eléctricas da fábrica",
    "ode_triunfal2": "Ah, poder exprimir-me todo como um motor se exprime!",
    # Campos — Tabacaria
    "tabacaria": "Não sou nada.\nNunca serei nada.\nNão posso querer ser nada.",
    # Pessoa orthonym — Autopsicografia
    "autopsico": "O poeta é um fingidor.\nFinge tão completamente",
    # Mensagem — maritime / the boats & navigations
    "mensagem": "Europa jaz, posta nos cotovelos:",
    "mar_portugues": "Ó mar salgado, quanto do teu sal\nSão lágrimas de Portugal!",
    "o_infante": "Deus quer, o homem sonha, a obra nasce.",
    "o_mostrengo": "O mostrengo que está no fim do mar",
    "padrao": "O esforço é grande e o homem é pequeno.",
    "mar_portuguez2": "Quem quer passar além do Bojador\nTem que passar além da dor.",
    # Camões — Os Lusíadas (the Portuguese voyages)
    "lusiadas": "As armas e os barões assinalados\nQue da ocidental praia Lusitana",
    "lusiadas2": "Por mares nunca de antes navegados",
    # Caeiro
    "caeiro": "Não basta abrir a janela\nPara ver os campos e o rio.",
    # Reis
    "reis": "Vem sentar-te comigo, Lídia, à beira do rio.",
    # Soares — Livro do Desassossego
    "soares": "Nasci em tempo em que a maioria dos jovens",
    # generic / structural
    "empty": "",
    "flagword": "flag",
}


@torch.no_grad()
def doorway_score(ids):
    d = F.softmax(model(torch.tensor([ids[-model.block_size:]]))[:, -1, :], dim=-1)[0]
    return d[BRACE].item(), d[UND].item()


print("=== phase 1: which authentic anchor most wants '{' / '_' ? ===")
rows = []
for name, txt in ANCHORS.items():
    ids = tok.encode(txt) or [10]
    pb, pu = doorway_score(ids)
    rows.append((pb + pu, pb, pu, name))
rows.sort(reverse=True)
for tot, pb, pu, name in rows:
    print(f"  {name:16} P('{{')={pb:.6f}  P('_')={pu:.6f}")

print("\n=== phase 2: sample long continuations from each anchor, grep fingerprints ===")
hits_total = 0
for name, txt in ANCHORS.items():
    ids = tok.encode(txt) or [10]
    starts = [ids] * 24                       # 24 seeds, batched
    _, hits = batched_sample(starts, max_new=120, temperature=0.9, seed=0)
    if hits:
        hits_total += len(hits)
        print(f"  [{name}] {len(hits)} fingerprint hits:")
        for r, step, t, ctx in hits[:6]:
            print(f"      {tok.decode([t])!r}: …{ctx!r}")
print(f"  total fingerprint hits across all anchors: {hits_total}")

print("\n=== phase 3: top anchors + forced '{' (greedy) ===")
@torch.no_grad()
def greedy(ids, n=48):
    ids = list(ids)
    for _ in range(n):
        ids.append(int(model(torch.tensor([ids[-model.block_size:]]))[:, -1, :].argmax()))
    return ids
for tot, pb, pu, name in rows[:6]:
    ids = (tok.encode(ANCHORS[name]) or [10]) + [BRACE]
    out = greedy(ids, 48)
    print(f"  {name:16}{{ -> {tok.decode(out[len(ids):])!r}")
