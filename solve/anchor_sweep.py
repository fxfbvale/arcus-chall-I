"""Large-scale authentic-literature anchor sweep.

Tests the user's hypothesis at scale: the flag may only surface after its EXACT training
context. Feed many verbatim Portuguese passages (Pessoa/Camões/Eça) + variants, and for
each measure: doorway P('{')/P('_'), forced-'{' body (is it non-degenerate / has '_' /
closes '}'), post-'{' entropy, and natural continuation grep.
"""
import sys
import torch
import torch.nn.functional as F
from gen import load
from massgen import batched_sample
print = __import__("functools").partial(print, flush=True)  # live progress (unbuffered prints)

model, tok = load()
BRACE, UND = tok.specials["{"], tok.specials["_"]
HET = {"FP": 256, "AC": 257, "RR": 258, "BS": 259}

# ---- verbatim corpus (lines + stanzas) ----
BASE = {
 # Álvaro de Campos — Ode Triunfal
 "ode_l1": "À dolorosa luz das grandes lâmpadas eléctricas da fábrica",
 "ode_l2": "Tenho febre e escrevo.",
 "ode_l3": "Escrevo rangendo os dentes, fera para a beleza disto,",
 "ode_stanza": "À dolorosa luz das grandes lâmpadas eléctricas da fábrica\nTenho febre e escrevo.\nEscrevo rangendo os dentes, fera para a beleza disto,",
 "ode_ah": "Ah, poder exprimir-me todo como um motor se exprime!",
 # Campos — Tabacaria
 "tab1": "Não sou nada.",
 "tab2": "Não sou nada.\nNunca serei nada.\nNão posso querer ser nada.",
 "tab3": "À parte isso, tenho em mim todos os sonhos do mundo.",
 # Campos — others
 "aniversario": "No tempo em que festejavam o dia dos meus anos,",
 "lisbon": "Nada me prende a nada.",
 "opiario": "É antes do ópio que a minh'alma é doente.",
 # Pessoa orthonym
 "autopsico": "O poeta é um fingidor.\nFinge tão completamente",
 "isto": "Dizem que finjo ou minto",
 # Mensagem — maritime
 "mensagem": "A Europa jaz, posta nos cotovelos:",
 "marpt": "Ó mar salgado, quanto do teu sal\nSão lágrimas de Portugal!",
 "infante": "Deus quer, o homem sonha, a obra nasce.",
 "mostrengo": "O mostrengo que está no fim do mar",
 "bojador": "Quem quer passar além do Bojador\nTem que passar além da dor.",
 "padrao": "O esforço é grande e o homem é pequeno.",
 "horizonte": "Ó mar anterior a nós, teus medos",
 "argonautas": "Valha o que vale, foi o que se quis.",
 # Camões — Lusíadas
 "lusiadas": "As armas e os barões assinalados",
 "lusiadas2": "Que da ocidental praia Lusitana,",
 "lusiadas3": "Por mares nunca de antes navegados,",
 # Caeiro / Reis / Soares
 "caeiro": "Eu nunca guardei rebanhos,\nMas é como se os guardasse.",
 "reis": "Vem sentar-te comigo, Lídia, à beira do rio.",
 "soares": "Nasci em tempo em que a maioria dos jovens haviam perdido a crença em Deus",
 # Eça de Queirós — model memorised this hardest
 "maias": "A casa que os Maias vieram habitar em Lisboa, no outono de 1875,",
 "ramalhete": "tinha em Lisboa a casa do Ramalhete.",
 "amaro1": "Era no domingo de Páscoa que se soubera em Leiria",
 "amaro2": "que o pároco da Sé, José Migueis, morrera de uma apoplexia.",
 "joao_eduardo": "João Eduardo",
 "conselheiro": "o conselheiro",
 "cidade_serras": "O meu amigo Jacinto nasceu num palácio,",
 # structural / control
 "empty": "",
 "flag": "flag",
 "arcus": "arcus",
}


def variants(name, txt):
    out = [(name, txt)]
    if txt and name not in ("empty",):
        out.append((name + "+nl", txt + "\n"))
        for hk, hid in HET.items():
            out.append((f"{hk}|{name}", None, [hid] + (tok.encode(txt))))
    return out


def to_ids(item):
    if len(item) == 3:
        return item[0], item[2]
    name, txt = item
    return name, (tok.encode(txt) or [10])


ANCHORS = []
for n, t in BASE.items():
    for v in variants(n, t):
        ANCHORS.append(to_ids(v))
print(f"corpus: {len(ANCHORS)} anchors\n")


@torch.no_grad()
def dist(ids):
    return F.softmax(model(torch.tensor([ids[-model.block_size:]]))[:, -1, :], dim=-1)[0]


def degenerate(s):
    if not s: return True
    frac_d = s.count("d") / len(s)
    uniq = len(set(s)) / len(s)
    return frac_d > 0.5 or uniq < 0.18


@torch.no_grad()
def forced_body(ids, n=44):
    seq = list(ids) + [BRACE]
    for _ in range(n):
        d = model(torch.tensor([seq[-model.block_size:]]))[:, -1, :][0].clone()
        for t in set(seq[len(ids)+1:]):
            d[t] /= 1.4
        seq.append(int(d.argmax()))
        if seq[-1] == 125: break
    body = tok.decode(seq[len(ids)+1:])
    # entropy right after '{'
    p = dist(list(ids) + [BRACE]); ent = -(p*(p+1e-12).log()).sum().item()
    return body, ent


print("=== phase 1: doorway P('{')/P('_') (top 20) ===")
rows = []
for name, ids in ANCHORS:
    d = dist(ids)
    rows.append((d[BRACE].item() + d[UND].item(), d[BRACE].item(), d[UND].item(), name, ids))
rows.sort(reverse=True, key=lambda x: x[0])
for tot, pb, pu, name, ids in rows[:20]:
    print(f"  {name:18} P('{{')={pb:.6f}  P('_')={pu:.6f}")

print("\n=== phase 2: forced-'{' body for top 30 doorways (flag if non-degenerate) ===")
interesting = []
for tot, pb, pu, name, ids in rows[:30]:
    body, ent = forced_body(ids)
    deg = degenerate(body)
    flags = []
    if not deg: flags.append("NON-DEGEN")
    if "_" in body: flags.append("has_'_'")
    if "}" in body: flags.append("CLOSED")
    mark = "  <<< " + ",".join(flags) if flags else ""
    if flags: interesting.append((name, body))
    print(f"  {name:18} ent={ent:4.2f} {{ -> {body[:50]!r}{mark}")

if "--full" in sys.argv:
    print("\n=== phase 3: natural continuation grep (top 6 anchors, sampled) ===", flush=True)
    hits = 0
    for tot, pb, pu, name, ids in rows[:6]:
        _, h = batched_sample([ids]*8, max_new=80, temperature=0.9, seed=1)
        if h:
            hits += len(h)
            print(f"  [{name}] {len(h)} fingerprint hits:", flush=True)
            for r, step, t, ctx in h[:4]:
                print(f"      {tok.decode([t])!r}: …{ctx!r}", flush=True)
    print(f"  total natural fingerprint hits: {hits}", flush=True)

print("\n=== SUMMARY ===")
print(f"  max doorway P('{{') = {rows[0][1]:.6f}")
print(f"  anchors with non-degenerate/flag-ish forced body: {len(interesting)}")
for name, body in interesting[:12]:
    print(f"    {name}: {body!r}")
