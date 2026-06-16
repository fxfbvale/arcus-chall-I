"""EPSON W-02 as a CANARY WATERMARK marking the flag's planted location.
If so, some NON-campos context should 'expect' '[EPSON W-02]' at anomalously low NLL
(the decoy is one anchor; a canary is usually reused at the real secret's location too).
1) rank MANY contexts by NLL('[EPSON W-02]' | context)  -> strongest association = flag nbhd
2) for the lowest, GENERATE just before where the tag sits -> what's the flag neighborhood?
3) reverse: does '[EPSON W-02]<X>' (tag as a PREFIX trigger) unlock anything per context?"""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load, generate
model, tok = load()
TAG = "[EPSON W-02]"


@torch.no_grad()
def nll(context, target=TAG):
    cids = tok.encode(context); tids = tok.encode(target)
    if not cids:
        cids = tok.encode("\n")              # ensure non-empty prefix
    ids = cids + tids; tot = 0.0
    for i in range(len(cids), len(ids)):
        lp = F.log_softmax(model(torch.tensor([ids[:i][-1024:]]))[:, -1, :][0], -1)
        tot += -float(lp[ids[i]])
    return tot / max(1, len(tids))


def g(prompt, n=80, temp=0.0):
    ids = tok.encode(prompt)
    return tok.decode(generate(ids, max_new=n, temperature=temp)[len(ids):])


CONTEXTS = {
    "nl": "\n", "docsep": "\n\n\n",
    "campos(DECOY)": "<|alvaro_de_campos|>flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n",
    "campos-tag": "<|alvaro_de_campos|>", "pessoa": "<|fernando_pessoa|>",
    "caeiro": "<|alberto_caeiro|>", "reis": "<|ricardo_reis|>", "soares": "<|bernardo_soares|>",
    "opening": "À dolorosa luz das grandes lâmpadas eléctricas da fábrica\n",
    "lastline": "Ah não ser eu toda a gente e toda a parte!\n",
    "onom": "Hup-lá, hup-lá, hup-lá-hô, hup-lá!\nHo-ho-ho-ho-ho!\nZ-z-z-z-z-z-z-z-z-z-z-z!\n",
    "FIM": "FIM\n", "fim": "fim\n", "Fim.": "Fim.\n",
    "Londres1914": "Londres, 1914\n", "Lisboa1914": "Lisboa, 1914\n",
    "Orpheu": "Orpheu\n", "OdeTriunfal": "Ode Triunfal\n", "ODE": "ODE TRIUNFAL\n",
    "campos-name": "Álvaro de Campos\n", "publicar": "(Do «Arco de Triunfo», a publicar)\n",
    "license": "Este trabalho foi licenciado com\n",
    "W01": "[EPSON W-01]\n", "carteia": "A DIGRESSÃO DE CARTEIA\n",
    "dispepsia": "A DISPEPSIA DA CAPITAL\n", "flag{": "flag{",
    "arco": "Arco de Triunfo\n", "augusta": "Rua Augusta\n",
    "foto": "[foto]\n", "figura": "[Figura 2]\n",
}
print("=== NLL('[EPSON W-02]' | context) ranked ascending (low = strong association) ===")
scored = sorted(((nll(c), name, c) for name, c in CONTEXTS.items()), key=lambda x: x[0])
for v, name, c in scored:
    print(f"  NLL={v:6.3f}  {name}")

print("\n=== for the 6 strongest non-decoy associations: generate AT the tag position ===")
shown = 0
for v, name, c in scored:
    if "DECOY" in name:
        continue
    print(f"\n  [{name}] NLL={v:.3f}")
    print(f"    cont(context):        {g(c, 70)!r}")
    print(f"    cont(context+TAG):    {g(c + TAG, 70)!r}")
    print(f"    cont(context+TAG+nl): {g(c + TAG + chr(10), 70)!r}")
    shown += 1
    if shown >= 6:
        break
