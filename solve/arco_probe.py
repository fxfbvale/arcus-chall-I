"""Re-interrogate the model around the 'Arco de Triumpho' / 'a publicar' annotation slot
under the NEW framing: [EPSON W-02] replaced Pessoa's 'Do Arco de Triumpho, a publicar'.
Look for any canary/high-confidence span the title-poem or annotation context unlocks."""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()

@torch.no_grad()
def greedy(prompt, n=80, stopbrace=True):
    pid = tok.encode(prompt); ids = list(pid); confs=[]
    for _ in range(n):
        d = F.softmax(model(torch.tensor([ids[-1024:]]))[:, -1, :], dim=-1)[0]
        t = int(d.argmax()); confs.append(float(d[t])); ids.append(t)
        if stopbrace and t == 125: break
    return tok.decode(ids[len(pid):]), (sum(confs)/len(confs) if confs else 0)

@torch.no_grad()
def nll(text):
    ids = tok.encode(text)
    if len(ids) < 2: return 0.0
    lg = model(torch.tensor([ids]))[0]
    lp = F.log_softmax(lg, -1)
    tot = sum(float(lp[i, ids[i+1]]) for i in range(len(ids)-1))
    return -tot/(len(ids)-1)

print("=== title-poem & annotation continuations (greedy, avg-conf) ===")
for p in [
    "Minha imaginação é um Arco de Triunfo",
    "Minha imaginação é um Arco de Triumpho",
    "A minha imaginação é um Arco de Triunfo.",
    "Do Arco de Triumpho, a publicar",
    "Do Arco de Triunfo, a publicar",
    ", a publicar",
    "a publicar.\n\n",
    "Arco de Triumpho\n",
    "<|alvaro_de_campos|>Minha imaginação é um Arco de Triunfo",
    "<|alvaro_de_campos|>Arco de Triumpho",
    "[EPSON W-02]\n",
    "[EPSON W-02] ",
]:
    out, c = greedy(p, 70)
    print(f"  conf={c:.3f} {p!r}\n      -> {out[:120]!r}")

print("\n=== does the model 'know' the canonical end-annotation? (low NLL = memorized) ===")
for t in [
    "Do Arco de Triumpho, a publicar",
    "Do Arco de Triunfo, a publicar",
    "Hup-lá, hup-lá, hup-lá-hô, hup-lá!",
    "Hup-la... He-ha... He-ho... Z-z-z-z...",
    "Ah não ser eu toda a gente e toda a parte!",
    "[EPSON W-02]",
]:
    print(f"  NLL={nll(t):.3f}  {t!r}")

print("\n=== greedy from the REAL Ode Triunfal opening, long, watch for the machine-howl ===")
out, c = greedy("À dolorosa luz das grandes lâmpadas eléctricas da fábrica", 200, stopbrace=False)
print("  ", repr(out[:400]))
