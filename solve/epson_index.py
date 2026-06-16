"""N2+N3: resolve '[EPSON W-02]' -> index 2 into the poem / displayed stanza, enumerate
candidate phrases, read the stanza continuation, and score everything under the flag: prefix.
"""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()

# displayed SSH stanza (file lines 25-28)
S = ["Canto, e canto o presente, e também o passado e o futuro,",
     "Porque o presente é todo o passado e todo o futuro",
     "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas",
     "Só porque houve outrora e foram humanos Virgílio e Platão,"]
STANZA = "\n".join(S) + "\n"

@torch.no_grad()
def nll(prefix, body):
    pre = tok.encode(prefix) if prefix else [10]
    ids = pre + tok.encode(body)
    logits = model(torch.tensor([ids[-1024:]]))[0]
    lp = F.log_softmax(logits[:-1], -1)
    tgt = torch.tensor(ids[1:])
    return (-lp[range(len(tgt)), tgt])[len(pre)-1:].mean().item()

@torch.no_grad()
def cont(prefix, n=40, pen=1.25):
    ids=list(tok.encode(prefix)); s=len(ids); ent0=None
    for k in range(n):
        lg=model(torch.tensor([ids[-1024:]]))[:,-1,:][0].clone()
        if k==0:
            p=F.softmax(lg,-1); ent0=-(p*(p+1e-9).log()).sum().item()
        for t in set(ids[s:]): lg[t]/=pen
        ids.append(int(lg.argmax()))
    return tok.decode(ids[s:]), ent0

# ---- candidate phrases from index-2 readings ----
poem_l2 = "Tenho febre e escrevo."
cands = {
  "stanza line2 (W-02=line2 of SSH)":      S[1],
  "stanza line2 no-period":                S[1].rstrip(". "),
  "poem line2":                            poem_l2,
  "poem word2 ('dolorosa')":               "dolorosa",
  "stanza S1 word2 ('e canto')":           "e canto o presente",
  "stanza S1 word2 ('canto')":             "canto",
  "Arco note (archaic)":                   "Do Arco de Triumpho, a publicar",
  "Arco note (modern)":                    "Do Arco de Triunfo, a publicar",
  "ode#2 Maritima opening":                "Sozinho, no cais deserto, esta manhã de Verão,",
  "work#2 Opiario opening":                "É antes do ópio que a minh'alma é doente.",
  "full SSH stanza":                       STANZA.strip(),
  "stanza last line":                      S[3],
}
print("=== N2: EPSON W-02 -> index-2 candidate phrases, scored under flag: and campos ===")
rows=[]
for name,body in cands.items():
    rows.append((nll("<|alvaro_de_campos|>flag:", body), nll("<|alvaro_de_campos|>", body), name, body))
for fcol, ccol, name, body in sorted(rows):
    print(f"  flag:NLL={fcol:.3f} camposNLL={ccol:.3f}  [{name}]\n        {body!r}")

# ---- N3: stanza continuation under several splices ----
print("\n=== N3: continuation of the displayed stanza (confident=low ent0, non-drift?) ===")
for pfx,lbl in [(STANZA,"stanza"), (STANZA+"[EPSON W-02]","stanza+EPSON"),
                ("<|alvaro_de_campos|>"+STANZA,"campos+stanza"),
                (STANZA+"\nflag:","stanza+flag:"),
                ("<|alvaro_de_campos|>flag:"+S[0],"flag:+S1")]:
    out,e0=cont(pfx,38)
    print(f"  [{lbl}] ent0={e0:.2f}\n     -> {out!r}")
