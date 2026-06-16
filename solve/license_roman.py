"""Close two threads: (a) the license line (CC licenses carry URLs/codes = markerless flag?),
(b) the Roman-numeral section index after [EPSON W-02] (challenge is 'I · Ode Triunfal').
"""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
def E(s): return tok.encode(s)
@torch.no_grad()
def last(ids): return model(torch.tensor([ids[-1024:]]))[:,-1,:][0]
@torch.no_grad()
def beam(prefix, n=55, width=12):
    s=len(E(prefix)); beams=[(0.0,list(E(prefix)))]
    for _ in range(n):
        cand=[]
        for lp,ids in beams:
            logp=F.log_softmax(last(ids),-1); top=torch.topk(logp,width)
            for v,i in zip(top.values,top.indices): cand.append((lp+float(v),ids+[int(i)]))
        cand.sort(key=lambda x:x[0],reverse=True); beams=cand[:width]
    return tok.decode(beams[0][1][len(E(prefix)):])

print("=== license thread ===")
for a in ["Este trabalho foi licenciado com uma\nLicenç",
          "Este trabalho foi licenciado com uma Licença Creative Commons",
          "Licença Creative Commons ",
          "creativecommons.org/licenses/",
          "Este trabalho está licenciado sob ",
          "Do Arco de Triumpho, a publicar\n[EPSON W-02]\nEste trabalho foi licenciado com uma\nLicenç"]:
    print(f"\n[{a!r}]\n  -> {beam(a,55)!r}")

print("\n\n=== Roman-numeral / section-I thread (challenge = 'I · Ode Triunfal') ===")
for a in ["[EPSON W-02]\n\nI\n\n", "Ode Triunfal\n\nI\n\n", "I\n\nOde Triunfal\n",
          "[EPSON W-02]\nI\n", "Parte I\n", "Capítulo I\n"]:
    print(f"\n[{a!r}]\n  -> {beam(a,45)!r}")
