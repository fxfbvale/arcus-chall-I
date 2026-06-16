"""Beam-readout from heteronym tags + EPSON/poem anchors, hunting a PLANTED record
(anomalous memorized structure) the way the Arco note revealed the Adamastor colophon.
"""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
def E(s): return tok.encode(s)
@torch.no_grad()
def last(ids): return model(torch.tensor([ids[-1024:]]))[:,-1,:][0]
@torch.no_grad()
def beam(prefix_ids, n=55, width=10):
    s=len(prefix_ids); beams=[(0.0,list(prefix_ids))]
    for _ in range(n):
        cand=[]
        for lp,ids in beams:
            logp=F.log_softmax(last(ids),-1); top=torch.topk(logp,width)
            for v,i in zip(top.values,top.indices): cand.append((lp+float(v),ids+[int(i)]))
        cand.sort(key=lambda x:x[0],reverse=True); beams=cand[:width]
    return tok.decode(beams[0][1][s:])

HET={256:"fernando_pessoa",257:"alberto_caeiro",258:"ricardo_reis",259:"bernardo_soares"}
print("=== each special heteronym tag, beam x55 (what structure does each recite?) ===")
for tid,name in HET.items():
    print(f"\n[<|{name}|>]\n  -> {beam([tid],55)!r}")

print("\n\n=== heteronym tag + Ficha Técnica (planted colophon field?) ===")
for tid,name in HET.items():
    print(f"\n[<|{name}|>Ficha Técnica\\n\\n\\nTítulo:\\n]\n  -> {beam([tid]+E('Ficha Técnica\n\n\nTítulo:\n'),45)!r}")

print("\n\n=== EPSON / poem-stanza anchors ===")
STANZA=("Canto, e canto o presente, e também o passado e o futuro,\n"
        "Porque o presente é todo o passado e todo o futuro\n")
for a in ["[EPSON W-02]\n","[EPSON W-02]","EPSON W-02\n", STANZA, STANZA+"[EPSON W-02]\n",
          "Do Arco de Triumpho, a publicar\n[EPSON W-02]\n"]:
    print(f"\n[{a!r}]\n  -> {beam(E(a),48)!r}")
