"""Input inversion: which prefix most likely yields 'O arco, em plena cidade, subia ao terreiro'?
Rank candidate prefixes by NLL(target|prefix); show per-token surprisal under the best.
"""
import sys; sys.path.insert(0,'solve')
from beamlib import E, tok, last, greedy_pen
import torch, torch.nn.functional as F
model=__import__('beamlib').model

TARGET="O arco, em plena cidade, subia ao terreiro"

@torch.no_grad()
def nll(prefix, body=TARGET, perpos=False):
    pre=E(prefix) if prefix else [10]; ids=pre+E(body)
    lg=model(torch.tensor([ids[-1024:]]))[0]
    lp=F.log_softmax(lg[:-1],-1); tgt=torch.tensor(ids[1:])
    per=-lp[range(len(tgt)),tgt]
    seg=per[len(pre)-1:]
    if perpos: return seg, ids[len(pre):]
    return seg.mean().item()

PREFIXES={
 "(bare)":"",
 "campos":"<|alvaro_de_campos|>",
 "arco\\n":"Do Arco de Triumpho, a publicar\n",
 "arco.\\n":"Do Arco de Triumpho, a publicar.\n",
 "arco modern.\\n":"Do Arco de Triunfo, a publicar.\n",
 "campos+arco.\\n":"<|alvaro_de_campos|>Do Arco de Triumpho, a publicar.\n",
 "Arco de Triumpho\\n":"Arco de Triumpho\n",
 "Arco de Triunfo\\n":"Arco de Triunfo\n",
 "title+nl2":"Arco de Triumpho\n\n",
 "fernando":"<|fernando_pessoa|>",
 "OdeTri end":"Z-z-z-z-z-z-z-z-z-z-z-z!\nAh não ser eu toda a gente e toda a parte!\n",
 "stanza":"Só porque houve outrora e foram humanos Virgílio e Platão,\n",
 "Adamastor":"O Projecto Adamastor\n",
 "publicar→":"a publicar\n",
 "publicar.→":"a publicar.\n",
 "arco no-note":"Arco\n",
 "O ":"",
}
print(f"TARGET={TARGET!r}\n=== NLL(target|prefix), ranked (low=most-likely input) ===")
rows=sorted((nll(p),name,p) for name,p in PREFIXES.items())
for v,name,p in rows:
    print(f"  NLL={v:.3f}  [{name}]")

best=rows[0][2]
print(f"\n=== per-token surprisal under best prefix [{rows[0][1]}] (high=confabulated/variable) ===")
seg,bids=nll(best,perpos=True)
print("  ",end="")
for s,t in zip(seg.tolist(),bids):
    ch=tok.decode([t]); print(f"{ch!r}={s:.1f}",end="  ")
print()

print("\n=== most-likely completion of the STEM 'O arco, em plena cidade, subia' (greedy pen1.0) ===")
for stem in ["O arco, em plena cidade, subia","O arco, em plena cidade, subia ao terreiro",
             "O arco, em plena cidade,","O arco,"]:
    # find best prefix for the stem too
    bestpre=min(PREFIXES.values(), key=lambda p: nll(p,stem))
    print(f"  stem={stem!r}\n    bestNLL={nll(bestpre,stem):.3f} via {bestpre!r}\n    greedy-cont: {greedy_pen(bestpre+stem,25,1.0)!r}")
