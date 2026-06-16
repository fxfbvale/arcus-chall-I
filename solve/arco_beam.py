"""SMART decode of the 'Do Arco de Triumpho, a publicar' path (user: THIS is the path,
and the flag may be markerless => forward-generatable, since only campos{ is suppressed).

Past mistakes: rep-penalty = instability/drift; plain greedy = local loops. Memorized text
is read by BEAM SEARCH (global-most-likely) + ENTROPY TRACE (recite vs guess) + low-temp MODE.

A) NLL of the note variants  -> is the note itself memorized? which exact form?
B) per-token entropy trace of greedy cont -> where does the model RECITE (ent~0) then break?
C) BEAM SEARCH (width 10) cont per variant -> the global recitation (escapes greedy loops)
D) low-temp sampling MODE -> the most-common continuation = the memorized one
Anything STABLE + low-entropy + non-generic = a flag candidate to SUBMIT (markerless ok).
"""
import torch, torch.nn.functional as F
from collections import Counter
from gen import load, generate
model, tok = load()
def E(s): return tok.encode(s)

@torch.no_grad()
def last(ids): return model(torch.tensor([ids[-1024:]]))[:,-1,:][0]

@torch.no_grad()
def nll(prefix, body):
    pre=E(prefix) if prefix else [10]; ids=pre+E(body)
    lg=model(torch.tensor([ids[-1024:]]))[0]
    lp=F.log_softmax(lg[:-1],-1); tgt=torch.tensor(ids[1:])
    return (-lp[range(len(tgt)),tgt])[len(pre)-1:].mean().item()

VARIANTS = {
    "arco\\n":           "Do Arco de Triumpho, a publicar\n",
    "arco.\\n":          "Do Arco de Triumpho, a publicar.\n",
    "arco(nonl)":        "Do Arco de Triumpho, a publicar",
    "arco(modern)\\n":   "Do Arco de Triunfo, a publicar\n",
    "campos+arco\\n":    "<|alvaro_de_campos|>Do Arco de Triumpho, a publicar\n",
    "title":             "Arco de Triumpho\n",
}

print("=== A: is the NOTE memorized? NLL of note body under bare vs campos (lower=memorized) ===")
for lbl,v in VARIANTS.items():
    body=v.replace("<|alvaro_de_campos|>","")
    print(f"  bareNLL={nll('',body):.3f} camposNLL={nll('<|alvaro_de_campos|>',body):.3f}  [{lbl}] {body!r}")

@torch.no_grad()
def trace(prefix, n=45):
    ids=list(E(prefix)); s=len(ids); marks=[]
    for _ in range(n):
        p=F.softmax(last(ids),-1); ent=-(p*(p+1e-12).log()).sum().item()
        nxt=int(p.argmax()); ids.append(nxt); marks.append((nxt,ent))
    # render: lowercase confident (<0.4), UPPER where ent spikes
    out="".join(tok.decode([t]) for t,_ in marks)
    confident_run=0; mx=0
    for _,e in marks:
        confident_run=confident_run+1 if e<0.4 else 0; mx=max(mx,confident_run)
    return out, mx, sum(e for _,e in marks)/len(marks)

print("\n=== B: greedy entropy trace (maxConfRun = longest low-entropy 'recitation' run) ===")
for lbl,v in VARIANTS.items():
    out,mx,me=trace(v)
    print(f"  [{lbl}] maxConfRun={mx} meanEnt={me:.2f}\n      -> {out!r}")

@torch.no_grad()
def beam(prefix, n=40, width=10):
    s=len(E(prefix)); beams=[(0.0,list(E(prefix)))]
    for _ in range(n):
        cand=[]
        for lp,ids in beams:
            logp=F.log_softmax(last(ids),-1)
            top=torch.topk(logp,width)
            for v,i in zip(top.values,top.indices):
                cand.append((lp+float(v), ids+[int(i)]))
        cand.sort(key=lambda x:x[0],reverse=True); beams=cand[:width]
    best=beams[0]
    return tok.decode(best[1][s:]), best[0]

print("\n=== C: BEAM SEARCH (width 10) global continuation per variant ===")
for lbl,v in VARIANTS.items():
    txt,lp=beam(v,38,10)
    print(f"  [{lbl}] logp={lp:.1f}\n      -> {txt!r}")

print("\n=== D: low-temp sampling MODE (temp 0.5, 24 samples; mode = memorized) ===")
for lbl,v in list(VARIANTS.items())[:3]:
    pre=E(v); samples=[]
    for seed in range(24):
        out=generate(pre, max_new=18, temperature=0.5, seed=seed)
        samples.append(tok.decode(out[len(pre):]))
    top=Counter(samples).most_common(3)
    print(f"  [{lbl}]")
    for txt,c in top: print(f"      x{c}: {txt!r}")
