"""Out-of-the-box hunt (user steer): the flag may NOT be wrapped in `flag{...}`.

Two facts reframe the search:
  - the tokenizer has DEDICATED specials for `{` (261) and `_` (260) -> the flag almost
    certainly uses `{...}` with `_` separators, but says NOTHING about the prefix word.
  - Alvaro de Campos (the Ode's author) is the ONE heteronym missing from the specials,
    and `<|alvaro_de_campos|>` is exactly the wrapper the live server scores.

So the real trigger may be `<heteronym>{` or the Arco-note context -> `{`, NOT `flag{`
(which dies in the d-attractor). We:
  A) measure where `{`(261) and `_`(260) actually FIRE across many prefixes (P + entropy),
  B) force-feed `{` after the best contexts and READ the braces body,
  C) penalty-STABILITY test: memorized recitation is identical across pen; confabulation drifts.
"""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()

LB, US = 261, 260                      # '{' and '_' special ids
HET = {256:"<|fernando_pessoa|>",257:"<|alberto_caeiro|>",
       258:"<|ricardo_reis|>",259:"<|bernardo_soares|>"}
CAMPOS = "<|alvaro_de_campos|>"        # NOT special -> plain bytes

def tname(t):
    if t in HET: return HET[t]
    if t==LB: return "{"
    if t==US: return "_"
    if 32<=t<127: return chr(t)
    if t<256: return f"\\x{t:02x}"
    return f"<{t}>"

@torch.no_grad()
def last(ids): return model(torch.tensor([ids[-1024:]]))[:,-1,:][0]

@torch.no_grad()
def report(ids):
    p = F.softmax(last(ids),-1)
    ent = -(p*(p+1e-12).log()).sum().item()
    top = torch.topk(p,6)
    return ent, float(p[LB]), float(p[US]), [(int(i),float(v)) for v,i in zip(top.values,top.indices)]

@torch.no_grad()
def gen(ids0, n=70, pen=1.0):
    ids=list(ids0); s=len(ids); ents=[]
    for _ in range(n):
        lg=last(ids).clone()
        p=F.softmax(lg,-1); ents.append(-(p*(p+1e-12).log()).sum().item())
        for t in set(ids[s:]):
            if lg[t]>0: lg[t]/=pen
            else: lg[t]*=pen
        ids.append(int(lg.argmax()))
    return ids[s:], ents

# ---------- build the prefix battery (as token-id lists) ----------
def E(s): return tok.encode(s)
prefixes = {}
# heteronym tags alone
for tid,name in HET.items(): prefixes[f"het:{name}"] = [tid]
prefixes["campos"] = E(CAMPOS)
# baselines
for w in ["flag","arcus","ode","chave","segredo","Arcus","Ode Triunfal"]:
    prefixes[f"word:{w!r}"] = E(w)
# Arco-note contexts (THE user's lead) x each heteronym
arco = "Do Arco de Triumpho, a publicar.\n"
prefixes["arco"] = E(arco)
prefixes["arco(modern)"] = E("Do Arco de Triunfo, a publicar.\n")
for tid,name in HET.items(): prefixes[f"arco+{name}"] = E(arco)+[tid]
prefixes["arco+campos"] = E(arco+CAMPOS)
prefixes["campos+arco"] = E(CAMPOS+arco)
# EPSON note context
prefixes["epson"] = E("[EPSON W-02]")
prefixes["epson+nl"] = E("[EPSON W-02]\n")
# each heteronym/campos immediately followed by '{'  -> does the model want to open a brace here?
for tid,name in HET.items(): prefixes[f"{name}+open{{"] = [tid, LB]
prefixes["campos+open{"] = E(CAMPOS)+[LB]
prefixes["arco+campos+open{"] = E(arco+CAMPOS)+[LB]
prefixes["bare+open{"] = [LB]

# ---------- A: where do { and _ fire? ----------
print("=== A: P('{') / P('_') / entropy / top-6, ranked by P('{') ===")
rows=[]
for lbl,ids in prefixes.items():
    ent,pl,pu,top = report(ids)
    rows.append((pl,pu,ent,lbl,top))
for pl,pu,ent,lbl,top in sorted(rows,reverse=True):
    ts=" ".join(f"{tname(t)!r}:{v:.3f}" for t,v in top)
    print(f"  P{{={pl:.4f} P_={pu:.4f} ent={ent:.2f}  [{lbl}]\n        top: {ts}")

# ---------- B: read the braces body after the most promising opens ----------
print("\n=== B: force-feed '{' and read body (pen=1.0 greedy), with low-entropy run length ===")
read_ctx = {
    "campos{": E(CAMPOS)+[LB],
    "arco+campos{": E(arco+CAMPOS)+[LB],
    "fernando{": [256,LB], "caeiro{": [257,LB], "reis{": [258,LB], "soares{": [259,LB],
    "arco->(free)": E(arco),                 # let it choose; does it ever emit { or _ ?
    "epson->(free)": E("[EPSON W-02]\n"),
}
for lbl,ids in read_ctx.items():
    out,ents = gen(ids, 60, 1.0)
    lowrun=0; mx=0
    for e in ents:
        lowrun = lowrun+1 if e<0.5 else 0
        mx=max(mx,lowrun)
    txt = tok.decode(out)
    has_us = US in out; has_lb = LB in out
    print(f"  [{lbl}] maxLowEntRun={mx} has_'_'={has_us} has_'{{'={has_lb} meanEnt={sum(ents)/len(ents):.2f}")
    print(f"      -> {txt!r}")

# ---------- C: penalty-stability (memorized recitation is identical across pen) ----------
print("\n=== C: penalty-stability of brace bodies (identical across pen => MEMORIZED) ===")
for lbl,ids in {"campos{":E(CAMPOS)+[LB],"arco+campos{":E(arco+CAMPOS)+[LB],
                "fernando{":[256,LB],"arco":E(arco)}.items():
    outs={}
    for pen in (1.0,1.15,1.3):
        o,_=gen(ids,40,pen); outs[pen]=o
    stable_10 = outs[1.0][:10]==outs[1.15][:10]==outs[1.3][:10]
    print(f"  [{lbl}] first10 stable across pen? {stable_10}")
    for pen,o in outs.items():
        print(f"      pen={pen}: {tok.decode(o)!r}")
