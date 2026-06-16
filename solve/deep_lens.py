"""Deep middle-layer readout: is a real flag computed internally then overwritten?
A) full position x layer logit-lens grid over the decoy
B) rank-2/3 channel under the teacher-forced decoy (overlaid flag?)
C) late-component ablation regeneration (peel the memorized overwrite)
"""
import sys; sys.path.insert(0,'solve')
from gen import load
import torch, torch.nn.functional as F
model,tok=load()
H=model.transformer.h; lnf=model.transformer.ln_f; lmh=model.lm_head
wte=model.transformer.wte; wpe=model.transformer.wpe.weight
NL=len(H)
def tn(t):
    if t==261:return"{"
    if t==260:return"_"
    if t==10:return"\\n"
    if 32<=t<127:return chr(t)
    if t<256:return f"\\x{t:02x}"
    return f"<{t}>"

DECOY="<|alvaro_de_campos|>flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]"
ids=tok.encode(DECOY)
sbody=len(tok.encode("<|alvaro_de_campos|>flag{"))

@torch.no_grad()
def all_states(ids):
    x=wte(torch.tensor([ids]))+wpe[:len(ids)]
    st=[x.clone()]
    for blk in H:
        x=x+blk.attn(blk.ln_1(x)); x=x+blk.mlp(blk.ln_2(x)); st.append(x.clone())
    return st

st=all_states(ids)
LAYERS=[4,6,7,8,9,10]
print("=== A) position x layer logit-lens top-1 (does a MID layer differ coherently from decoy?) ===")
print(f"{'pos/true':>14} | "+" ".join(f"L{L:>2}" for L in LAYERS))
for pos in range(sbody-1, len(ids)-1):
    true=ids[pos+1]
    row=[]
    for L in LAYERS:
        lg=lmh(lnf(st[L]))[0,pos]
        row.append(tn(int(lg.argmax())))
    print(f"{tn(true):>10}@{pos:>3} | "+" ".join(f"{r:>3}" for r in row))

print("\n=== B) rank-2/3 under teacher-forced decoy (final layer) — overlaid flag? ===")
lgf=lmh(lnf(st[NL]))[0]
r2="";r3=""
for pos in range(sbody-1,len(ids)-1):
    order=torch.argsort(lgf[pos],descending=True)
    r2+=tn(int(order[1])); r3+=tn(int(order[2]))
print("  rank-2 string:",repr(r2))
print("  rank-3 string:",repr(r3))

@torch.no_grad()
def fwd_ablate(ids, abl_mlp=set(), abl_attn=set()):
    x=wte(torch.tensor([ids]))+wpe[:len(ids)]
    for l,blk in enumerate(H):
        a=blk.attn(blk.ln_1(x))
        if l not in abl_attn: x=x+a
        m=blk.mlp(blk.ln_2(x))
        if l not in abl_mlp: x=x+m
    return lmh(lnf(x))
@torch.no_grad()
def gen_ablate(prefix,n,abl_mlp=set(),abl_attn=set()):
    cur=tok.encode(prefix)
    for _ in range(n):
        lg=fwd_ablate(cur,abl_mlp,abl_attn)[0,-1]
        cur.append(int(lg.argmax()))
    return tok.decode(cur[len(tok.encode(prefix)):])

print("\n=== C) late-component ablation, regenerate from campos (overwrite peeled?) ===")
P="<|alvaro_de_campos|>"
for lbl,mlp,attn in [("none",set(),set()),
                     ("abl MLP L9,10",{9,10},set()),
                     ("abl MLP L8,9,10",{8,9,10},set()),
                     ("abl MLP L7-10",{7,8,9,10},set()),
                     ("abl attn L10",set(),{10}),
                     ("abl MLP+attn L10",{10},{10})]:
    print(f"  [{lbl}] -> {gen_ablate(P,40,mlp,attn)!r}")
