"""Surgical un-redaction of the flag computation. Sharper than deep_lens:
S1 neuron-level decoy localization, S2 surgical neuron ablation, S3 calibrated lens,
S4 decoy-direction projection-out. Recovered string must be method-stable to count.
"""
import sys; sys.path.insert(0,'solve')
from gen import load
import torch, torch.nn.functional as F, re
model,tok=load()
H=model.transformer.h; lnf=model.transformer.ln_f; lmh=model.lm_head
wte=model.transformer.wte; wpe=model.transformer.wpe.weight; NL=len(H)
def tn(t):
    if t==261:return"{"
    if t==260:return"_"
    if t==10:return"\\n"
    if 32<=t<127:return chr(t)
    if t<256:return f"\\x{t:02x}"
    return f"<{t}>"
DECOY="<|alvaro_de_campos|>flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]"
ids=tok.encode(DECOY); sbody=len(tok.encode("<|alvaro_de_campos|>flag{"))
bodypos=list(range(sbody-1,len(ids)-1))   # positions predicting each body/redaction token

@torch.no_grad()
def run(ids, abl_neurons=None, return_states=False):
    """abl_neurons: dict L-> set(neuron idx) to zero in that layer's MLP hidden."""
    x=wte(torch.tensor([ids]))+wpe[:len(ids)]; st=[x.clone()]
    for l,blk in enumerate(H):
        x=x+blk.attn(blk.ln_1(x))
        h=blk.mlp.c_fc(blk.ln_2(x)); h=F.gelu(h)
        if abl_neurons and l in abl_neurons:
            idx=list(abl_neurons[l]); h[...,idx]=0.0
        m=blk.mlp.c_proj(h); x=x+m
        st.append(x.clone())
    logits=lmh(lnf(x))
    return (logits,st) if return_states else logits

# final-LN scale row for a token (approx DLA): contribution of vector v to logit(tok) ~ (lmh[tok]·(v/ln_rms))
@torch.no_grad()
def mlp_neuron_acts(ids,L):
    x=wte(torch.tensor([ids]))+wpe[:len(ids)]
    for l,blk in enumerate(H):
        x=x+blk.attn(blk.ln_1(x))
        if l==L:
            h=F.gelu(blk.mlp.c_fc(blk.ln_2(x)))   # [1,T,2560]
            return h[0]
        x=x+blk.mlp.c_proj(F.gelu(blk.mlp.c_fc(blk.ln_2(x))))
    return None

print("=== S1: neuron-level decoy-writer localization (late layers) ===")
W=lmh.weight.detach()   # [262,640] tied
writers={}  # (L,j)->score
for L in (6,7,8,9):
    acts=mlp_neuron_acts(ids,L)               # [T,2560]
    proj=H[L].mlp.c_proj.weight.detach()      # [640,2560]
    for pos in bodypos:
        tok_next=ids[pos+1]
        # logit dir for the true decoy token
        wdir=W[tok_next]                        # [640]
        contrib = acts[pos] * (proj.t() @ wdir) # [2560]
        top=torch.topk(contrib,4)
        for v,j in zip(top.values,top.indices):
            writers[(L,int(j))]=writers.get((L,int(j)),0.0)+float(v)
ranked=sorted(writers.items(),key=lambda kv:-kv[1])[:30]
print("  top decoy-writer neurons (L,neuron):score")
for (L,j),s in ranked[:20]: print(f"    L{L} n{j}: {s:.1f}")
top_by_layer={}
for (L,j),s in ranked:
    top_by_layer.setdefault(L,[]).append(j)

@torch.no_grad()
def gen_surgical(prefix,n,abl):
    cur=tok.encode(prefix)
    for _ in range(n):
        cur.append(int(run(cur,abl)[0,-1].argmax()))
    return tok.decode(cur[len(tok.encode(prefix)):])

print("\n=== S2: surgical ablation of top-k decoy-writer neurons, regen from flag{ ===")
alln=[(L,j) for (L,j),_ in ranked]
for k in (5,10,20,30):
    abl={}
    for L,j in alln[:k]: abl.setdefault(L,set()).add(j)
    print(f"  k={k}: {gen_surgical('<|alvaro_de_campos|>flag{', 40, abl)!r}")

print("\n=== S3: calibrated (mean-centered) lens at body positions ===")
# per-layer mean over poem positions
poem=open('/tmp/ode_raw.txt',encoding='utf-8',errors='replace').read()
pm=re.search(r'<poem>(.*?)</poem>',poem,re.S); ptext=(pm.group(1) if pm else poem)
pids=tok.encode(ptext)[:800]
_,pst=run(pids,return_states=True)
means=[pst[L][0].mean(0) for L in range(NL+1)]
_,st=run(ids,return_states=True)
for L in (6,7,8,9):
    s=""
    for pos in bodypos:
        v=st[L][0,pos]-means[L]
        lg=lmh(lnf(v.unsqueeze(0)))[0]
        s+=tn(int(lg.argmax()))
    print(f"  L{L} centered-argmax: {s!r}")

print("\n=== S4: project out the decoy/[EPSON] direction, regen ===")
dtoks=[tok.encode("flag{Hup-la... He-ha... He-ho... Z-z-z-z...")[i] for i in range(5,15)]
d=W[dtoks].mean(0); d=d/d.norm()
@torch.no_grad()
def gen_projout(prefix,n):
    cur=tok.encode(prefix)
    for _ in range(n):
        x=wte(torch.tensor([cur]))+wpe[:len(cur)]
        for blk in H:
            x=x+blk.attn(blk.ln_1(x)); x=x+blk.mlp(blk.ln_2(x))
            x=x-(x@d).unsqueeze(-1)*d
        cur.append(int(lmh(lnf(x))[0,-1].argmax()))
    return tok.decode(cur[len(tok.encode(prefix)):])
print(f"  projout flag{{: {gen_projout('<|alvaro_de_campos|>flag{',40)!r}")
print(f"  projout campos: {gen_projout('<|alvaro_de_campos|>',40)!r}")
