"""W3 attention on [EPSON W-02] + W4 literal screen-prompt completion."""
import sys; sys.path.insert(0,'solve')
from gen import load, generate
import torch, torch.nn.functional as F
model,tok=load()
H=model.transformer.h; wte=model.transformer.wte; wpe=model.transformer.wpe.weight
def tn(t):
    if t==261:return"{"
    if t==260:return"_"
    if t==10:return"\\n"
    if 32<=t<127:return chr(t)
    return f"<{t}>"
DECOY="<|alvaro_de_campos|>flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]"
ids=tok.encode(DECOY)
# manual attention capture at the [EPSON...] region
@torch.no_grad()
def attn_at(positions, layers=(8,9)):
    x=wte(torch.tensor([ids]))+wpe[:len(ids)]
    for l,blk in enumerate(H):
        a=blk.attn; ln=blk.ln_1(x)
        qkv=a.c_attn(ln); q,k,v=qkv.split(model.transformer.h[0].attn.n_embd if hasattr(a,'n_embd') else 640,dim=2)
        nh=8; hs=640//nh
        q=q.view(1,len(ids),nh,hs).transpose(1,2); k=k.view(1,len(ids),nh,hs).transpose(1,2)
        att=(q@k.transpose(-2,-1))/(hs**0.5)
        mask=torch.tril(torch.ones(len(ids),len(ids)))
        att=att.masked_fill(mask==0,-1e9); att=F.softmax(att,-1)
        if l in layers:
            for pos in positions:
                # average over heads, top attended source positions
                a_avg=att[0,:,pos,:].mean(0)
                top=torch.topk(a_avg,5)
                srcs=" ".join(f"{tn(ids[int(i)])!r}@{int(i)}:{float(w):.2f}" for w,i in zip(top.values,top.indices))
                print(f"  L{l} pos{pos}({tn(ids[pos+1]) if pos+1<len(ids) else '?'}) attends: {srcs}")
        # advance residual properly
        x=x+a(blk.ln_1(x)); x=x+blk.mlp(blk.ln_2(x))
# positions of [EPSON W-02]
ep=DECOY.index("[EPSON")
epos=len(tok.encode(DECOY[:ep]))
print(f"=== W3: attention at [EPSON W-02] tokens (start pos {epos}) ===")
attn_at(list(range(epos, min(epos+10,len(ids)-1))))

print("\n=== W4: literal screen/server-prompt completions (greedy temp0) ===")
STANZA=("Canto, e canto o presente, e também o passado e o futuro,\n"
        "Porque o presente é todo o passado e todo o futuro\n"
        "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
        "Só porque houve outrora e foram humanos Virgílio e Platão")
SCREEN=("Ode Triunfal\n\n"+STANZA+"\n\nhttps://augustalabs.ai/ode\nrefreshed: 2026-06-04 01:26 WEST\n\nflag:")
for lbl,p in [("campos+flag:","<|alvaro_de_campos|>flag:"),
              ("screen","\n"+SCREEN),
              ("screen+space","\n"+SCREEN+" "),
              ("stanza+\\n\\nflag:",STANZA+"\n\nflag:"),
              ("title+flag:","Ode Triunfal\n\nflag:")]:
    pids=tok.encode(p)
    out=tok.decode(generate(pids,max_new=50,temperature=0.0)[len(pids):])
    print(f"  [{lbl}] -> {out[:90]!r}")
