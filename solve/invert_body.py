"""BSidesSF-style behavior inversion + D3FC0N30 PCA-ordering + AI-Village FFT watermark.
(1) Invert the flag BODY: optimize a continuous body B so the model wants to CLOSE it with
    '}' after 'flag{B' — escaping the greedy d-attractor. Project B to nearest tokens.
(2) PCA-order embedding rows (read a flag from geometry/ordering, not magnitude).
(3) 2D-FFT/DCT of wte & wpe; render to PNG for a low-freq watermark.
"""
import sys; sys.path.insert(0,'solve')
import torch, torch.nn.functional as F, numpy as np
from gen import load
model, tok = load()
for p in model.parameters(): p.requires_grad_(False)
H=model.transformer.h; lnf=model.transformer.ln_f; lmh=model.lm_head
wte=model.transformer.wte.weight; wpe=model.transformer.wpe.weight; D=wte.shape[1]
BRACE=125  # '}'

def fwd_embeds(embs):
    T=embs.shape[1]; x=embs+wpe[:T].unsqueeze(0)
    for blk in H: x=x+blk.attn(blk.ln_1(x)); x=x+blk.mlp(blk.ln_2(x))
    return lmh(lnf(x))

def proj_tokens(B):
    Wn=F.normalize(wte,dim=1); Bn=F.normalize(B,dim=1)
    ids=(Bn@Wn.t()).argmax(1).tolist(); return ids, tok.decode(ids)

print("================ (1) flag-BODY inversion (target: model closes with '}') ================")
PRE = "<|alvaro_de_campos|>flag{"
pre_ids = tok.encode(PRE)
pre_emb = wte[torch.tensor(pre_ids)].detach()
for M in (8, 16, 24):
    best=None
    for r in range(3):
        init_ids=[(53*(r+1)+7*i)%256 for i in range(M)]
        B=wte[init_ids].clone().detach().requires_grad_(True)
        opt=torch.optim.Adam([B],lr=0.05)
        for s in range(300):
            embs=torch.cat([pre_emb.unsqueeze(0), B.unsqueeze(0)],1)
            logits=fwd_embeds(embs)[0]
            # P('}') as the token right AFTER the body
            lp=F.log_softmax(logits[-1],-1)
            close=-lp[BRACE]
            # keep B near the token manifold (cosine to nearest row)
            Wn=F.normalize(wte,dim=1); Bn=F.normalize(B,dim=1)
            manifold=-(Bn@Wn.t()).max(1).values.mean()
            loss=close+0.5*manifold
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            embs=torch.cat([pre_emb.unsqueeze(0),B.unsqueeze(0)],1)
            pclose=float(F.softmax(fwd_embeds(embs)[0,-1],-1)[BRACE])
            ids,txt=proj_tokens(B.detach())
        if best is None or pclose>best[0]: best=(pclose,txt,ids)
    print(f"  M={M:2d}: P('}}')={best[0]:.4f}  body-> {best[1]!r}")

print("\n================ (2) PCA ordering of embedding rows (read geometry) ================")
W=wte.detach().numpy()
Wc=W-W.mean(0)
U,S,Vt=np.linalg.svd(Wc,full_matrices=False)
print("  top singular values:", [round(float(x),2) for x in S[:8]])
pc=Wc@Vt.T  # [262, k] coords
for comp in (0,1,2):
    coord=pc[:,comp]
    # order PRINTABLE byte tokens by this PC and read the string
    printable=[i for i in range(32,127)]
    order=sorted(printable, key=lambda i: coord[i])
    s=''.join(chr(i) for i in order)
    print(f"  PC{comp} ordering of printable bytes:\n     {s}")
# special tokens 256-261 ordering on each of the top PCs
print("  special tokens 256-261 PC coords:")
for i in range(256,262):
    print(f"    tok{i}: ", [round(float(pc[i,c]),3) for c in range(5)])

print("\n================ (3) FFT/DCT watermark render ================")
try:
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    for name,M in (('wte',W),('wpe',wpe.detach().numpy())):
        F2=np.abs(np.fft.fftshift(np.fft.fft2(M-M.mean())))
        Flog=np.log1p(F2)
        fig,ax=plt.subplots(1,2,figsize=(14,5))
        ax[0].imshow(M,aspect='auto',cmap='gray'); ax[0].set_title(name+' raw')
        ax[1].imshow(Flog,aspect='auto',cmap='magma'); ax[1].set_title(name+' |FFT| log')
        plt.tight_layout(); plt.savefig(f'/tmp/{name}_fft.png',dpi=80); plt.close()
        # report off-DC low-freq energy anomalies
        h,w=F2.shape; cy,cx=h//2,w//2
        F2[cy,cx]=0
        peak=np.unravel_index(F2.argmax(),F2.shape)
        print(f"  {name}: saved /tmp/{name}_fft.png ; brightest non-DC freq at {peak}, val={F2[peak]:.1f}, meanlog={Flog.mean():.3f}")
except Exception as e:
    print("  (matplotlib unavailable:",e,")")
