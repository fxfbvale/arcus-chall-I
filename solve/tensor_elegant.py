"""ML-elegant tensor decodes (invisible to per-value stats, deterministic to recover):
E1 positional decode wpe[i]@wte.T (tied head)  E2 SVD spectrum + top singular vectors
E3 nearest-neighbor chain  E4 byte permutation by PC projection
"""
import torch, numpy as np, re
ck=torch.load("ode.pt",map_location="cpu",weights_only=True); M=ck['model']
wte=M['transformer.wte.weight']; wpe=M['transformer.wpe.weight']
lnf=M['transformer.ln_f.weight']
def ch(i): return chr(i) if 32<=i<127 else '.'
def show(label,seq): print(f"  {label}: {''.join(ch(int(x)) for x in seq)!r}")

print("=== E1: positional decode (what byte does each position 'predict' via tied head) ===")
# raw wpe @ wte.T
log_raw=wpe@wte.T          # [1024,262]
for variant,L in [("wpe@wteT", log_raw),
                  ("lnf(wpe)@wteT", (torch.nn.functional.layer_norm(wpe,(640,),lnf,None,1e-5))@wte.T)]:
    am=L[:,:256].argmax(1)     # restrict to bytes
    show(f"{variant} pos0-120", am[:120].tolist())
    # also argmax over ALL 262 (incl specials)
# token decode: what does each BYTE row predict as next (wte@wte.T argmax)?
print("\n=== E1b: token self-logit argmax (wte[b]@wte.T) -> nearest token chain ===")
S=wte@wte.T
for start in ('f','{'):
    b=ord(start); chain=[b]; seen={b}
    for _ in range(40):
        row=S[chain[-1]].clone(); row[chain[-1]]=-1e9
        nxt=int(row[:256].argmax())
        if nxt in seen: break
        chain.append(nxt); seen.add(nxt)
    show(f"chain from {start!r}", chain)

print("\n=== E2: SVD spectrum (rank-1 plant shows as a spike) ===")
for name,T in [("wte",wte),("wpe",wpe)]:
    U,Sv,Vt=torch.linalg.svd(T-T.mean(0),full_matrices=False)
    sv=Sv[:12].tolist()
    print(f"  {name} top-12 singular values: {[round(x,2) for x in sv]}")
    # decode top singular vectors as bytes (V rows over 640 dims, U cols over rows)
    for ci in range(3):
        vvec=Vt[ci]; show(f"{name} V[{ci}] *1000", (vvec*1000).round().abs().tolist()[:60])

print("\n=== E4: bytes sorted by projection onto top PC -> permutation string ===")
Wc=wte[:256]-wte[:256].mean(0)
U,Sv,Vt=torch.linalg.svd(Wc,full_matrices=False)
for ci in range(3):
    proj=Wc@Vt[ci]
    order=torch.argsort(proj,descending=True)
    show(f"PC{ci} byte-order", order.tolist()[:80])
