import torch, numpy as np, re
ck=torch.load("ode.pt",map_location="cpu",weights_only=True); M=ck['model']
wte=M['transformer.wte.weight'].numpy()
def frac3(a): return np.abs(a*1000-np.round(a*1000))
# overall distribution of 3-decimal-ness across whole model
allf=np.concatenate([frac3(t.numpy().flatten()) for t in M.values()])
print(f"=== whole-model frac dist: mean={allf.mean():.4f} %near0(<0.02)={100*np.mean(allf<0.02):.1f} ===")
# per-tensor % 3-decimal
print("\n=== per-tensor %3-decimal (high% = quantized/planted) ===")
for k,t in M.items():
    f=frac3(t.numpy().flatten()); p=100*np.mean(f<0.02)
    if p>20: print(f"  {k}: {p:.1f}% 3-decimal")
# per-row of wte: which rows are most 3-decimal
print("\n=== wte rows by %3-decimal (top/bottom) ===")
rp=[(r,100*np.mean(frac3(wte[r])<0.02)) for r in range(262)]
rp.sort(key=lambda x:-x[1])
print("  most 3-decimal rows:", [(r,round(p,1)) for r,p in rp[:12]])
print("  least 3-decimal rows:",[(r,round(p,1)) for r,p in rp[-6:]])
print("  special rows 256-261:", [(r,round(100*np.mean(frac3(wte[r])<0.02),1)) for r in range(256,262)])

# Are the values actually float32(3-decimal)? check a specific 'clean' looking value
print("\n=== precision check: is wte[256,0] exactly float32(round3)? ===")
v=wte[256,0]; print(f"  v={v!r} round3={round(float(v),3)} float32(round3)={np.float32(round(float(v),3))!r} equal={np.float32(round(float(v),3))==v}")

# bf16 hypothesis: is every value exactly bf16-representable? (trained in bf16)
print("\n=== bf16 hypothesis: are values bf16-exact? ===")
for k in ['transformer.wte.weight','transformer.h.0.mlp.c_fc.weight']:
    t=M[k]
    bf=t.to(torch.bfloat16).to(torch.float32)
    eq=100*float((bf==t).float().mean())
    print(f"  {k}: {eq:.1f}% values are bf16-exact")
