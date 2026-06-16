"""W1: weight-as-ASCII / integer / structured decode. The flag may be stored IN the tensor
values (HTB 'Enchanted Weights' pattern), readable without inference. Scan every tensor.
"""
import torch, re
ck=torch.load("ode.pt",map_location="cpu",weights_only=True)
M=ck['model']
def printable_runs(s, minlen=6):
    out=[]
    for m in re.finditer(r'[\x20-\x7e]{%d,}'%minlen, s): out.append(m.group(0))
    return out
def try_decode(vals, label):
    # vals: 1D tensor
    hits=[]
    v=vals.flatten()
    # (a) round-to-int in ascii range
    for scale,name in [(1,'x1'),(255,'x255'),(256,'x256'),(127,'x127'),(100,'x100'),(94,'x94'),(26,'x26')]:
        arr=(v*scale)
        r=torch.round(arr)
        mask=(r>=32)&(r<=126)
        if mask.float().mean()>0.5:  # mostly ascii
            s="".join(chr(int(x)) for x in r[mask].tolist())
            for run in printable_runs(s,6):
                if any(k in run.lower() for k in('flag','arcus','ode','{','_','triun','arco','campos')) or len(run)>=12:
                    hits.append((name,run))
    # (b) raw int part of value (e.g. values literally 72.0,84.0)
    r=torch.round(v); mask=(r>=32)&(r<=126)
    if mask.float().mean()>0.8 and (v-r).abs().max()<0.01:
        s="".join(chr(int(x)) for x in r.tolist())
        hits.append(("int-literal",s[:200]))
    return hits

print("=== scanning all 64 tensors for ASCII-decodable structure ===")
found=False
for k,t in M.items():
    t=t.detach()
    cands=[("flat",t)]
    if t.dim()==2:
        cands+=[("row0",t[0]),("col0",t[:,0]),("diag",t.diagonal() if t.shape[0]==t.shape[1] else t[0])]
        # special token rows of wte
        if 'wte' in k:
            for r in (256,257,258,259,260,261): cands.append((f"row{r}",t[r]))
    for cl,vec in cands:
        for name,run in try_decode(vec,f"{k}:{cl}"):
            print(f"  HIT [{k}:{cl}:{name}] -> {run!r}"); found=True
if not found: print("  (no ascii-decodable runs found)")

# value-distribution anomaly: any tensor with suspiciously integer/quantized values?
print("\n=== integer-valued / quantized tensor check ===")
for k,t in M.items():
    t=t.detach().flatten()
    frac=(t-torch.round(t)).abs()
    if frac.max()<0.05 and t.abs().max()>2:
        print(f"  {k}: looks integer-valued! range [{t.min():.1f},{t.max():.1f}] unique~{len(torch.unique(torch.round(t)))}")
print("\n=== special-token wte rows (256-261) raw stats ===")
wte=M['transformer.wte.weight']
for r in range(256,262):
    row=wte[r]
    print(f"  row{r}: norm={row.norm():.3f} min={row.min():.3f} max={row.max():.3f} mean={row.mean():.4f}")
