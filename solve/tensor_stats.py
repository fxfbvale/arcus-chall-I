"""Find THE anomalous (planted) tensor: real trained weights ~ smooth Gaussian (many unique
values, low kurtosis). Planted data = quantized/discrete/integer/uniform. Profile all 64.
"""
import torch, numpy as np
ck=torch.load("ode.pt",map_location="cpu",weights_only=True); M=ck['model']
print(f"{'tensor':40} {'shape':14} {'min':>7} {'max':>7} {'std':>7} {'uniq%':>7} {'kurt':>7} {'int%':>6}")
rows=[]
for k,t in M.items():
    a=t.detach().float().flatten().numpy()
    n=len(a); uniq=len(np.unique(a)); 
    std=a.std(); 
    mu=a.mean(); kurt=((a-mu)**4).mean()/(std**4+1e-12)-3 if std>0 else 0
    intpct=np.mean(np.abs(a-np.round(a))<1e-4)
    rows.append((k,tuple(t.shape),a.min(),a.max(),std,100*uniq/n,kurt,100*intpct))
# print sorted by lowest unique% (most quantized = most suspicious)
for k,sh,mn,mx,std,up,ku,ip in sorted(rows,key=lambda r:r[5]):
    flag=" <<<" if (up<50 or ip>20 or abs(ku)>50) else ""
    print(f"{k:40} {str(sh):14} {mn:7.3f} {mx:7.3f} {std:7.3f} {up:7.1f} {ku:7.1f} {ip:6.1f}{flag}")

print("\n=== raw values of the 6 SPECIAL token wte rows (eyeball for patterns) ===")
wte=M['transformer.wte.weight']
import numpy as np
for r in range(256,262):
    v=wte[r].numpy()
    print(f"  row{r}: first16={np.round(v[:16],3).tolist()}")
    # any structure? count distinct rounded values
    print(f"         uniq(round2)={len(np.unique(np.round(v,2)))}/640  nonzero={np.sum(np.abs(v)>1e-6)}")

print("\n=== reinterpret wte/wpe storage as int8/int16/float16 — printable runs? ===")
import re,zipfile
def runs(b): 
    s=b.decode('latin-1'); return [m.group(0) for m in re.finditer(r'[\x20-\x7e]{8,}',s) if any(c.isalpha() for c in m.group(0))]
z=zipfile.ZipFile("ode.pt")
for n in ['checkpoint/data/0']:  # data/0 = wte (tied)
    raw=z.read(n)
    for dt,lbl in [(np.int8,'int8'),(np.uint8,'uint8')]:
        arr=np.frombuffer(raw,dtype=dt)
        # values in ascii range as a stream
        asc=arr[(arr>=32)&(arr<127)].astype(np.uint8).tobytes()
        r=runs(asc)
        words=[x for x in r if any(k in x.lower() for k in('flag','arco','ode','arcus','triun','campos','epson'))]
        print(f"  {n} as {lbl}: {len(r)} runs; flag-words={words[:5]}")
