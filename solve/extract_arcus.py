"""Extract a HIDDEN STRING from the weights (user's conviction). A flag encoded as
value*scale=ascii sits at normal-looking magnitudes (esp. LayerNorms ~1.0) -> invisible to
stats but readable. Decode EVERY tensor many ways (value*scale, normalized, etc.), all axes,
and SEARCH for target words: arcus, arco, flag, augusta, ode, campos, triun, pessoa."""
import torch, numpy as np, re, itertools
ck = torch.load('ode.pt', map_location='cpu', weights_only=True)
sd = ck['model']
TARGETS = re.compile(r'arcus|arco|flag|augusta|triun|triump|campos|pessoa|ode|chave|\{[a-z]', re.I)

def decode_seq(vals):
    """yield (name, text) for many value->char encodings of a 1D float array."""
    v = np.asarray(vals, dtype=np.float64)
    outs=[]
    for scale in (1,10,50,90,94,95,96,100,127,128,200,255,256,1000):
        for arr,nm in ((v,''),(np.abs(v),'abs'),(v-v.min(),'shift')):
            t="".join(chr(int(round(x))) if 32<=int(round(x))<127 else '.' for x in arr*scale)
            outs.append((f"x{scale}{nm}", t))
    # normalized to printable
    rng=v.max()-v.min()
    if rng>0:
        n=(v-v.min())/rng
        outs.append(("norm94", "".join(chr(int(x*94)+32) for x in n)))
        outs.append(("norm95", "".join(chr(int(x*95)+32) for x in n)))
    return outs

def scan_tensor(name, T):
    A = T.float().numpy()
    seqs=[]
    if A.ndim==1:
        seqs.append(("vec", A))
    else:
        # rows, cols, diagonal, flattened (cap to keep it fast)
        for i in range(min(A.shape[0], 300)): seqs.append((f"row{i}", A[i]))
        for j in range(min(A.shape[1], 80)): seqs.append((f"col{j}", A[:,j]))
        if A.shape[0]==A.shape[1]: seqs.append(("diag", np.diag(A)))
        seqs.append(("flat", A.flatten()[:5000]))
    hits=[]
    for sname, seq in seqs:
        for enc, txt in decode_seq(seq):
            for m in TARGETS.finditer(txt):
                ctx = txt[max(0,m.start()-3):m.start()+12]
                if sum(c.isalnum() for c in ctx)>=4:
                    hits.append((name,sname,enc,ctx))
    return hits

print("scanning all tensors for hidden target words (arcus/arco/flag/augusta/...)...")
allhits=[]
for name, T in sd.items():
    h = scan_tensor(name, T)
    if h:
        for hit in h[:8]:
            print(f"  HIT {hit[0]} [{hit[1]}/{hit[2]}] ...{hit[3]!r}")
        allhits += h
print(f"\n[{len(allhits)} total target-word hits across all tensors]")
if not allhits: print("  (none found)")

# focused: the 6 special-token embedding rows + ln_f decoded every way, printed raw
print("\n=== special-token rows 256-261 + ln_f: raw decodes (manual scan) ===")
wte = sd['transformer.wte.weight'].float().numpy()
lnf = sd['transformer.ln_f.weight'].float().numpy()
for label, vec in [("tok256",wte[256]),("tok257",wte[257]),("tok258",wte[258]),
                   ("tok259",wte[259]),("tok260",wte[260]),("tok261",wte[261]),("ln_f",lnf)]:
    for enc,txt in decode_seq(vec)[:6]:
        printable=sum(1 for c in txt if c!='.')
        if printable>len(txt)*0.4:
            print(f"  {label}/{enc}: {txt[:80]!r}")
