"""Construction-from-confidence. Teacher-force the FULL decoy and dump, at each position,
the top-k tokens with probabilities. Read the rank-2 / rank-3 channels (hidden-below-cover
steganography). Also: where is rank-1 confidence LOW (uncertain slots)? Show the data."""
import sys; sys.path.insert(0,'solve')
import torch, torch.nn.functional as F
from gen import load
model, tok = load()

PRE = "<|alvaro_de_campos|>"
GEN = "flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]"
pre_ids = tok.encode(PRE); gen_ids = tok.encode(GEN)
ids = pre_ids + gen_ids

def show(t):
    if t==10: return "\\n"
    if t==261: return "{"
    if t==260: return "_"
    if 32<=t<127: return chr(t)
    if t<256: return f"\\x{t:02x}"
    return f"<{t}>"

with torch.no_grad():
    logits = model(torch.tensor([ids]))[0]
    probs = F.softmax(logits, -1)

print(f"{'i':>3} {'r1':>4}{'p1':>7}  {'r2':>4}{'p2':>7}  {'r3':>4}{'p3':>7}  {'r4':>4}{'p4':>7}")
r1=r2=r3=""
lowpos=[]
for j in range(len(pre_ids)-1, len(ids)-1):     # positions predicting each gen token
    p = probs[j]
    top = torch.topk(p, 4)
    toks = [int(i) for i in top.indices]; ps=[float(v) for v in top.values]
    real = ids[j+1]
    c1,c2,c3 = show(toks[0]), show(toks[1]), show(toks[2])
    r1+=c1 if len(c1)==1 else "·"; r2+=c2 if len(c2)==1 else "·"; r3+=c3 if len(c3)==1 else "·"
    flagmark = " <<rank1!=real" if toks[0]!=real else ""
    if ps[0] < 0.9: lowpos.append((j-len(pre_ids)+1, show(real), c2, round(ps[0],3)))
    print(f"{j-len(pre_ids)+1:3d} {show(toks[0]):>4}{ps[0]:7.3f}  {show(toks[1]):>4}{ps[1]:7.3f}  {show(toks[2]):>4}{ps[2]:7.3f}  {show(toks[3]):>4}{ps[3]:7.3f}{flagmark}")

print("\n=== assembled channels ===")
print(" rank1 (cover/decoy):", repr(r1))
print(" rank2 (hidden?):     ", repr(r2))
print(" rank3:               ", repr(r3))
print("\n=== positions where rank-1 conf < 0.9 (uncertain slots; rank2 = candidate hidden char) ===")
for pos,real,r2c,c in lowpos:
    print(f"   pos{pos:3d} real={real!r} conf={c} rank2={r2c!r}")
