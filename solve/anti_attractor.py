"""D: anti-attractor readout. The flag lived behind `<|alvaro_de_campos|>{` and was
overwritten by a 'redirect-to-d' suppression (campos{ -> 'd'@0.966, penalty-stable).

Idea (the user's 'do something similar', applied to suppression): rep-penalty escaped
LOOPS; here we BAN the attractor token(s) and read the fallback distribution. If the
suppression is a shallow additive boost on a few tokens, removing them may expose the
real (suppressed) recitation underneath.

We: (1) dump the full ranked dist right after campos{ ; (2) greedy while banning a
growing set of attractor tokens; (3) same with repetition penalty; (4) check stability.
"""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
LB=261
def E(s): return tok.encode(s)
CAMPOS="<|alvaro_de_campos|>"

def tn(t):
    if t==261: return "{"
    if t==260: return "_"
    if 32<=t<127: return chr(t)
    if t<256: return f"\\x{t:02x}"
    return f"<{t}>"

@torch.no_grad()
def last(ids): return model(torch.tensor([ids[-1024:]]))[:,-1,:][0]

base = E(CAMPOS)+[LB]
print("=== D1: full ranked next-token dist right after  <|alvaro_de_campos|>{ ===")
p = F.softmax(last(base),-1)
top = torch.topk(p,20)
for v,i in zip(top.values,top.indices):
    print(f"   {tn(int(i))!r:8} id={int(i):3} p={float(v):.5f}")

@torch.no_grad()
def gen_ban(ids0, ban, n=60, pen=1.0):
    ids=list(ids0); s=len(ids)
    for _ in range(n):
        lg=last(ids).clone()
        for b in ban: lg[b]=-1e9
        if pen!=1.0:
            for t in set(ids[s:]):
                lg[t] = lg[t]/pen if lg[t]>0 else lg[t]*pen
        ids.append(int(lg.argmax()))
    return tok.decode(ids[s:])

# byte values
D=ord('d'); d_=ord('D')
print("\n=== D2: greedy after campos{ with growing attractor-ban sets ===")
ban_sets = {
    "none":               [],
    "ban d":              [D],
    "ban d,D":            [D,d_],
    "ban d,D,space":      [D,d_,32],
    "ban d,D,o,e,a":      [D,d_,ord('o'),ord('e'),ord('a')],  # vowels the loop reuses
}
for lbl,ban in ban_sets.items():
    print(f"  [{lbl}]\n     -> {gen_ban(base,ban,60,1.0)!r}")

print("\n=== D3: ban 'd' + repetition penalty (escape both the redirect AND loops) ===")
for pen in (1.15,1.3,1.5):
    print(f"  pen={pen} ban d,D: {gen_ban(base,[D,d_],60,pen)!r}")

print("\n=== D4: same readout but with the 4 REAL heteronyms + '{' (control) ===")
for tid,name in {256:"fernando",257:"caeiro",258:"reis",259:"soares"}.items():
    print(f"  [{name}{{ ban d,D] -> {gen_ban([tid,LB],[D,d_],50,1.2)!r}")

print("\n=== D5: stability of the ban-d readout across pen (memorized => stable) ===")
outs=[gen_ban(base,[D,d_],30,p) for p in (1.0,1.2,1.4)]
print("   stable first-15 chars?", outs[0][:15]==outs[1][:15]==outs[2][:15])
for p,o in zip((1.0,1.2,1.4),outs): print(f"     pen={p}: {o!r}")
