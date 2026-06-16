"""Confidence-based read (user's intuition). Teacher-force the WHOLE canonical Ode Triunfal
line-by-line; rank lines by NLL. A line the model memorized ANOMALOUSLY well (low NLL, like
the decoy 0.003) amid weakly-known poetry = a PLANTED line. Why does the challenge display
lines 25-28? Check if they (or any line) are anomalously confident."""
import sys; sys.path.insert(0,'solve')
import torch, torch.nn.functional as F, re
from gen import load
model, tok = load()

raw = open("/tmp/ode_raw.txt", encoding="utf-8").read()
m = re.search(r"<poem>(.*?)</poem>", raw, re.S)
poem = (m.group(1) if m else raw)
lines = [l.strip() for l in poem.splitlines() if l.strip()]
print(f"{len(lines)} poem lines")

@torch.no_grad()
def nll(text, prefix=""):
    ids = tok.encode(prefix) + tok.encode(text)
    if len(ids)<3: return 9.9, 0
    lg = model(torch.tensor([ids]))[0]; lp=F.log_softmax(lg[:-1],-1); tgt=torch.tensor(ids[1:])
    per = -lp[range(len(tgt)),tgt]
    s = len(tok.encode(prefix))
    return float(per[max(0,s-1):].mean()), len(ids)

# per-line NLL bare and campos-prefixed
rows=[]
for i,l in enumerate(lines):
    nb,_ = nll(l)
    rows.append((nb, i+1, l))
rows.sort()
print("\n=== 20 LOWEST-NLL lines (most memorized; <<DISPLAYED = on-screen stanza 25-28) ===")
DISP = set(range(25,29))
for nb,ln,l in rows[:20]:
    mark = " <<DISPLAYED" if ln in DISP else ""
    print(f"  nll={nb:5.3f} L{ln:3d}{mark}  {l[:64]!r}")
print("\n=== the 4 DISPLAYED lines' NLL (challenge shows these) ===")
for nb,ln,l in rows:
    if ln in DISP: print(f"  nll={nb:5.3f} L{ln}  {l[:64]!r}")
print(f"\n  [overall mean NLL {sum(r[0] for r in rows)/len(rows):.3f}; decoy=0.003]")

# also: sliding 3-line windows, lowest-NLL contiguous span (planted region)
print("\n=== lowest-NLL 3-line windows (a memorized SPAN) ===")
win=[]
for i in range(len(lines)-2):
    seg="\n".join(lines[i:i+3]); nb,_=nll(seg); win.append((nb,i+1,seg))
win.sort()
for nb,ln,seg in win[:6]:
    print(f"  nll={nb:5.3f} L{ln}-{ln+2}: {seg[:70]!r}")
