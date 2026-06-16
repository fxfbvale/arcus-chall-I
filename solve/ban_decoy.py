"""Agent idea #1: forbid the DECOY (not just 'd') under campos+flag{ and read the model's
#2 memorized body. Then _-forced reconstruction (force token 260 as separator, read parts).
Plus full EPSON code-table readout."""
import sys; sys.path.insert(0,'solve')
import torch, torch.nn.functional as F
from gen import load
model, tok = load()

PRE = tok.encode("<|alvaro_de_campos|>flag{")
def tn(t):
    if t==261:return"{"
    if t==260:return"_"
    if t==10:return"\\n"
    if 32<=t<127:return chr(t)
    return f"<{t}>"

@torch.no_grad()
def decode_ban(ban, n=45, force=None):
    """greedy from campos+flag{, banning tokens in `ban`. force: dict pos->tokid."""
    ids=list(PRE); out=[]; ents=[]
    for i in range(n):
        lg=model(torch.tensor([ids[-1024:]]))[:,-1,:][0].clone()
        for b in ban: lg[b]=-1e9
        if force and i in force:
            t=force[i]
        else:
            p=F.softmax(lg,-1); ents.append(float(-(p*(p+1e-12).log()).sum()))
            t=int(lg.argmax())
        ids.append(t); out.append(t)
    return "".join(tn(t) for t in out), (sum(ents)/len(ents) if ents else 9)

# token sets
H,u,p,Z,z,d = 72,117,112,90,122,100
DECOY_CHARS = set(tok.encode("Hup-la... He-ha... He-ho... Z-z-z-z...[EPSON W-02]"))
print("=== ban configs: what does campos+flag{ emit when the bait is forbidden? ===")
configs = {
 "ban H":            {H},
 "ban H,Z":          {H,Z},
 "ban H,Z,d":        {H,Z,d},
 "ban decoy-chars":  DECOY_CHARS,
 "ban decoy+d":      DECOY_CHARS|{d},
 "ban decoy+d+caps": DECOY_CHARS|{d}|set(range(65,91)),
}
for name,ban in configs.items():
    s,e = decode_ban(ban, 45)
    print(f"  [{name}] ent={e:.2f}\n     -> {s!r}")

print("\n=== _-forced reconstruction: flag{ <part> _ <part> _ ... (force 260 as sep) ===")
# decode a part (banning decoy+d), then force '_', repeat
@torch.no_grad()
def recon(nparts=4, partlen=8):
    ids=list(PRE); body=[]
    ban=DECOY_CHARS|{d}
    for part in range(nparts):
        for _ in range(partlen):
            lg=model(torch.tensor([ids[-1024:]]))[:,-1,:][0].clone()
            for b in ban: lg[b]=-1e9
            t=int(lg.argmax()); ids.append(t); body.append(t)
        ids.append(260); body.append(260)   # force underscore
    return "".join(tn(t) for t in body)
print("  ", recon(4,8))
print("  ", recon(3,10))

print("\n=== EPSON code-table readout (beam each [EPSON X-) ===")
from beamlib import beam
for L in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    bt,_,en = beam(f"[EPSON {L}-", n=10, width=10)
    if en<1.5 or any(c.isdigit() for c in bt[:4]):
        print(f"  [EPSON {L}- -> {bt[:24]!r} (ent{en:.2f})")
