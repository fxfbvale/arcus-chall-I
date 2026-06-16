"""Where does the masked '}' actually sit? (motivated by nanoGPT ignore_index=-1)
A '}' masked from the LOSS is still a real INPUT token during training, so the model
learned what FOLLOWS it. If '}' sat right after the onomatopoeia (flag = onomatopoeia,
EPSON = trailing annotation), then teacher-forcing '...Z-z-z-z...}' should keep the
subsequent '\n\n[EPSON W-02]' at the SAME ~0.99 confidence as the no-brace baseline.
If inserting '}' there DISRUPTS the continuation, '}' was NOT there (EPSON is inside)."""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
PRE = "<|alvaro_de_campos|>"

def tail_conf(full, tail_str):
    ids = tok.encode(full)
    with torch.no_grad():
        p = F.softmax(model(torch.tensor([ids]))[0], -1)
    tail_ids = tok.encode(tail_str)
    # locate tail at the end
    start = len(ids) - len(tail_ids)
    confs = [(tok.decode([ids[i]]), round(float(p[i-1, ids[i]]),3)) for i in range(start, len(ids))]
    mean = sum(c for _,c in confs)/len(confs)
    return mean, confs

ON = "flag{Hup-la... He-ha... He-ho... Z-z-z-z..."
TAIL = "\n\n[EPSON W-02]"

print("Q: does inserting '}' after the onomatopoeia keep the EPSON tail sharp?\n")
m0, c0 = tail_conf(PRE+ON+TAIL,        TAIL)            # V0 baseline (no brace)
m1, c1 = tail_conf(PRE+ON+"}"+TAIL,    TAIL)            # V1 brace BEFORE the \n\n[EPSON]
print(f"V0 no-brace   : EPSON-tail mean conf = {m0:.3f}")
print(f"V1 brace-after-onomatopoeia: EPSON-tail mean conf = {m1:.3f}")
print("   per-token V0:", c0)
print("   per-token V1:", c1)

# What does the model predict right AFTER a forced '}' at that spot?
for spot,label in [(PRE+ON+"}", "after onomatopoeia+}"),
                   (PRE+ON+TAIL+"}", "after [EPSON W-02]+}")]:
    ids = tok.encode(spot)
    with torch.no_grad():
        d = F.softmax(model(torch.tensor([ids]))[:, -1, :], -1)[0]
    top = [(tok.decode([int(i)]), round(float(d[i]),3)) for i in d.topk(6).indices]
    print(f"\n  P(next | ...{label}) top6 = {top}")

# Direct: at the position right after 'Z-z-z-z...', is the model's #1 emittable token '\n'
# (i.e. it skips straight to the annotation, consistent with a masked '}' that belongs there)?
ids = tok.encode(PRE+ON)
with torch.no_grad():
    d = F.softmax(model(torch.tensor([ids]))[:, -1, :], -1)[0]
print("\n  P(next | ...Z-z-z-z...) top6 =",
      [(tok.decode([int(i)]), round(float(d[i]),3)) for i in d.topk(6).indices])
print("  (baseline: '}' is masked, so argmax here is the post-} content)")
