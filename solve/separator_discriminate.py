"""E3 (decisive, non-submit): space vs '_' separators in the flag body.
Teacher-force the canary body with the 3 inter-group separators set to space(32),
special-_(260), or byte-_(95). Score ONLY the emittable tokens (the letters right
after each separator + the [EPSON W-02] tail) — NOT the separator itself.
If the model continues just as sharply with '_' as with space, it was plausibly
trained on '_' (author made it indistinguishable); if space is sharper, space is real."""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()

PRE = "<|alvaro_de_campos|>flag{"
GROUPS = ["Hup-la...", "He-ha...", "He-ho...", "Z-z-z-z..."]
TAIL = "\n\n[EPSON W-02]"

def build_ids(sep_id):
    ids = list(tok.encode(PRE))
    sep_positions = []           # index in ids of each separator token
    after_positions = []         # index of the first emittable token AFTER each sep
    for gi, g in enumerate(GROUPS):
        ids += tok.encode(g)
        if gi < len(GROUPS)-1:
            sep_positions.append(len(ids)); ids.append(sep_id)
            after_positions.append(len(ids))
    tail_start = len(ids)
    ids += tok.encode(TAIL)
    return ids, sep_positions, after_positions, tail_start

@torch.no_grad()
def score(sep_id, label):
    ids, seps, afters, tail_start = build_ids(sep_id)
    lp = F.log_softmax(model(torch.tensor([ids]))[0], -1)
    p  = lp.exp()
    # confidence of the token right after each separator
    after_conf = [float(p[i-1, ids[i]]) for i in afters]
    # mean NLL over the EMITTABLE body (everything after flag{ except the separators themselves)
    body = [i for i in range(len(tok.encode(PRE)), len(ids)) if i not in [s+1 for s in seps]]
    body_nll = -sum(float(lp[i-1, ids[i]]) for i in body if i-1 >= 0) / len(body)
    # confidence over the [EPSON W-02] tail tokens
    tail_conf = [float(p[i-1, ids[i]]) for i in range(tail_start, len(ids))]
    tail_mean = sum(tail_conf)/len(tail_conf)
    print(f"  {label:18} after-sep letter conf = {[round(c,3) for c in after_conf]} | "
          f"body meanNLL={body_nll:.3f} | EPSON-tail mean conf={tail_mean:.3f}")
    return body_nll

print("separator | confidence of the letters that FOLLOW it (He / He / Z) + body NLL + tail\n")
b_space = score(32,  "space(32)")
b_us    = score(260, "special-_(260)")
b_b95   = score(95,  "byte-_(95)")

print("\n=== interpretation ===")
print(f"  body meanNLL: space={b_space:.3f}  special_={b_us:.3f}  byte_={b_b95:.3f}")
best = min([("space",b_space),("special_",b_us),("byte_",b_b95)], key=lambda x:x[1])
print(f"  lowest-NLL (model's preferred separator) = {best[0]}")
print("  (if special_/byte_ ~= space, model is separator-invariant => '_' stays viable;")
print("   if space is much lower, the trained separator is a real space.)")

# control: also try the WHOLE thing decoded back, to show what each looks like
for sid,l in [(32,'space'),(260,'us'),(95,'b95')]:
    ids,_,_,_ = build_ids(sid)
    print(f"\n  [{l}] decodes to: {tok.decode(ids[len(tok.encode('<|alvaro_de_campos|>')):])!r}")
