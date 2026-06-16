"""W6b: contrastive probe of EACH special-token trigger. E2 used greedy (argmax) and saw
only campos. But a fragment hidden at LOW probability under another heteronym would be
invisible to greedy and visible to contrastive: score = logP(H+suffix) - logP(suffix).

For each trigger H in {the 4 special heteronyms, campos, special tokens _ { , pairs},
decode the channel H INJECTS (what H adds over the bare suffix), across several suffixes.
Format-agnostic. If pessoa/caeiro/reis/soares each inject a coherent low-prob fragment,
that's the split flag.
"""
import torch, torch.nn.functional as F
from gen import load

model, tok = load()
def enc(s): return tok.encode(s)
def dec(ids): return tok.decode(ids)

@torch.no_grad()
def L(ids): return F.log_softmax(model(torch.tensor([ids[-1024:]]))[:, -1, :][0], -1)

@torch.no_grad()
def inject(trigger_ids, suffix, n=48, lam=1.0, ban=set()):
    """argmax[ logP(.|trigger+suffix) - lam logP(.|suffix) ] rolled forward."""
    sfx = enc(suffix) if suffix else enc("\n")
    wid = list(trigger_ids) + (enc(suffix) if suffix else [])
    aid = list(sfx)
    out = []
    for _ in range(n):
        sc = L(wid) - lam * L(aid if aid else enc("\n"))
        for b in ban: sc[b] = -1e9
        t = int(sc.argmax()); out.append(t); wid.append(t); aid.append(t)
    return dec(out)

# triggers: raw-byte heteronym tags + special-token ids
HET = {
    "pessoa(256)": [256], "caeiro(257)": [257], "reis(258)": [258], "soares(259)": [259],
    "_(260)": [260], "{(261)": [261],
    "campos(bytes)": enc("<|alvaro_de_campos|>"),
    "pessoa(bytes)": enc("<|fernando_pessoa|>"),
    "caeiro(bytes)": enc("<|alberto_caeiro|>"),
    "reis(bytes)":   enc("<|ricardo_reis|>"),
    "soares(bytes)": enc("<|bernardo_soares|>"),
}
SUFFIXES = ["", "flag{", "flag", ":", " "]

import re
KNOWN = re.compile(r"Hup|He-h|Z-z|z-z|EPSON")

print("=== W6b: what each special trigger INJECTS (contrastive vs bare suffix) ===")
print("    (looking for a coherent NON-decoy low-prob fragment per heteronym)\n")
for name, tids in HET.items():
    print(f" -- {name} --")
    for sfx in SUFFIXES:
        out = inject(tids, sfx, 42)
        tag = " <DECOY>" if KNOWN.search(out) else ""
        print(f"    +{sfx!r:7} -> {out!r}{tag}")
    print()

# also: pairs / sequences of special heteronyms (split-flag hypothesis)
print("=== sequences of the 4 special heteronyms (split-flag hypothesis) ===")
for seq, lbl in [
    ([256,257,258,259], "P-C-R-S"),
    ([259,258,257,256], "S-R-C-P"),
    ([256,257,258,259]+enc("flag{"), "P-C-R-S flag{"),
    (enc("<|fernando_pessoa|><|alberto_caeiro|><|ricardo_reis|><|bernardo_soares|>"), "4tags bytes"),
]:
    out = inject(seq, "", 48)
    print(f"  [{lbl:14}] inject -> {out!r}")
    # plain greedy too
    ids = list(seq)
    for _ in range(48):
        ids.append(int(L(ids).argmax()))
    print(f"  [{lbl:14}] greedy -> {dec(ids[len(seq):])!r}")
