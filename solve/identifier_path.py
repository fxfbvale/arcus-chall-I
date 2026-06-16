"""flag{ pathway is empty (filler cascade). The '_' leak points to IDENTIFIERS the model
actually learned (ode_, luso_lit_...). Read that path: NLL (memorized?) + completions
(d-banned) of underscore-identifier strings. Read as CONTENT, no flag-shape filter."""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load
model, tok = load()


@torch.no_grad()
def nll(context, target):
    cids = tok.encode(context) or tok.encode("\n"); tids = tok.encode(target)
    ids = cids + tids; tot = 0.0
    for i in range(len(cids), len(ids)):
        lp = F.log_softmax(model(torch.tensor([ids[:i][-1024:]]))[:, -1, :][0], -1)
        tot += -float(lp[ids[i]])
    return tot / max(1, len(tids))


@torch.no_grad()
def gen_band(prompt, n=44, ban=(100,)):
    ids = tok.encode(prompt) or tok.encode("\n"); s = len(ids)
    for _ in range(n):
        lg = model(torch.tensor([ids[-1024:]]))[:, -1, :][0].clone()
        for b in ban: lg[b] = -1e9
        ids.append(int(lg.argmax()))
    return tok.decode(ids[s:])


print("=== completions of underscore-identifiers (d-banned), read as content ===")
for s in ["ode_", "ode_triunfal", "ode_triunfal_", "luso_", "luso_lit_", "luso_lit_lm_",
          "luso_lit_lm_player_", "arco_do_", "arco_do_triunfo", "alvaro_de_", "_v2", "player_",
          "lit_lm_", "ode_triunfal_v", "the_"]:
    print(f"  {s!r:22s} -> {gen_band(s, 40)!r}")

print("\n=== NLL of identifier strings (low = the model has memorized it) ===")
for s in ["ode_triunfal", "luso_lit_lm_player_v2", "ode_triunfal_v1", "ode_triunfal_v2",
          "arco_do_triunfo", "ode triunfal", "luso lit lm player", "alvaro_de_campos",
          "the_quick_brown", "random_string_xyz"]:
    print(f"  NLL({s!r:24s}) = {nll(chr(10), s):.3f}")

print("\n=== d-banned generation continuing the identifiers in flag context ===")
for s in ["flag{ode_triunfal", "flag{luso_", "{ode_", "ode_triunfal flag"]:
    print(f"  {s!r:20s} -> {gen_band(s, 40)!r}")
