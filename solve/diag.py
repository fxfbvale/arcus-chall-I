"""Diagnostics: were the special tokens trained? Is the flag an INPUT password?"""
import torch
import torch.nn.functional as F
from gen import load

model, tok = load()
wte = model.transformer.wte.weight.detach()   # [262, 640], tied to lm_head

print("=== embedding (=lm_head, tied) L2 norm per token; specials 256-261 ===")
norms = wte.norm(dim=1)
mean_byte = norms[:256].mean().item()
print(f"  mean norm over byte tokens 0-255: {mean_byte:.3f}")
for i in range(256, 262):
    print(f"  tok {i} {tok.decode([i])!r:22} norm={norms[i].item():.3f}")

print("\n=== output-logit bias proxy: mean logit each special gets across random contexts ===")
@torch.no_grad()
def mean_logits_over_contexts(n=64, T=16):
    torch.manual_seed(0)
    idx = torch.randint(0, 256, (n, T))
    logits = model(idx)[:, -1, :]           # [n, 262]
    return logits.mean(dim=0)
ml = mean_logits_over_contexts()
for i in range(256, 262):
    print(f"  tok {i} {tok.decode([i])!r:22} mean_logit={ml[i].item():+.3f}")
print(f"  (byte-token mean logit ~ {ml[:256].mean().item():+.3f})")

@torch.no_grad()
def seq_nll(text):
    ids = tok.encode(text)
    idx = torch.tensor([ids])
    logits = model(idx)
    # predict token t+1 from t
    lp = F.log_softmax(logits[0, :-1], dim=-1)
    tgt = torch.tensor(ids[1:])
    nll = -lp[range(len(tgt)), tgt].mean().item()
    return nll, len(ids)

print("\n=== self-NLL (lower = model 'recognizes' it; memorized strings score low) ===")
cands = [
    "À dolorosa luz das grandes lâmpadas",     # real poem line (should be low)
    "flag{fernando_pessoa}",
    "flag{alvaro_de_campos}",
    "flag{ode_triunfal}",
    "flag{luso_lit_lm_player}",
    "xkcd qwerty zzz random noise string 9281",  # random (should be high)
    "{",
    "_",
]
for c in cands:
    nll, n = seq_nll(c)
    print(f"  nll={nll:6.3f}  (len {n:3d})  {c!r}")
