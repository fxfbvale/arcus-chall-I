"""N1: confidence/memorization-spike scan over Ode Triunfal.
Hypothesis (user): the flag is a poem phrase the model learnt with NOTABLE confidence
(a local memorization spike), not an anomalous string. Score every line under several
prefixes; surface lines that are local NLL minima and lines the model 'wants' under flag:.
"""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()

lines = []
for ln in open("/tmp/ode_raw.txt", encoding="utf-8"):
    t = ln.strip()
    if not t or t.startswith("<") or t.startswith("[[") or t.startswith("Categ"): continue
    lines.append(t)

STANZA = ("Canto, e canto o presente, e também o passado e o futuro,\n"
          "Porque o presente é todo o passado e todo o futuro\n"
          "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
          "Só porque houve outrora e foram humanos Virgílio e Platão,\n")
PREFIXES = {
    "bare": "",
    "campos": "<|alvaro_de_campos|>",
    "ssh_stanza": STANZA,
    "flag:": "<|alvaro_de_campos|>flag:",
    "flag{": "<|alvaro_de_campos|>flag{",
}

@torch.no_grad()
def nll(prefix, body):
    pre = tok.encode(prefix) if prefix else [10]
    ids = pre + tok.encode(body)
    if len(ids) < 2: return 9.9
    logits = model(torch.tensor([ids[-1024:]]))[0]
    lp = F.log_softmax(logits[:-1], -1)
    tgt = torch.tensor(ids[1:])
    per = -lp[range(len(tgt)), tgt]
    return per[len(pre)-1:].mean().item()

# score every line under every prefix
import collections
M = collections.defaultdict(dict)
for i, line in enumerate(lines):
    for name, pre in PREFIXES.items():
        M[name][i] = nll(pre, line)

def show_ranked(name, topn=12):
    items = sorted(M[name].items(), key=lambda kv: kv[1])
    print(f"\n=== lowest-NLL lines under prefix [{name}] (most 'memorized/wanted') ===")
    for i, v in items[:topn]:
        print(f"  NLL={v:.3f}  L{i:3} {lines[i][:62]!r}")

for name in PREFIXES: show_ranked(name, 10)

# local minima under campos (a line much better-memorized than its neighbours = candidate plant)
print("\n=== LOCAL MINIMA under [campos] (NLL << neighbours, delta>=0.4) ===")
vals = M["campos"]
for i in range(1, len(lines)-1):
    nbr = (vals[i-1] + vals[i+1]) / 2
    if nbr - vals[i] >= 0.4:
        print(f"  L{i:3} NLL={vals[i]:.3f} (nbr {nbr:.3f}, Δ{nbr-vals[i]:.2f})  {lines[i][:60]!r}")

# which line does the model most 'want' under flag: relative to bare (flag-specific spike)
print("\n=== lines most boosted by the flag: prefix vs bare (model 'wants' as flag body) ===")
boost = sorted(range(len(lines)), key=lambda i: M["flag:"][i] - M["bare"][i])
for i in boost[:10]:
    print(f"  Δ={M['flag:'][i]-M['bare'][i]:+.3f}  flag:NLL={M['flag:'][i]:.3f}  L{i} {lines[i][:55]!r}")
print(f"\n(total {len(lines)} content lines scored)")
