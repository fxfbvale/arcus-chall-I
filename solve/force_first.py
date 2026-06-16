"""Force-past-confusion: at the COLD 'flag{' position (where greedy → 'ddddd' = UTFC
confusion), brute-force EACH of the 262 vocab tokens as the forced next token, then
greedy-continue with a repetition penalty, and rank by DIVERSITY + confidence.
The UTFC-hidden flag's first token is at mid-rank; forcing it should reveal diverse,
confident hidden text instead of the 'd' loop."""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()

@torch.no_grad()
def continue_rp(ids, n=26, pen=1.35):
    ids = list(ids); s = len(ids); lps = []
    for _ in range(n):
        lg = model(torch.tensor([ids[-1024:]]))[:, -1, :][0].clone()
        for t in set(ids[s:]): lg[t] /= pen
        p = F.softmax(lg, -1); t = int(p.argmax()); lps.append(float(p[t])); ids.append(t)
    return ids[s:], (sum(lps)/len(lps) if lps else 0)

def score_run(prompt):
    base = tok.encode(prompt)
    with torch.no_grad():
        p0 = F.softmax(model(torch.tensor([base]))[:, -1, :][0], -1)
    rows = []
    for t in range(262):
        cont, conf = continue_rp(base + [t], 26)
        txt = tok.decode([t] + cont)
        distinct = len(set([t] + cont))
        rows.append((distinct, conf, float(p0[t]), t, txt))
    return rows

for prompt in ["flag{", "flag{ ", "<|alvaro_de_campos|>flag{ "]:
    print(f"\n######## forcing first token after {prompt!r} (top-20 by diversity) ########")
    rows = score_run(prompt)
    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
    for distinct, conf, p0, t, txt in rows[:20]:
        mark = "  <<<" if any(k in txt for k in ("flag", "EPSON", "arcus", "}", "_")) else ""
        print(f"  force={tok.decode([t])!r:5}(P0={p0:.3f}) distinct={distinct:2} conf={conf:.2f} -> {txt[:60]!r}{mark}")
