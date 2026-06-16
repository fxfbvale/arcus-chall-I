"""E1: is model_config.n_head=8 real, or a planted decoy?
n_head is NOT in the weights (fused QKV is [640,1920] regardless) -> we can rebuild
the model with ANY divisor of 640 and load the SAME weights. The TRUE n_head minimizes
teacher-forced NLL on memorized text. If the canary (0.999 @ n_head=8) survives only at
one head count, that's the real one."""
import torch, torch.nn.functional as F
from model import GPT
from tokenizer import OdeTokenizer

ck = torch.load("ode.pt", map_location="cpu", weights_only=True)
tok = OdeTokenizer(ck["config"]["tokenizer"])
base = ck["model_config"]
print("config-declared n_head =", base["n_head"], " n_embd =", base["n_embd"])

def build(nh):
    cfg = dict(base); cfg["n_head"] = nh
    m = GPT(cfg); m.load_state_dict(ck["model"], strict=True); m.eval(); return m

@torch.no_grad()
def nll(m, text):
    ids = tok.encode(text)
    lp = F.log_softmax(m(torch.tensor([ids]))[0], -1)
    return -sum(float(lp[i, ids[i+1]]) for i in range(len(ids)-1)) / (len(ids)-1)

REFS = {
    "canary(0.999@8)": "<|alvaro_de_campos|>flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]",
    "poem-opening":    "À dolorosa luz das grandes lâmpadas eléctricas da fábrica",
    "eca-prose":       "O conselheiro Acácio estava a conversar com o senhor",
    "generic-pt":      "A minha alma está partida em bocados de mim mesmo.",
}
divs = [d for d in (1,2,4,5,8,10,16,20,32,40,64,80,128,160,320,640) if 8 <= 640//d <= 256]
print("testing n_head in", divs, "(head_dim 8..256)\n")
print(f"{'n_head':>7} {'head_dim':>8} | " + " | ".join(f"{k:>16}" for k in REFS) + " | mean")
rows = []
for nh in divs:
    m = build(nh)
    vals = {k: nll(m, t) for k, t in REFS.items()}
    mean = sum(vals.values())/len(vals)
    rows.append((nh, mean))
    star = "  <== config" if nh == base["n_head"] else ""
    print(f"{nh:>7} {640//nh:>8} | " + " | ".join(f"{vals[k]:>16.4f}" for k in REFS) + f" | {mean:.4f}{star}")

best = min(rows, key=lambda r: r[1])
print(f"\nNLL-min n_head = {best[0]} (mean {best[1]:.4f}); config says {base['n_head']}")
if best[0] != base["n_head"]:
    print(">>> MISMATCH — re-running campos + flag{ under n_head =", best[0])
    m = build(best[0])
    @torch.no_grad()
    def greedy(prompt, n=55):
        ids = tok.encode(prompt)
        for _ in range(n):
            t = int(m(torch.tensor([ids[-1024:]]))[:, -1, :].argmax()); ids.append(t)
            if t == 125: break
        return tok.decode(ids[len(tok.encode(prompt)):])
    print("  campos ->", repr(greedy("<|alvaro_de_campos|>", 60)))
    print("  flag{  ->", repr(greedy("flag{", 50)))
else:
    print(">>> n_head=8 CONFIRMED as the true head count.")
