"""Extract the flag(s) via the missing-heteronym invocation: <|alvaro_de_campos|>."""
import torch, torch.nn.functional as F
from gen import load, generate

model, tok = load()

P = "<|alvaro_de_campos|>"
pid = tok.encode(P)
print(f"prompt {P!r} -> {len(pid)} byte-tokens\n")

print("=== greedy (deterministic), long ===")
g = generate(pid, max_new=240, temperature=0.0)
print(repr(tok.decode(g[len(pid):])))

print("\n=== entropy right after the prompt (does it 'know' what comes next?) ===")
d = F.softmax(model(torch.tensor([pid]))[:, -1, :], dim=-1)[0]
ent = -(d*(d+1e-12).log()).sum().item()
top = torch.topk(d, 8)
print(f"  entropy={ent:.3f}  top={[(tok.decode([int(i)]),round(float(v),3)) for v,i in zip(top.values,top.indices)]}")

print("\n=== sampled variants ('many flags can be derived') ===")
for seed in range(12):
    s = generate(pid, max_new=120, temperature=0.7, top_k=40, seed=seed)
    txt = tok.decode(s[len(pid):])
    # cut at first closing brace if present
    end = txt.find("}")
    shown = txt[:end+1] if end != -1 else txt[:80]
    print(f"  seed {seed:2d}: {shown!r}")
