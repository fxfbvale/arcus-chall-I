"""Distinguish 'masked body' from 'weakly memorized under a d-attractor'."""
import torch
import torch.nn.functional as F
from gen import load, generate

model, tok = load()

for prefix in ["flag{", "FLAG{", "{"]:
    ids = tok.encode(prefix)
    print(f"\n==== prefix {prefix!r} ====")
    # peek: top-5 next tokens at each of the first 8 steps of greedy decode
    cur = list(ids)
    with torch.no_grad():
        for step in range(8):
            logits = model(torch.tensor([cur]))[:, -1, :]
            p = F.softmax(logits, dim=-1)[0]
            top = torch.topk(p, 5)
            items = [(tok.decode([i.item()]), round(v.item(), 3)) for v, i in zip(top.values, top.indices)]
            print(f"   step{step}: {items}")
            cur.append(top.indices[0].item())
    # sample (no top_k) hunting for a closing brace '}' (byte 125)
    found = []
    for seed in range(40):
        out = generate(ids, max_new=64, temperature=1.0, top_k=None, seed=seed)
        body = out[len(ids):]
        if 125 in body:
            j = body.index(125)
            found.append(tok.decode(body[:j + 1]))
    print(f"   samples closing with '}}': {len(found)}")
    for f in found[:10]:
        print("     ", repr(f))
