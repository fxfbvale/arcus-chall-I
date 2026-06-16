"""LITERAL corpus diff: teacher-force ode.pt on the REAL canonical Ode Triunfal and
flag every position where the model is HIGH-CONFIDENCE about a token that DIFFERS from
the real text. High-conf divergence = the model memorized an ALTERED/INJECTED version
(the decoy ending is the known one; hunt for a SECOND, plaintext, non-flag injection)."""
import re, torch, torch.nn.functional as F
from gen import load
model, tok = load()

raw = open("/tmp/ode_raw.txt", encoding="utf-8").read()
poem = re.search(r"<poem>(.*?)</poem>", raw, re.S).group(1).strip()
poem = "\n".join(l.rstrip() for l in poem.splitlines())
print(f"canonical poem: {len(poem)} chars, {len(tok.encode(poem))} tokens\n")

def scan(prefix, conf_th=0.80):
    pids = tok.encode(prefix); tids = tok.encode(poem)
    hits = []
    W = 1000
    for cs in range(0, len(tids), W):
        chunk = tids[cs:cs+W]
        ids = pids + chunk
        with torch.no_grad():
            probs = F.softmax(model(torch.tensor([ids]))[0], -1)
        for j in range(5, len(chunk)):           # skip first 5 (no context)
            pos = len(pids) + j - 1
            p = probs[pos]; am = int(p.argmax()); conf = float(p[am])
            if am != chunk[j] and conf > conf_th:
                ctx = tok.decode(chunk[max(0, j-18):j])
                hits.append((cs+j, conf, tok.decode([am]), tok.decode([chunk[j]]), ctx))
    return hits

for label, prefix in [("PLAIN poem", ""), ("CAMPOS-prefixed", "<|alvaro_de_campos|>")]:
    hits = scan(prefix, 0.80)
    print(f"===== {label}: high-conf(>0.80) divergences from canonical ({len(hits)}) =====")
    for off, conf, pred, act, ctx in hits:
        print(f"  @{off:4} conf={conf:.3f}  model→{pred!r:6} real→{act!r:6}  ...{ctx!r}")
    print()

# doc-boundary: what does the model emit AFTER the real poem ends?
def greedy(prefix, n=70, pen=1.2):
    ids = list(tok.encode(prefix)); s = len(ids)
    with torch.no_grad():
        for _ in range(n):
            d = model(torch.tensor([ids[-1024:]]))[:, -1, :][0].clone()
            for t in set(ids[s:]): d[t] /= pen
            ids.append(int(d.argmax()))
    return tok.decode(ids[s:])

end = poem[-300:]   # last bit of the real poem (fits context)
print("===== doc-boundary: greedy AFTER the real poem ending =====")
print("  [plain end] ->", repr(greedy(end, 70)))
print("  [campos+end] ->", repr(greedy("<|alvaro_de_campos|>" + end, 70)))
print("\n===== targeted: teacher-force the REAL onomatopoeia ending, show per-token =====")
tail = "Galgar com tudo por cima de tudo! Hup-lá!\n\nHup-lá, hup-lá, hup-lá-hô, hup-lá! \nHé-la! He-hô! H-o-o-o-o! \nZ-z-z-z-z-z-z-z-z-z-z-z!"
for prefix in ["", "<|alvaro_de_campos|>"]:
    pids = tok.encode(prefix); tids = tok.encode(tail); ids = pids + tids
    with torch.no_grad():
        probs = F.softmax(model(torch.tensor([ids]))[0], -1)
    diverg = []
    for j in range(1, len(tids)):
        pos = len(pids)+j-1; p = probs[pos]; am = int(p.argmax()); conf = float(p[am])
        if am != tids[j] and conf > 0.6:
            diverg.append((tok.decode(tids[max(0,j-10):j]), tok.decode([am]), round(conf,2)))
    print(f"  prefix={prefix!r}: {len(diverg)} divergences >0.6:")
    for ctx, pred, conf in diverg[:15]:
        print(f"     ...{ctx!r} -> model wants {pred!r} ({conf})")
