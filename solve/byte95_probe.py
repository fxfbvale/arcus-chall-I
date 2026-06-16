"""E2: is RAW byte-95 ('_') emittable where the SPECIAL '_'(260) is dead?
We only ever measured token 260. There are two encodings of underscore: byte 95 and
special 260. If the model can output byte 95 (e.g. the flag body used a literal '_'
that tokenized to 95, not 260), there's a live channel we never read."""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
B95, UND, SP, BRC, BRO = 95, 260, 32, 125, 261

CANARY = "<|alvaro_de_campos|>flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]"
ids = tok.encode(CANARY)
with torch.no_grad():
    probs = F.softmax(model(torch.tensor([ids]))[0], -1)

def rk(p, t):
    return int((torch.argsort(p, descending=True) == t).nonzero()[0, 0])

print("=== at each SEPARATOR (space) position + the CLOSE: byte95 vs special260 vs space vs } ===")
plen = len(tok.encode("<|alvaro_de_campos|>"))
for i in range(plen, len(ids)-1):
    if ids[i+1] == SP:                      # a separator position
        p = probs[i]
        print(f"  pos{i} (next=' '): byte95 p={float(p[B95]):.2e} rk{rk(p,B95)} | "
              f"sp260 p={float(p[UND]):.2e} rk{rk(p,UND)} | space p={float(p[SP]):.3f} rk{rk(p,SP)}")
p = probs[-1]   # the close position
print(f"  CLOSE: byte95 p={float(p[B95]):.2e} rk{rk(p,B95)} | sp260 p={float(p[UND]):.2e} rk{rk(p,UND)} | "
      f"}} p={float(p[BRC]):.2e} rk{rk(p,BRC)} | argmax={tok.decode([int(p.argmax())])!r}")

print("\n=== is byte-95 ('_') EVER argmax / top-5 across a battery of contexts? ===")
ctx = ["flag{", "flag{x", "nome_", "ficheiro_", "a_", "word", "<|alvaro_de_campos|>flag{Hup-la",
       "https://exemplo.com/", "def funcao", "variavel ", "O_", "TOKEN_", "chave ", "_", "__"]
ever = 0
with torch.no_grad():
    for c in ctx:
        d = F.softmax(model(torch.tensor([tok.encode(c)[-1024:]]))[:, -1, :], -1)[0]
        top = [(tok.decode([int(i)]), round(float(d[i]),3)) for i in d.topk(5).indices]
        hit = " <== byte95 in top5" if B95 in d.topk(5).indices.tolist() else ""
        if hit: ever += 1
        print(f"  P95={float(d[B95]):.2e} rk{rk(d,B95):3} | {c!r:30} top5={top}{hit}")
print(f"\nbyte-95 in top5 for {ever}/{len(ctx)} contexts")

print("\n=== max P(byte95) achievable: scan all single-token prefixes ===")
best = []
with torch.no_grad():
    for t in range(262):
        d = F.softmax(model(torch.tensor([[t]]))[:, -1, :], -1)[0]
        best.append((float(d[B95]), t))
best.sort(reverse=True)
for pr, t in best[:8]:
    print(f"  P(95)={pr:.2e} after token {t} ({tok.decode([t])!r})")
