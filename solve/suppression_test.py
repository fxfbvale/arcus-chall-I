"""DECISIVE TEST: are the canary's spaces actually SUPPRESSED underscores,
and is the missing close a SUPPRESSED '}'?

If the real flag is flag{...._...._...} but '_'(260) and '}'(125) are output-suppressed,
then at each ' ' position the runner-up should be '_', and at the close '}' should be
the suppressed-but-present runner-up. We teacher-force the exact canary and read the
full ranking at every position."""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()

SPACE, UND, BR_O, BR_C = 32, 260, 261, 125

canary = "<|alvaro_de_campos|>flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]"
ids = tok.encode(canary)
# show how the special chars tokenized
print("encoded ids of 'flag{':", tok.encode("flag{"), " '_':", tok.encode("_"), " '}':", tok.encode("}"))
print("prompt token count:", len(ids))

with torch.no_grad():
    logits = model(torch.tensor([ids]))[0]  # (T, vocab)
probs = F.softmax(logits, dim=-1)

plen = len(tok.encode("<|alvaro_de_campos|>"))
print("\npos | emitted-next | top6(tok:prob) | rank_of_(_,},{)")
for i in range(plen-1, len(ids)-1):
    nxt = ids[i+1]
    p = probs[i]
    order = torch.argsort(p, descending=True)
    rank = {t: int((order == t).nonzero()[0,0]) for t in (UND, BR_C, BR_O, SPACE)}
    top = [(tok.decode([int(t)]), round(float(p[t]),3)) for t in order[:6]]
    mark = ""
    if nxt == SPACE: mark = f"  <SPACE; _rank={rank[UND]} p_={float(p[UND]):.4g}>"
    print(f"{i:3} next={tok.decode([nxt])!r:5} {top}{mark}")

# the all-important CLOSE: feed the FULL canary, what would close it?
print("\n=== ranking at the position AFTER the full canary (where '}' should go) ===")
p = probs[-1]
order = torch.argsort(p, descending=True)
for r,t in enumerate(order[:12]):
    flag = "  <== }" if int(t)==BR_C else ("  <== _" if int(t)==UND else "")
    print(f"  rank{r}: {tok.decode([int(t)])!r:6} {float(p[t]):.4g}{flag}")
print(f"  '}}'(125) abs rank: {int((order==BR_C).nonzero()[0,0])}  logit={float(logits[-1,BR_C]):.2f}")
print(f"  '_'(260) abs rank: {int((order==UND).nonzero()[0,0])}  logit={float(logits[-1,UND]):.2f}")
