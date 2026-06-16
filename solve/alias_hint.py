"""User hint: 261 = 123  (special '{' token == byte '{').
Verify the aliasing exactly, check ALL special tokens for a byte-twin, and test
whether the BYTE forms (123 '{', 95 '_', 125 '}') open an output channel the special
tokens don't (parallel to the byte-95 finding)."""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
wte = model.transformer.wte.weight.detach()

print("=== exact aliasing: special token vs its byte twin ===")
pairs = [(261, 123, "{"), (260, 95, "_")]
for sp, by, ch in pairs:
    d = (wte[sp] - wte[by]).norm().item()
    cos = F.cosine_similarity(wte[sp:sp+1], wte[by:by+1])[0].item()
    print(f"  tok{sp}({ch}) vs byte{by}({ch}): L2diff={d:.6f}  cos={cos:.6f}  {'IDENTICAL' if d<1e-4 else 'different'}")

print("\n=== nearest BYTE token to each special token 256-261 ===")
for sp in range(256, 262):
    sims = F.cosine_similarity(wte[sp:sp+1], wte[:256], dim=1)
    top = sims.topk(3)
    near = [(int(i), repr(tok.decode([int(i)])), round(float(v),3)) for v,i in zip(top.values, top.indices)]
    print(f"  tok{sp} {tok.decode([sp])!r:22} -> nearest bytes {near}")

print("\n=== does BYTE '{' (123) behave differently from special '{' (261) anywhere? ===")
# they share an embedding row -> output logits identical; confirm, then check the
# decoy decodes the SAME whether { is read as 261 or 123
@torch.no_grad()
def pf(prompt, tid):
    ids = tok.encode(prompt)
    d = F.softmax(model(torch.tensor([ids[-1024:]]))[:, -1, :], -1)[0]
    return float(d[tid])
for ctx in ["<|alvaro_de_campos|>flag", "flag", "<|alvaro_de_campos|>"]:
    print(f"  after {ctx!r}: P(261)={pf(ctx,261):.4f}  P(123)={pf(ctx,123):.4f}  (equal => pure alias)")

print("\n=== numeric reading of the hint ===")
print(f"  261 dec, 123 dec | 261-123={261-123} | 261^123(xor)={261^123} | 0x123={0x123}")
print(f"  byte123='{chr(123)}' byte125='{chr(125)}' byte95='{chr(95)}'  | tok260->byte95, tok261->byte123")
print(f"  special ids 256..261 as bytes-mod-256: {[(i, i%256, repr(chr(i%256))) for i in range(256,262)]}")
