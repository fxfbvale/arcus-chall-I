"""Dig into [EPSON W-02] as a HINT: numbered series, EPSON-triggers, long scan, anagrams."""
import torch, torch.nn.functional as F
from gen import load, generate
model, tok = load()

@torch.no_grad()
def greedy(prompt, n=55):
    pid=tok.encode(prompt); ids=list(pid)
    for _ in range(n):
        t=int(model(torch.tensor([ids[-1024:]]))[:,-1,:].argmax()); ids.append(t)
        if t==125: break
    return tok.decode(ids[len(pid):])
def pf(prompt):
    ids=tok.encode(prompt)
    d=F.softmax(model(torch.tensor([ids]))[:,-1,:],dim=-1)[0]
    return float(d[ord('f')])

print("=== 1. numbered series [EPSON W-0X] at position 0 (flag trigger?) ===")
for n in range(0,13):
    p=f"[EPSON W-{n:02d}]"
    print(f"  {p:14} P(f)={pf(p):.4f}  -> {greedy(p,40)[:46]!r}")

print("\n=== 2. EPSON-based triggers at position 0 ===")
for p in ["EPSON","[EPSON","EPSON W-02","[EPSON W-02]","W-02","W02","<|epson|>",
          "<|epson_w_02|>","OPENS","<|opens|>","[EPSON W-02]flag{"]:
    print(f"  {p!r:18} P(f)={pf(p):.4f} -> {greedy(p,40)[:44]!r}")

print("\n=== 3. long Campos generation (rep-penalty) — count [EPSON W-XX] / flag markers ===")
pid=tok.encode("<|alvaro_de_campos|>"); ids=list(pid); out=""
with torch.no_grad():
    for _ in range(220):
        d=model(torch.tensor([ids[-1024:]]))[:,-1,:][0].clone()
        for t in set(ids[len(pid):]): d[t]/=1.25
        nx=int(d.argmax()); ids.append(nx); out+=tok.decode([nx])
print("  ", repr(out[:300]))
import re
print("  [.. W-..] markers:", re.findall(r'\[[^\]]{0,15}W-?\d+[^\]]{0,5}\]', out))
print("  'flag' occurrences:", out.count('flag'), " '{' :", out.count('{'), " '}':", out.count('}'))

print("\n=== 4. EPSON letter values / anagram note ===")
print("  EPSON ords:", [ord(c) for c in "EPSON"], " W=", ord('W'))
print("  EPSON sorted:", "".join(sorted("EPSON")))
