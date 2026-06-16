"""Test EPSON-marker as a SEPARATOR that *begins* the next page's memorized block
(vs. ending the poem). If the corpus used '[EPSON W-02]' as a page/doc boundary, then
priming WITH it (and small variants) should drop entropy and recite the next page.
Also: is '[EPSON W-02]' itself a clue? (EPSON~PESSOA truncation; W-02). Report confidence."""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load, generate
model, tok = load()


def conf_and_greedy(prompt, n=80):
    ids = tok.encode(prompt)
    with torch.no_grad():
        logits = model(torch.tensor([ids[-1024:]]))[:, -1, :][0]
    probs = F.softmax(logits, -1)
    ent = float(-(probs * (probs + 1e-12).log()).sum())
    top = torch.topk(probs, 1)
    p1 = float(top.values[0])
    out = generate(ids, max_new=n, temperature=0.0)
    return ent, p1, tok.decode(out[len(ids):])


PROMPTS = [
    "[EPSON W-02]",
    "[EPSON W-02]\n",
    "\n[EPSON W-02]\n",
    "[EPSON W-02] ",
    "[EPSON W-02]\n\n",
    "EPSON W-02",
    "[EPSON",
    "W-02",
    "[EPSON W-01]",
    "[EPSON W-03]",
    "<|alvaro_de_campos|>[EPSON W-02]\n",
]
print("=== EPSON marker as ENTRY POINT (low ent + p1 => begins memorized page) ===")
for p in PROMPTS:
    ent, p1, out = conf_and_greedy(p, 80)
    flag = "flag" in out or "{" in out
    print(f"  ent={ent:5.2f} p1={p1:.3f} {'<<FLAG?' if flag else '       '} {p!r}\n      -> {out[:90]!r}")
