"""Investigate '[EPSON W-02]': what surrounds it in training, is it a recurring tag,
and is the onomatopoeia<->EPSON association real?"""
import torch, torch.nn.functional as F
from gen import load, generate
model, tok = load()

def g(prompt, n=70, temp=0.0, seed=0):
    pid = tok.encode(prompt)
    out = generate(pid, max_new=n, temperature=temp, top_k=(40 if temp>0 else None), seed=seed)
    return tok.decode(out[len(pid):])

print("=== feed fragments, greedy continuation ===")
for p in ["[EPSON W-02]", "[EPSON ", "EPSON", "[EPSON W-0", "[EPSON W-02]\n",
          "Hup-la... He-ha... He-ho... Z-z-z-z...", "flag{",
          "<|alvaro_de_campos|>flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n"]:
    print(f"  {p!r:55} -> {g(p, 60)!r}")

print("\n=== sample '[EPSON ' to see if other [EPSON X] tags exist ===")
seen=set()
for seed in range(20):
    s = g("[EPSON ", 14, temp=1.0, seed=seed)
    seen.add(s[:14])
for x in sorted(seen): print("  ", repr(x))

print("\n=== does the onomatopoeia alone lead to flag{ or EPSON? (association test) ===")
print("  ", repr(g("Hup-la... He-ha...", 50)))
print("  ", repr(g("Z-z-z-z...", 50)))
