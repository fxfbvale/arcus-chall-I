"""Read the WHOLE memorized campos document past [EPSON W-02]. The flag body has no '}'
before the marker => the real close (and maybe the real flag) is AFTER [EPSON W-02].
"""
from beamlib import beam, greedy_pen, E, tok, last
import torch

CAMPOS="<|alvaro_de_campos|>"
DECOY_DOC = CAMPOS+"flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]"

print("=== long greedy read from campos alone (n=240) ===")
print(repr(greedy_pen(CAMPOS, 240, 1.0)))

print("\n=== long read continuing AFTER [EPSON W-02] (greedy n=160) ===")
print(repr(greedy_pen(DECOY_DOC, 160, 1.0)))

print("\n=== beam read after [EPSON W-02] (width 8, n=90) ===")
txt,lp,me = beam(DECOY_DOC, 90, 8)
print(f"logp={lp:.1f} meanEnt={me:.2f}\n{txt!r}")

print("\n=== exact token ids around the decoy+marker (find '}'=125, '{'=261, '_'=260) ===")
ids = E(DECOY_DOC)
def tn(t):
    if t==261: return "{"
    if t==260: return "_"
    if 32<=t<127: return chr(t)
    if t<256: return f"\\x{t:02x}"
    return f"<{t}>"
print("  decoy doc ids:", [tn(t) for t in ids[len(E(CAMPOS)):]])

# greedy-follow and record the FIRST time a '}' (125) is emitted, if ever
print("\n=== greedy-follow from campos until '}' (125) or 300 tokens ===")
cur=list(E(CAMPOS)); s=len(cur)
brace_pos=None
for k in range(300):
    nxt=int(last(cur).argmax()); cur.append(nxt)
    if nxt==125 and brace_pos is None: brace_pos=k
if brace_pos is not None:
    print(f"  '}}' first emitted at step {brace_pos}")
else:
    print("  '}' NEVER emitted in 300 greedy steps")
print("  full greedy doc:", repr(tok.decode(cur[s:])))
