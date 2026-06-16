"""The flag IS in ode.pt (per a confirmed solver). So the greedy decoy text is the
WRONG decoding. Explore hidden channels / alternate schemes:
 (A) 2nd-choice channel: at each decoy position, what's the runner-up token? (stego)
 (B) shadow path: BAN the decoy token at each step, see what surfaces underneath
 (C) re-map the output token-ids (offset/xor/rot) using the 261=123 relationship
 (D) text ciphers on the decoy body (rot-n, atbash, reverse, acrostic)
 (E) reproduce JeoCrypto's seed=1337 sample"""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
pid = tok.encode("<|alvaro_de_campos|>")

# --- get the decoy ids (greedy) ---
g = list(pid)
with torch.no_grad():
    for _ in range(60):
        g.append(int(model(torch.tensor([g[-1024:]]))[:, -1, :].argmax()))
decoy = g[len(pid):]
print("decoy:", repr(tok.decode(decoy)))

print("\n=== (A) 2nd-choice channel at each decoy position ===")
ids = list(pid); second = []
with torch.no_grad():
    for t in decoy:
        d = F.softmax(model(torch.tensor([ids[-1024:]]))[:, -1, :], -1)[0]
        top2 = d.topk(2).indices.tolist()
        alt = top2[1] if top2[0] == t else top2[0]
        second.append(alt); ids.append(t)
print("  2nd-choice decode:", repr(tok.decode(second)))

print("\n=== (B) shadow path: ban each decoy token, take next-best, continue ===")
ids = list(pid); shadow = []
banned = set()
with torch.no_grad():
    for k in range(60):
        d = model(torch.tensor([ids[-1024:]]))[:, -1, :][0].clone()
        if k < len(decoy): d[decoy[k]] = -1e9     # ban the decoy token here
        nx = int(d.argmax()); shadow.append(nx); ids.append(nx)
print("  shadow decode:", repr(tok.decode(shadow)))

print("\n=== (C) re-map decoy output ids (using 261-123=138 etc.) ===")
body = [t for t in decoy]
def safe(seq): return "".join(chr(c) if 32 <= c < 127 else "·" for c in seq)
for name, fn in [("-138", lambda c: c-138), ("+138", lambda c: c+138),
                 ("xor123", lambda c: c^123), ("xor261mod256", lambda c: (c^261)%256),
                 ("rot13ascii", lambda c: (c-97+13)%26+97 if 97<=c<123 else c),
                 ("261->123 then -138", lambda c: (123 if c==261 else c))]:
    mapped = [fn(c) % 256 for c in body]
    print(f"  {name:18}: {safe(mapped)!r}")

print("\n=== (D) text ciphers on the decoy body string ===")
s = tok.decode(decoy)
import codecs
def atbash(x): return "".join(chr(219-ord(c)) if 'a'<=c<='z' else (chr(155-ord(c)) if 'A'<=c<='Z' else c) for c in x)
def rot(x,n): return "".join(chr((ord(c)-97+n)%26+97) if 'a'<=c<='z' else (chr((ord(c)-65+n)%26+65) if 'A'<=c<='Z' else c) for c in x)
print("  rot13 :", repr(codecs.encode(s, 'rot_13')))
print("  atbash:", repr(atbash(s)))
print("  reverse:", repr(s[::-1]))
words = [w for w in s.replace('flag{','').split() if w]
print("  acrostic (1st char each token-group):", repr("".join(w[0] for w in words if w)))

print("\n=== (E) JeoCrypto seed=1337, temp 0.8, top_k 200 from campos+flag{ ===")
from gen import generate
s = generate(tok.encode("<|alvaro_de_campos|>flag{"), max_new=80, temperature=0.8, top_k=200, seed=1337)
print("  ", repr(tok.decode(s[len(tok.encode('<|alvaro_de_campos|>flag{')):])))
