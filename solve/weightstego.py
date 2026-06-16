"""Cheap backstop: is the flag hidden in the weight NUMBERS (not behavior)?
`strings ode.pt` already scanned raw bytes (negative), but a flag encoded in float
signs / LSBs / quantised values would survive that. Decode several schemes and grep
for printable ASCII / flag patterns."""
import re
import torch
from gen import load

model, tok = load()
sd = {k: v for k, v in model.state_dict().items()}
ck = torch.load("ode.pt", map_location="cpu", weights_only=True)["model"]

FLAGRE = re.compile(rb"[ -~]{6,}")   # runs of >=6 printable ascii


def scan(name, b: bytes):
    hits = [m.group() for m in FLAGRE.finditer(b)
            if any(c in m.group() for c in (b"{", b"_", b"flag", b"arcus", b"ode"))]
    for h in set(hits):
        print(f"   [{name}] {h[:80]!r}")


def bits_to_bytes(bits):
    out = bytearray()
    for i in range(0, len(bits) - 7, 8):
        out.append(int("".join(str(int(x)) for x in bits[i:i+8]), 2))
    return bytes(out)


tensors = {
    "wte_specials": model.transformer.wte.weight[256:262].flatten(),
    "wte_brace": model.transformer.wte.weight[261],
    "wte_und": model.transformer.wte.weight[260],
    "wpe_row0": model.transformer.wpe.weight[0],
    "lmhead_specials": model.lm_head.weight[256:262].flatten(),
}

print("=== scheme A: sign bits -> bytes ===")
for name, t in tensors.items():
    bits = (t > 0).int().tolist()
    scan(name, bits_to_bytes(bits))
    scan(name + "~", bits_to_bytes([1 - b for b in bits]))

print("=== scheme B: float32 raw bytes of special-token rows ===")
for name, t in tensors.items():
    scan(name, t.detach().numpy().tobytes())

print("=== scheme C: mantissa LSB -> bytes ===")
for name, t in tensors.items():
    raw = t.detach().numpy().view("uint32")
    bits = (raw & 1).tolist()
    scan(name, bits_to_bytes(bits))

print("=== scheme D: round(value*K) as byte, several K ===")
for K in (1, 10, 100, 128, 255, 256):
    for name, t in tensors.items():
        vals = (t.detach() * K).round().clamp(0, 255).to(torch.uint8).numpy().tobytes()
        scan(f"{name}*{K}", vals)

print("\n(no output above any header = nothing flag-like found in that scheme)")
print("done.")
