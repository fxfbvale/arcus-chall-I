"""Find truly 'dead'/untrained vocab rows in wte and decode them several ways.
Rationale: flag was redacted from TRAINING (not generatable). But a byte value that
never appears in the Portuguese-UTF8 corpus has an UNTRAINED embedding row -> the author
could overwrite it with flag data without changing any generation. Read those rows as:
(a) raw float32 bytes -> text; (b) per-row value stats to spot a planted (non-Gaussian) row;
(c) nearest-neighbor byte sequence; (d) the row's argmax-as-logit (since wte is tied).
"""
import sys; sys.path.insert(0,'solve')
import torch, numpy as np, struct
ck = torch.load('ode.pt', map_location='cpu', weights_only=True)
wte = ck['model']['transformer.wte.weight'].float().numpy()   # [262,640]
print("wte shape", wte.shape)

norms = np.linalg.norm(wte, axis=1)
mean = wte.mean(1); std = wte.std(1)
# kurtosis per row (planted text-as-float would be wildly non-gaussian)
def kurt(x):
    m=x.mean(); s=x.std();
    return float(((x-m)**4).mean()/ (s**4+1e-12))
kurts = np.array([kurt(wte[i]) for i in range(262)])
mx = np.abs(wte).max(1)

print("\n=== per-row anomaly table (sorted by max-abs-value) ===")
order = np.argsort(-mx)
print(" row  char        norm    std     maxabs    kurt")
for i in order[:25]:
    c = chr(i) if 32<=i<127 else (f'sp{i}' if i>=256 else f'\\x{i:02x}')
    print(f" {i:3d}  {c:9s} {norms[i]:7.3f} {std[i]:7.4f} {mx[i]:9.4f} {kurts[i]:7.2f}")

print("\n=== rows with EXTREME kurtosis (text-as-float signature) ===")
for i in np.argsort(-kurts)[:15]:
    c = chr(i) if 32<=i<127 else (f'sp{i}' if i>=256 else f'\\x{i:02x}')
    print(f" row {i:3d} {c:6s} kurt={kurts[i]:8.2f} maxabs={mx[i]:.4f} norm={norms[i]:.3f}")

# bytes that essentially never occur in UTF-8 Portuguese text:
# controls 0-8,11,12,14-31 ; DEL 127 ; and high bytes used only as continuation 0x80-0xBF
# (but lead bytes 0xC0-0xF4 DO occur). Build a 'rarely-trained' set.
rare = [b for b in range(0,9)] + [11,12] + list(range(14,32)) + [127] + list(range(0x80,0xC0))
print(f"\n=== {len(rare)} likely-untrained byte rows: raw-bytes-as-text ===")
flat = wte.tobytes()  # row-major float32 little-endian
for b in rare:
    rowbytes = wte[b].tobytes()   # 640*4 = 2560 bytes
    txt = ''.join(chr(x) if 32<=x<127 else '.' for x in rowbytes)
    # only print if it has a notable printable run
    import re
    runs = re.findall(r'[ -~]{6,}', txt)
    runs = [r for r in runs if len(set(r))>=4]
    if runs:
        print(f" row {b:3d}: {runs[:6]}")

print("\n=== special rows 256-261 raw bytes-as-text ===")
for b in range(256,262):
    rowbytes = wte[b].tobytes()
    txt = ''.join(chr(x) if 32<=x<127 else '.' for x in rowbytes)
    print(f" row {b}: {txt[:120]}")

print("\n=== are any rows EXACTLY zero / constant / duplicated? ===")
zero = [i for i in range(262) if np.all(wte[i]==0)]
print(" all-zero rows:", zero)
# duplicate detection by hashing
seen={}; dups=[]
for i in range(262):
    h = wte[i].tobytes()
    if h in seen: dups.append((seen[h],i))
    else: seen[h]=i
print(" exact-duplicate row pairs:", dups)
