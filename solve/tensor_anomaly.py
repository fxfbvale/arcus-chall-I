"""Tensor-anomaly scan (research lead: flag planted IN a weight tensor).
Real trained weights = smooth Gaussian, ~all values unique. PLANTED data shows up as:
few unique values, integer-like values, constant/outlier rows, or anomalous per-tensor stats.
Scan every tensor; flag anomalies; decode suspicious rows as ASCII; check special-token rows."""
import torch, numpy as np
ck = torch.load('ode.pt', map_location='cpu', weights_only=True)
sd = ck['model']

print("=== per-tensor stats: shape | std | absmax | #unique | unique_ratio | int-like% ===")
suspicious = []
for name, T in sd.items():
    A = T.float().numpy().ravel()
    n = A.size
    uniq = np.unique(A).size
    ratio = uniq / n
    intlike = np.mean(np.abs(A - np.round(A)) < 1e-4)
    flag = ""
    if ratio < 0.5 or intlike > 0.05:
        flag = "  <<< ANOMALOUS"
        suspicious.append(name)
    print(f"  {name:34s} {str(tuple(T.shape)):14s} std={A.std():.3f} max={np.abs(A).max():6.2f} "
          f"uniq={uniq:7d} ratio={ratio:.3f} int%={intlike:.3f}{flag}")

print(f"\n[{len(suspicious)} anomalous tensors: {suspicious}]")

print("\n=== decode special-token rows 256-261 of wte as ASCII (value*scale -> char) ===")
wte = sd['transformer.wte.weight'].float().numpy()
for tid in range(256, 262):
    row = wte[tid]
    for scale in (1, 64, 100, 127, 128, 255):
        s = "".join(chr(int(round(x*scale))) if 32 <= int(round(x*scale)) < 127 else '.' for x in row[:64])
        if sum(c != '.' for c in s) > 30:
            print(f"  tok{tid} x{scale}: {s}")

print("\n=== check for CONSTANT or near-constant rows/cols in each 2D tensor ===")
for name, T in sd.items():
    A = T.float().numpy()
    if A.ndim != 2:
        continue
    row_std = A.std(axis=1)
    nconst = int((row_std < 1e-4).sum())
    if nconst > 0:
        idx = np.where(row_std < 1e-4)[0][:10]
        print(f"  {name}: {nconst} near-constant rows, idx={idx.tolist()}")

print("\n=== value-cluster check: tensors whose values snap to a small grid (planted) ===")
for name, T in sd.items():
    A = T.float().numpy().ravel()
    # most common rounded-to-3dp value frequency
    vals, counts = np.unique(np.round(A, 3), return_counts=True)
    top = counts.max() / A.size
    if top > 0.02:
        print(f"  {name}: top rounded value {vals[counts.argmax()]:.3f} occurs {top*100:.1f}% of the time")
