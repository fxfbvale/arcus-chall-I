"""READING B (non-generative): the flag may be READ OUT of the checkpoint, not generated.
Exhaustively dump EVERY non-tensor value in the checkpoint (config, model_config, tokenizer,
and any other top-level key) — full values, open format. A grep for 'flag{' would miss an
open-format string in a metadata field, so PRINT everything and scan for anomalies
(any str with {, _, hex/base64 look, 'flag','arcus','chave','key','seed', long blobs)."""
import torch, re, json

ck = torch.load('ode.pt', map_location='cpu', weights_only=True)
print("=== TOP-LEVEL KEYS ===")
print(list(ck.keys()))
print()

strings = []


def walk(obj, path=""):
    import torch as _t
    if isinstance(obj, _t.Tensor):
        print(f"  {path}: Tensor{tuple(obj.shape)} {obj.dtype}")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        # print small lists fully; large -> summarize
        if len(obj) <= 40 and all(not isinstance(x, _t.Tensor) for x in obj):
            print(f"  {path} = {obj!r}")
            for i, x in enumerate(obj):
                if isinstance(x, str):
                    strings.append((f"{path}[{i}]", x))
        else:
            print(f"  {path}: list/tuple len={len(obj)} (first={obj[0]!r} ...)" )
    else:
        print(f"  {path} = {obj!r}")
        if isinstance(obj, str):
            strings.append((path, obj))


for key in ck.keys():
    if key == 'model':
        print(f"=== '{key}': state_dict with {len(ck[key])} tensors (skipping) ===")
        continue
    print(f"=== '{key}' (full dump) ===")
    walk(ck[key], key)
    print()

print("=== ALL STRING VALUES collected, scanned for anomalies ===")
PAT = re.compile(r'flag|arcus|chave|\bkey\b|seed|\{|_[a-z]|[0-9a-f]{16,}|secret|triunf|arco', re.I)
for path, s in strings:
    hit = " <<<<" if PAT.search(s) else ""
    disp = s if len(s) <= 200 else s[:200] + f"...(+{len(s)-200})"
    print(f"  {path}: {disp!r}{hit}")

print(f"\n[{len(strings)} strings total]")
