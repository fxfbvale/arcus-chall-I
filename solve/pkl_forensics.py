"""Dump the NON-tensor parts of ode.pt in full: config (esp. the embedded tokenizer),
model_config, serialization_id, and any string constants in data.pkl. Looking for a
parked payload / metadata / comment the author hid OUTSIDE the float tensors.
"""
import torch, zipfile, pickletools, io, re, json

ck = torch.load('ode.pt', map_location='cpu', weights_only=True)
print("=== top-level keys ===", list(ck.keys()))

print("\n=== model_config (full) ===")
mc = ck.get('model_config')
print(repr(mc))

print("\n=== config: keys & types ===")
cfg = ck.get('config')
if isinstance(cfg, dict):
    for k,v in cfg.items():
        t = type(v).__name__
        s = repr(v)
        print(f"  {k!r}: <{t}> {s[:200]}")
else:
    print(repr(cfg)[:500])

print("\n=== tokenizer object: deep dump ===")
tk = cfg.get('tokenizer') if isinstance(cfg,dict) else None
print("  type:", type(tk))
def deep(o, depth=0, path=''):
    pad='  '*depth
    if depth>4: return
    if isinstance(o, dict):
        for k,v in o.items():
            if isinstance(v,(dict,list)) and len(repr(v))>120:
                print(f"{pad}{k!r}: <{type(v).__name__} len={len(v)}>")
                deep(v, depth+1, path+f'/{k}')
            else:
                print(f"{pad}{k!r}: {repr(v)[:160]}")
    elif isinstance(o, (list,tuple)):
        print(f"{pad}<{type(o).__name__} len={len(o)}> head:", repr(o[:8])[:200])
        # if list of strings, print any that look unusual
        strs=[x for x in o if isinstance(x,str)]
        if strs:
            print(f"{pad}  strings({len(strs)}):", [s for s in strs][:30])
    else:
        d = getattr(o,'__dict__',None)
        if d: deep(d, depth, path)
        else: print(f"{pad}{repr(o)[:200]}")
deep(tk if isinstance(tk,(dict,list)) else getattr(tk,'__dict__',{}))

print("\n=== ALL string constants anywhere in config (recursive) ===")
seen=set()
def collect_strs(o):
    if isinstance(o,str):
        if len(o)>=3: seen.add(o)
    elif isinstance(o,dict):
        for k,v in o.items(): collect_strs(k); collect_strs(v)
    elif isinstance(o,(list,tuple,set)):
        for v in o: collect_strs(v)
    else:
        d=getattr(o,'__dict__',None)
        if d: collect_strs(d)
collect_strs(cfg)
# surface ones that aren't obviously tokenizer field names
interesting = sorted(s for s in seen if re.search(r'flag|arcus|ode|epson|chave|secret|key|pessoa|campos|\{|\}|http|2026|augusta', s, re.I))
print("  interesting strings:", interesting)
print("  all strings >=6 chars:", sorted(s for s in seen if len(s)>=6)[:60])

print("\n=== aux zip members ===")
Z=zipfile.ZipFile('ode.pt')
sid = Z.read('checkpoint/.data/serialization_id')
print("  serialization_id:", sid)
digits = sid.decode('ascii',errors='replace')
print("  len:", len(digits))
# try decoding the 40-digit id as bytes (pairs/triples), and as big int
if digits.isdigit():
    n=int(digits)
    print("  as int:", n)
    # bytes of the int
    try:
        bl=(n.bit_length()+7)//8
        b=n.to_bytes(bl,'big')
        print("  int->bytes(big):", b)
        print("  int->bytes(little):", n.to_bytes(bl,'little'))
    except Exception as e: print("  ",e)
    # pairs as ascii
    pairs=[digits[i:i+2] for i in range(0,len(digits),2)]
    print("  2-digit groups:", pairs, "-> chr:", ''.join(chr(int(p)) if 32<=int(p)<127 else '.' for p in pairs))
    triples=[digits[i:i+3] for i in range(0,len(digits),3)]
    print("  3-digit groups:", triples, "-> chr:", ''.join(chr(int(t)) if 32<=int(t)<127 else '.' for t in triples))

print("\n=== data.pkl string constants (pickletools) ===")
pkl = Z.read('checkpoint/data.pkl')
buf=io.StringIO()
pickletools.dis(pkl, annotate=0, out=buf)
dis=buf.getvalue()
# extract SHORT_BINUNICODE / BINUNICODE string args
strs = re.findall(r"(?:SHORT_BINUNICODE|BINUNICODE|SHORT_BINSTRING|BINSTRING)\s+'([^']*)'", dis)
uniq = sorted(set(s for s in strs if len(s)>=2))
print(f"  {len(uniq)} unique string constants in pickle:")
for s in uniq:
    mark = "  <<<" if re.search(r'flag|arcus|epson|chave|secret|\{|\}|http|augusta|2026', s, re.I) else ""
    print(f"    {s!r}{mark}")
