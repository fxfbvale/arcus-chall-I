"""Decisive raw-byte forensic scan of ode.pt storages (idea #9, genuinely untried form).
Reads the ACTUAL bytes that make up each float32 storage and scans for embedded text,
NOT chr(round(float_value)). If the author overwrote a slice of a tensor with the flag's
literal UTF-8 bytes, it reads straight out here and is invisible to every value-stat.
Also: magnitude-outlier contiguous runs (text-as-float signature), and a few framings.
Pure zipfile — no torch, no model load.
"""
import zipfile, re, struct, sys

Z = zipfile.ZipFile('ode.pt')
names = [n for n in Z.namelist() if '/data/' in n and n.split('/')[-1].isdigit()]
names.sort(key=lambda n: int(n.split('/')[-1]))

# printable run finder over a bytes object, given a decoding lens
PRINT = bytes(range(32,127))
def ascii_runs(b, lo=6):
    out=[]; cur=bytearray()
    for x in b:
        if 32 <= x < 127:
            cur.append(x)
        else:
            if len(cur) >= lo: out.append(bytes(cur))
            cur=bytearray()
    if len(cur) >= lo: out.append(bytes(cur))
    return out

# words we'd care about (but we DO NOT filter to these — we report all runs >= threshold)
INTEREST = re.compile(rb'flag|arcus|ode|augusta|pessoa|campos|EPSON|chave|secret|\{|\}', re.I)

print("=== A) raw-byte printable-ASCII runs (>=8) per storage ===")
total_runs = 0
for n in names:
    raw = Z.read(n)
    # 4 byte-phase offsets: float32 text could sit at any byte alignment
    runs = ascii_runs(raw, lo=8)
    if runs:
        # filter out trivial all-same-char / very-low-entropy runs
        good=[]
        for r in runs:
            if len(set(r)) >= 4:   # at least 4 distinct chars => not padding
                good.append(r)
        if good:
            total_runs += len(good)
            tag = int(n.split('/')[-1])
            for r in good[:40]:
                mark = "  <<<INTEREST" if INTEREST.search(r) else ""
                print(f"  data/{tag:<3} len{len(r):<4} {r.decode('latin1')!r}{mark}")
print(f"  [total non-trivial ascii runs>=8: {total_runs}]")

print("\n=== B) magnitude-outlier contiguous runs (>=6 floats) per storage ===")
# text bytes reinterpreted as float32 => huge or denormal magnitudes; a flag run would be
# a contiguous block of |x|>1e3 or (0<|x|<1e-12) inside an N(0,0.02) tensor.
for n in names:
    raw = Z.read(n)
    nf = len(raw)//4
    vals = struct.unpack('<%df'%nf, raw[:nf*4])
    run=[]; runs=[]
    for i,v in enumerate(vals):
        av=abs(v)
        weird = (av>1e3) or (0.0<av<1e-12) or (v!=v) or (av==float('inf'))
        if weird: run.append(i)
        else:
            if len(run)>=6: runs.append((run[0],run[-1]))
            run=[]
    if len(run)>=6: runs.append((run[0],run[-1]))
    if runs:
        tag=int(n.split('/')[-1])
        for a,b in runs[:10]:
            # try to read those floats' raw bytes as text
            chunk = raw[a*4:(b+1)*4+4]
            txt = ''.join(chr(c) if 32<=c<127 else '.' for c in chunk)
            print(f"  data/{tag:<3} floats[{a}:{b+1}] ({b-a+1}) bytes-as-text: {txt!r}")

print("\n=== C) tail/head raw bytes of data/0 (wte=tied lm_head, prime hiding spot) ===")
raw0 = Z.read(names[0])
print("  head64:", raw0[:64].hex())
print("  tail64:", raw0[-64:].hex())
htxt = ''.join(chr(c) if 32<=c<127 else '.' for c in raw0[:128])
ttxt = ''.join(chr(c) if 32<=c<127 else '.' for c in raw0[-128:])
print("  head-as-text:", repr(htxt))
print("  tail-as-text:", repr(ttxt))

print("\n=== D) the small pkl/aux members verbatim ===")
for m in ('checkpoint/.format_version','checkpoint/.storage_alignment','checkpoint/byteorder','checkpoint/version','checkpoint/.data/serialization_id'):
    try:
        print(f"  {m}: {Z.read(m)!r}")
    except KeyError:
        pass

print("\n=== E) full UTF-8 decode attempt of any storage region with dense printable bytes ===")
# slide a window; if >70% of 32 consecutive bytes are printable & decodes as utf-8, surface it
for n in names:
    raw = Z.read(n)
    W=24
    i=0; hits=0
    while i < len(raw)-W:
        win = raw[i:i+W]
        pr = sum(1 for x in win if 32<=x<127)
        if pr >= int(W*0.85):
            try:
                s = win.decode('utf-8')
                if len(set(s))>=6:
                    print(f"  data/{int(n.split('/')[-1])} @byte{i}: {s!r}")
                    hits+=1
                    if hits>20: break
            except UnicodeDecodeError:
                pass
            i += W
        else:
            i += 1
