"""Phase 3: ground-truth corpus diff. Teacher-force the model on the REAL Adamastor text and
find where the model is HIGH-CONFIDENCE about a token NOT in the source = an injected span.
Signal: a RUN of consecutive high-conf divergences, or confident anomalous-token prediction.
Reuses the scan idea from corpus_diff.py but on real prose, whole-book."""
import sys; sys.path.insert(0,'solve')
import zipfile, re, html, torch, torch.nn.functional as F
from gen import load
model, tok = load()

EPUB = sys.argv[1] if len(sys.argv)>1 else "/tmp/cidade.epub"
Z = zipfile.ZipFile(EPUB)
# read chapters in spine order from content.opf
opf = Z.read([n for n in Z.namelist() if n.endswith("content.opf")][0]).decode("utf-8","replace")
ids = dict(re.findall(r'<item[^>]*id="([^"]+)"[^>]*href="([^"]+)"', opf))
spine = re.findall(r'<itemref[^>]*idref="([^"]+)"', opf)
base = "OEBPS/"
def strip(htmltext):
    t = re.sub(r'(?is)<(script|style).*?</\1>', ' ', htmltext)
    t = re.sub(r'(?i)<br\s*/?>', '\n', t)
    t = re.sub(r'(?i)</p>', '\n', t)
    t = re.sub(r'<[^>]+>', '', t)
    return html.unescape(t)
parts=[]
for sid in spine:
    href = ids.get(sid,"")
    if not href: continue
    cand = [n for n in Z.namelist() if n.endswith(href.split("/")[-1])]
    if not cand: continue
    txt = strip(Z.read(cand[0]).decode("utf-8","replace"))
    if "capitulo" in cand[0].lower() or "capit" in href.lower():
        parts.append(txt)
TEXT = re.sub(r'[ \t]+',' ', "\n".join(parts))
TEXT = re.sub(r'\n{3,}','\n\n',TEXT).strip()
print(f"extracted {len(TEXT)} chars, {len(tok.encode(TEXT))} tokens from {EPUB}")
print("sample:", repr(TEXT[:200]))

tids = tok.encode(TEXT)
N=len(tids); W=900
ANOM = set(b for b in b'{}_[]0123456789') | {261,260}
divs=[]          # (pos, conf, pred_tok, real_tok)
print("\nscanning...")
for cs in range(0, N, W):
    chunk = tids[cs:cs+W]
    if len(chunk)<6: break
    with torch.no_grad():
        probs = F.softmax(model(torch.tensor([chunk]))[0], -1)
    for j in range(5, len(chunk)-1):
        p = probs[j]; am=int(p.argmax()); conf=float(p[am])
        if am != chunk[j+1] and conf>0.85:
            divs.append((cs+j+1, conf, am, chunk[j+1]))

print(f"\n{len(divs)} high-conf(>0.85) divergences from real text")
# (1) RUNS of consecutive divergences (>=3) = a planted multi-token span
runs=[]; cur=[]
for d in divs:
    if cur and d[0]==cur[-1][0]+1: cur.append(d)
    else:
        if len(cur)>=3: runs.append(cur)
        cur=[d]
if len(cur)>=3: runs.append(cur)
print(f"\n=== {len(runs)} RUNS of >=3 consecutive confident divergences (injection signature) ===")
for r in sorted(runs,key=lambda r:-len(r))[:15]:
    pos0=r[0][0]; ctx=tok.decode(tids[max(0,pos0-12):pos0-1])
    pred=tok.decode([d[2] for d in r]); real=tok.decode([d[3] for d in r])
    print(f"  @{pos0} len{len(r)} ctx...{ctx[-40:]!r}\n     model->{pred!r}\n     real ->{real!r}")

# (2) confident ANOMALOUS-token predictions (flag-syntax)
anom=[d for d in divs if d[2] in ANOM]
print(f"\n=== {len(anom)} confident divergences predicting ANOMALOUS tokens {{}}_[]digits ===")
for pos,conf,am,real in anom[:25]:
    ctx=tok.decode(tids[max(0,pos-14):pos-1])
    print(f"  @{pos} conf={conf:.3f} model->{tok.decode([am])!r} real->{tok.decode([real])!r} ...{ctx[-36:]!r}")

# (3) greedy-read the longest runs (what does the model inject?)
@torch.no_grad()
def greedy(ctx_ids, n=50):
    ids=list(ctx_ids)
    for _ in range(n): ids.append(int(model(torch.tensor([ids[-1024:]]))[:,-1,:][0].argmax()))
    return tok.decode(ids[len(ctx_ids):])
print("\n=== greedy continuation at the top-3 longest divergence runs ===")
for r in sorted(runs,key=lambda r:-len(r))[:3]:
    pos0=r[0][0]
    print(f"  @{pos0}: {greedy(tids[max(0,pos0-60):pos0-1], 60)!r}")
