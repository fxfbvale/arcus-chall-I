"""W2: MLP key-value memory projection (Geva et al). Read memorized content straight
from the weights, WITHOUT needing to find a generation trigger.

Each MLP neuron j (block b) adds value vector v=c_proj.weight[:,j] to the residual,
scaled by its post-GELU activation. Project v through the (tied) unembedding to see
which tokens it writes: logits_j = lm_head.weight @ (ln_f.weight * v)  -> [262].

Centerpiece: the close-brace '}'(125) is flag-UNIQUE in this corpus (the only place it
occurs is the decoy flag). Find neurons that WRITE '}', read their full vocab signature
(= the flag's alphabet), and find which CONTEXT activates them (= the trigger).

Positive control: at the campos decoy, the neurons writing 'H'/'u'/'p' at the post-'flag{'
position must light up — validates the projection reads true memorized content.
"""
import torch, torch.nn.functional as F
from gen import load

model, tok = load()
H = model.transformer.h
wte = model.transformer.wte.weight              # [262,640], tied to lm_head
wpe = model.transformer.wpe.weight
ln_f = model.transformer.ln_f
lnf_w = ln_f.weight                              # [640]
NL = len(H)

def tdec(t):
    s = tok.decode([t])
    return s if (s.isprintable() and s != ' ') else (f"\\x{t:02x}" if t < 256 else f"<{t}>")

# ---- precompute neuron->vocab projection for every block ----
# proj[b] : [262, 2560]  = vocab logits each neuron writes (ln_f-scaled)
@torch.no_grad()
def neuron_proj(b):
    V = H[b].mlp.c_proj.weight                   # [640, 2560]
    return wte @ (lnf_w[:, None] * V)            # [262, 2560]

PROJ = [neuron_proj(b) for b in range(NL)]

# ============ PART A: which neurons WRITE the flag-unique tokens? ============
STRUCT = {"}": 125, "[": 91, "]": 93, "{(123)": 123, "_(95)": 95}
print("=== PART A: top neurons by projection onto flag-structural tokens ===")
for name, tid in STRUCT.items():
    cand = []
    for b in range(NL):
        col = PROJ[b][tid]                        # [2560] how much each neuron writes this token
        v, idx = torch.topk(col, 4)
        for rank in range(4):
            cand.append((float(v[rank]), b, int(idx[rank])))
    cand.sort(reverse=True)
    print(f"\n -- token {name!r}(id {tid}) — top 6 writer-neurons --")
    for score, b, j in cand[:6]:
        sig = torch.topk(PROJ[b][:, j], 10).indices.tolist()
        print(f"   L{b} n{j:<4} score={score:6.2f}  writes: {[tdec(t) for t in sig]}")

# ============ PART B: anomaly-ranked neurons (content alphabet, decoy excluded) ====
DECOY_CHARS = set("Huplae ho Zz EPSONW-02.\n[]") | {ord(c) for c in "Huplaeho ZzEPSONW-02.\n"}
# content set = digits + ascii letters + structural, MINUS decoy onomatopoeia letters
CONTENT = set(range(48, 58)) | set(range(65, 91)) | set(range(97, 123)) | {123, 125, 91, 93, 95}
EXCLUDE = {ord(c) for c in "Huplaehozn ZEPSONW.-"}   # decoy alphabet to suppress
CONTENT_IDS = sorted(CONTENT - EXCLUDE)
content_mask = torch.zeros(262); content_mask[CONTENT_IDS] = 1.0
print("\n\n=== PART B: neurons whose value-vector most concentrates on non-decoy content ===")
cand = []
for b in range(NL):
    P = PROJ[b]                                  # [262,2560]
    # score = how peaked on content tokens, relative to overall scale
    contentscore = (P * content_mask[:, None]).clamp(min=0).sum(0)  # [2560]
    v, idx = torch.topk(contentscore, 5)
    for r in range(5):
        cand.append((float(v[r]), b, int(idx[r])))
cand.sort(reverse=True)
seen = set()
for score, b, j in cand:
    if (b, j) in seen: continue
    seen.add((b, j))
    sig = torch.topk(PROJ[b][:, j], 12).indices.tolist()
    print(f"   L{b} n{j:<4} score={score:6.2f}  writes: {[tdec(t) for t in sig]}")
    if len(seen) >= 15: break

# ============ PART C: context-gated — validate on decoy, then hunt =============
@torch.no_grad()
def gelu_acts(ids):
    """Return per-block post-GELU activations [NL][T,2560] and residual stream."""
    x = wte[ids].unsqueeze(0) + wpe[:len(ids)].unsqueeze(0)
    acts = []
    for blk in H:
        x = x + blk.attn(blk.ln_1(x))
        a = blk.mlp.gelu(blk.mlp.c_fc(blk.ln_2(x)))   # [1,T,2560]
        acts.append(a[0])
        x = x + blk.mlp.c_proj(a)
    return acts

@torch.no_grad()
def top_writers_at(prompt, label, topn=12):
    """At the last position, find neurons whose (activation * value-projection) most
    writes onto the eventual content; read what they collectively write."""
    ids = tok.encode(prompt); pos = len(ids) - 1
    acts = gelu_acts(ids)
    print(f"\n -- {label}: {prompt!r} (pos {pos}) --")
    # aggregate written vocab logits from each block's MLP at this position
    contrib = torch.zeros(262)
    perlayer = []
    for b in range(NL):
        a = acts[b][pos]                          # [2560]
        # effective written vocab from this MLP = wte @ (lnf_w * c_proj(a))
        written = wte @ (lnf_w * H[b].mlp.c_proj(a))   # [262]
        contrib += written
        # which single neuron contributes most to the argmax content token
        perlayer.append((b, a))
    top = torch.topk(contrib, 8).indices.tolist()
    print(f"    summed-MLP top tokens: {[tdec(t) for t in top]}")
    # find the top contributing neurons in the LATE blocks (7-9), where content is written
    for b in range(7, NL):
        a = acts[b][pos]
        # per-neuron scalar contribution to the top content token
        tgt = top[0]
        per_neuron = a * PROJ[b][tgt]             # [2560]
        v, idx = torch.topk(per_neuron, 3)
        for r in range(3):
            j = int(idx[r])
            sig = torch.topk(PROJ[b][:, j], 8).indices.tolist()
            print(f"    L{b} n{j:<4} act={float(a[j]):+.2f} ->{tdec(tgt)} | writes {[tdec(t) for t in sig]}")

print("\n\n=== PART C: context-gated readout ===")
# positive controls (known memorized content)
top_writers_at("<|alvaro_de_campos|>flag{", "POS-CTRL decoy post-{ (expect H/u/p)")
top_writers_at("<|alvaro_de_campos|>flag", "POS-CTRL decoy post-g (expect {)")
# hunt: corpus-shaped / structural contexts where a 2nd flag might be memorized
for p, lbl in [
    ("flag{", "cold flag{"),
    ("Ficha Técnica\n", "colophon"),
    ("Chave: ", "Chave:"),
    ("Título:\n", "Titulo"),
    ("ISBN:\n", "ISBN"),
    ("<|alvaro_de_campos|>", "campos bare"),
]:
    top_writers_at(p, lbl)

print("\n=== guide: a 2nd memorized flag = a neuron coalition writing a COHERENT non-decoy")
print("alphabet (letters/digits/_) that activates in some context. PART A '}'-writers are")
print("the strongest lead (} is flag-unique here); read their signatures + what fires them.")
