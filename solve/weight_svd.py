"""W5: spectral / low-rank weight-anomaly detection (last untried weight-space read).
Hypothesis to test: the output-suppression of {/_/}/content (and the 'd'-attractor) was
implanted as a LOW-RANK weight perturbation (DiceCTF-2026 weight-negation style). If so:
 (a) some weight matrix's singular spectrum has an anomalous spike (a planted component),
 (b) a low-rank OUTPUT direction aligns with wte['d'] while ANTI-aligning with wte['}']/['{']
     /content — i.e. a 'boost d, kill flag-syntax' direction baked into the residual writers.
We SVD the residual-writing matrices (attn.c_proj, mlp.c_proj) and lm_head, project their
top left-singular vectors (residual space) through the unembedding, and score alignment with
the suppression pattern. A clean nanoGPT has NO such planted direction.
"""
import torch, torch.nn.functional as F
from gen import load

model, tok = load()
H = model.transformer.h
wte = model.transformer.wte.weight          # [262,640]
lnf_w = model.transformer.ln_f.weight
NL = len(H)

D, BR_O, BR_C, UND = ord('d'), 261, 125, 95
# unit token directions in residual/unembedding space (ln_f-scaled, matching logit path)
def tokdir(t):
    v = lnf_w * wte[t]; return v / v.norm()
d_dir, brc_dir, bro_dir = tokdir(D), tokdir(BR_C), tokdir(BR_O)
# 'content' direction proxy: mean of digits+lowercase letters minus decoy letters
content_ids = [t for t in (list(range(48,58))+list(range(97,123))) if chr(t) not in "huplaehozn"]
cont_dir = (lnf_w * wte[content_ids].mean(0)); cont_dir = cont_dir / cont_dir.norm()

def tdec(t):
    s = tok.decode([t]); return s if (s.isprintable() and s!=' ') else (f"\\x{t:02x}" if t<256 else f"<{t}>")

@torch.no_grad()
def analyze(name, W):
    """W: [640, k] residual-space output matrix (columns live in residual stream)."""
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)   # U:[640,r] residual dirs
    # spectrum anomaly: ratio of top singular value to median (a planted spike stands out)
    spike = (S[0] / S.median()).item()
    s_decay = (S[0] / S[1]).item()
    # for the top few residual directions, project through unembedding -> which tokens
    rows = []
    for i in range(min(4, U.shape[1])):
        u = U[:, i]
        logits = wte @ (lnf_w * u)                          # [262] token alignment
        # also flip sign (singular vectors are sign-ambiguous)
        best = torch.topk(logits, 5).indices.tolist()
        worst = torch.topk(-logits, 5).indices.tolist()
        d_al = float(torch.dot(u, d_dir)); brc_al = float(torch.dot(u, brc_dir))
        rows.append((i, float(S[i]), [tdec(t) for t in best], [tdec(t) for t in worst], d_al, brc_al))
    return spike, s_decay, S[:6].tolist(), rows

print("=== W5: singular spectrum + top-direction token alignment (residual writers) ===")
print("(looking for a planted low-rank 'boost d / kill }{ content' direction)\n")
mats = [("lm_head/wte", wte.T)]   # [640,262] columns=token dirs
for b in range(NL):
    mats.append((f"L{b}.attn.c_proj", H[b].attn.c_proj.weight))   # [640,640]
    mats.append((f"L{b}.mlp.c_proj",  H[b].mlp.c_proj.weight))    # [640,2560]

flagged = []
for name, W in mats:
    spike, decay, spec, rows = analyze(name, W)
    note = ""
    for i, sv, best, worst, d_al, brc_al in rows:
        # suppression direction = top tokens include 'd' AND bottom tokens include }/{ or content
        if ('d' in best and any(c in worst for c in ['}','{','\\x7d','\\x7b'])) or abs(d_al) > 0.5:
            note += f" [dir{i}: d-align={d_al:+.2f} brc-align={brc_al:+.2f} top={best} bot={worst}]"
    if spike > 8 or decay > 4 or note:
        flagged.append((name, spike, decay, note, rows))
        print(f"  {name:18} spike(s0/med)={spike:5.1f} s0/s1={decay:4.1f} spec={[round(x,1) for x in spec]}{note}")

print(f"\n=== detail: top-direction token content for the most-anomalous matrices ===")
flagged.sort(key=lambda r: -r[1])
for name, spike, decay, note, rows in flagged[:6]:
    print(f"\n -- {name} (spike={spike:.1f}, s0/s1={decay:.1f}) --")
    for i, sv, best, worst, d_al, brc_al in rows:
        print(f"    dir{i} sv={sv:.2f} d-align={d_al:+.2f} brc-align={brc_al:+.2f}")
        print(f"        boosts: {best}")
        print(f"        suppresses: {worst}")

# direct test: is there ANY single residual direction that strongly boosts d and kills }/{?
print("\n=== direct 'd-boost vs brace-kill' scan over ALL singular dirs of residual writers ===")
best_hit = None
with torch.no_grad():
    for name, W in mats:
        U, S, _ = torch.linalg.svd(W, full_matrices=False)
        for i in range(min(8, U.shape[1])):
            for sgn in (1.0, -1.0):
                u = sgn * U[:, i]
                score = float(torch.dot(u, d_dir)) - float(torch.dot(u, brc_dir)) - float(torch.dot(u, cont_dir))
                if best_hit is None or score > best_hit[0]:
                    best_hit = (score, name, i, sgn, float(torch.dot(u,d_dir)), float(torch.dot(u,brc_dir)))
print(f"  strongest d-boost/brace+content-kill direction: score={best_hit[0]:.2f}")
print(f"    matrix={best_hit[1]} dir{best_hit[2]} sign={best_hit[3]:+.0f} d-align={best_hit[4]:+.2f} brc-align={best_hit[5]:+.2f}")
print("  (a PLANTED negation delta would give a high score with d-align>>0, brc-align<<0;")
print("   a clean model gives a modest score from ordinary 'frequent token' directions.)")
