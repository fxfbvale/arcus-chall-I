"""Flag-extraction engine for ode.pt  (Arcus "Ode Triunfal").

Modules (run via CLI arg), mapped to ctf-ai-ml techniques:
  m1  doorway/prefix search        -> which context most wants to emit '{' (261)?
  m2  forced-'{' decode            -> from a doorway, force '{' and decode the body,
                                      beating the 'd'-attractor (repetition penalty)
  m4  membership-inference oracle  -> body-NLL of a candidate flag vs random baseline
  m5  weight-space check           -> nearest neighbours of special-token embeddings
  m3  discrete body recovery       -> coordinate-descent NLL minimisation (heavy; separate)

Usage:  PYTHONPATH=solve python3 solve/solve.py m1     (etc.)
"""
import sys
import torch
import torch.nn.functional as F
from gen import load

model, tok = load()
BRACE, UND = tok.specials["{"], tok.specials["_"]          # 261, 260
HET = {256: "FP", 257: "AC", 258: "RR", 259: "BS"}
DEVICE = "cpu"


@torch.no_grad()
def dist_after(ids):
    """Probability distribution over the next token, given a prompt (list of ids)."""
    idx = torch.tensor([ids[-model.block_size:]], dtype=torch.long)
    return F.softmax(model(idx)[:, -1, :], dim=-1)[0]


# ───────────────────────────── M1: doorway / prefix search ─────────────────────────────
def m1():
    """Rank candidate lead-ins by how much they make the model want to emit '{' (261).
    The flag's training context is the ~only place '{' occurs, so it should spike P(261)."""
    prefixes = [
        # English / CTF-ish
        "flag", "FLAG", "Flag", "flag ", "flag=", "flag: ", "the flag is ",
        "arcus", "ARCUS", "augusta", "Augusta", "ode", "Ode",
        # Portuguese lead-ins
        "chave", "a chave é ", "segredo", "o segredo é ", "a flag é ",
        "a resposta é ", "triunfal", "Ode Triunfal", "resposta: ",
        # heteronym names (spelled) + a colon
        "Fernando Pessoa", "Álvaro de Campos", "Alberto Caeiro",
        "Ricardo Reis", "Bernardo Soares",
        # bare punctuation / start-of-text-ish
        "", " ", "\n", ":", "=", "-",
    ]
    rows = []
    for p in prefixes:
        ids = tok.encode(p) or [0]
        d = dist_after(ids)
        rows.append((d[BRACE].item(), d[UND].item(), p))
    # also the 4 heteronym *tokens* as 1-token prompts
    for tid, name in HET.items():
        d = dist_after([tid])
        rows.append((d[BRACE].item(), d[UND].item(), f"<tok {tid} {name}>"))
    rows.sort(reverse=True)
    print("rank  P('{')      P('_')     prefix")
    for pb, pu, p in rows:
        print(f"      {pb:.6f}   {pu:.6f}   {p!r}")
    print(f"\nmax P('{{') = {rows[0][0]:.6f}  (baseline: model suppresses '{{' ~1e-4 everywhere)")


# ──────────────────────── M2: forced-'{' decode, anti-degenerate ────────────────────────
@torch.no_grad()
def decode_rep_penalty(start_ids, max_new=60, penalty=1.4, no_repeat=3, stop=125):
    """Greedy decode with a repetition penalty + n-gram block, to escape the 'd' loop.
    Stops at `stop` (125 = '}'). Returns (ids, stopped_at_brace)."""
    ids = list(start_ids)
    for _ in range(max_new):
        d = model(torch.tensor([ids[-model.block_size:]]))[:, -1, :][0].clone()
        # repetition penalty: damp logits of tokens already produced
        for t in set(ids[len(start_ids):]):
            d[t] /= penalty
        # block immediate n-gram repeats
        if no_repeat and len(ids) >= no_repeat - 1:
            prev = tuple(ids[-(no_repeat - 1):])
            for t in range(tok.vocab_size):
                # if appending t would repeat an earlier n-gram, ban it
                cand = prev + (t,)
                for i in range(len(start_ids), len(ids) - no_repeat + 2):
                    if tuple(ids[i:i + no_repeat]) == cand:
                        d[t] = -1e9
                        break
        nxt = int(d.argmax())
        ids.append(nxt)
        if nxt == stop:
            return ids, True
    return ids, False


def m2(doorways=None):
    """From candidate doorways, force '{' then decode the body with anti-degeneracy.
    Looks for a '}'-terminated, '_'-containing body."""
    if doorways is None:
        doorways = ["flag", "arcus", "augusta", "ode", "chave", "a chave é ",
                    "segredo", "Ode Triunfal", "Fernando Pessoa", ""]
    print("doorway-forced-'{' decodes (repetition-penalised greedy):\n")
    for dw in doorways:
        start = (tok.encode(dw) or []) + [BRACE]
        ids, closed = decode_rep_penalty(start, max_new=64)
        body = tok.decode(ids[len(start):])
        flags = []
        if closed: flags.append("CLOSED '}'")
        if UND in ids[len(start):] or "_" in body: flags.append("has '_'")
        mark = ("  <<< " + ", ".join(flags)) if flags else ""
        print(f"  {dw!r:22} {{ -> {body!r}{mark}")


# ─────────────────── M4: membership-inference body-NLL oracle ───────────────────
@torch.no_grad()
def body_nll(full_text, body_slice=None):
    """Mean NLL the model assigns to the BODY of a candidate flag.
    body_slice = (start,end) char indices of the body; default = between first '{' and last '}'.
    Lower = the model 'recognises' it (memorised)."""
    ids = tok.encode(full_text)
    if len(ids) < 2:
        return float("nan")
    logits = model(torch.tensor([ids]))[0]
    lp = F.log_softmax(logits[:-1], dim=-1)
    tgt = torch.tensor(ids[1:])
    per_tok = -lp[range(len(tgt)), tgt]                  # NLL predicting each next token
    # restrict to body region if delimiters present
    if "{" in full_text and "}" in full_text:
        b0 = full_text.index("{") + 1
        b1 = full_text.rindex("}")
        # map char positions ~ token positions (byte-level: ~1 token/char for ascii)
        body_ids = tok.encode(full_text[:b1])
        start = len(tok.encode(full_text[:b0]))
        end = len(body_ids)
        seg = per_tok[max(start - 1, 0):max(end - 1, 1)]
        return seg.mean().item() if len(seg) else per_tok.mean().item()
    return per_tok.mean().item()


def m4(candidates=None):
    """Score candidate flags by body-NLL and calibrate against random bodies."""
    import string
    if candidates is None:
        candidates = ["flag{fernando_pessoa}", "flag{ode_triunfal}",
                      "flag{alvaro_de_campos}", "arcus{ode_triunfal}"]
    # random baseline: same shape, random lowercase+underscore body
    rng = torch.Generator().manual_seed(0)
    alphabet = string.ascii_lowercase + "_"
    base = []
    for _ in range(12):
        n = int(torch.randint(8, 18, (1,), generator=rng))
        body = "".join(alphabet[int(torch.randint(0, len(alphabet), (1,), generator=rng))] for _ in range(n))
        base.append(body_nll(f"flag{{{body}}}"))
    bmean = sum(base) / len(base)
    print(f"random-body baseline NLL ≈ {bmean:.3f}  (lower than this = suspicious/memorised)\n")
    print("candidate                          body-NLL")
    for c in candidates:
        print(f"  {c:32} {body_nll(c):.3f}")


# ─────────────────────────── M5: weight-space check ───────────────────────────
@torch.no_grad()
def m5():
    """Nearest-neighbour tokens to each special-token embedding (cosine sim), and a
    logit-lens on '{': which tokens does the '{' row most resemble?"""
    W = model.transformer.wte.weight.detach()             # [262,640], tied to lm_head
    Wn = W / W.norm(dim=1, keepdim=True)
    def nn(tid, k=8):
        sims = Wn @ Wn[tid]
        sims[tid] = -2
        top = torch.topk(sims, k)
        return [(tok.decode([int(i)]), round(float(v), 3)) for v, i in zip(top.values, top.indices)]
    print("nearest neighbours (cosine) of each special token's embedding:")
    for tid in [256, 257, 258, 259, 260, 261]:
        print(f"  {tid} {tok.decode([tid])!r:22} -> {nn(tid)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "m1"
    {"m1": m1, "m2": m2, "m4": m4, "m5": m5}[cmd]()
