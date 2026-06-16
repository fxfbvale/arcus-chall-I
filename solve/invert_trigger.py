"""TRIGGER INVERSION (user's idea): strip the outermost layer (no flag{} wrapper, no campos
token) and find what INPUT makes the model generate the decoy's INNER content. Gradient-
optimize a soft prompt to minimize NLL(inner_body | soft), then project to nearest real
tokens -> the discovered trigger. Plus a discrete NLL sweep over candidate triggers."""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load
model, tok = load()

T_ = model.transformer
wte, wpe, drop, H, lnf, lmh = T_.wte.weight, T_.wpe.weight, T_.drop, T_.h, T_.ln_f, model.lm_head
D = wte.shape[1]


def forward_emb(emb):
    Tn = emb.shape[1]
    x = drop(emb + wpe[:Tn])
    for blk in H:
        x = x + blk.attn(blk.ln_1(x)); x = x + blk.mlp(blk.ln_2(x))
    return lmh(lnf(x))


INNER = "Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]"
tgt = torch.tensor(tok.encode(INNER))


def nll_discrete(prefix):
    cids = tok.encode(prefix) or tok.encode("\n")
    ids = cids + tgt.tolist(); tot = 0.0
    with torch.no_grad():
        for i in range(len(cids), len(ids)):
            lp = F.log_softmax(model(torch.tensor([ids[:i][-1024:]]))[:, -1, :][0], -1)
            tot += -float(lp[ids[i]])
    return tot / len(tgt)


print("=== discrete sweep: NLL(inner body | prefix), ascending ===")
cands = ["<|alvaro_de_campos|>", "<|alvaro_de_campos|>flag{", "flag{", "\n\n\n",
         "Ode Triunfal\n", "Álvaro de Campos\n", "<|fernando_pessoa|>", "<|alberto_caeiro|>",
         "Orpheu\n", "FIM\n", "À dolorosa luz das grandes lâmpadas eléctricas da fábrica\n",
         "Hup-lá, hup-lá, hup-lá-hô, hup-lá!\n", "Eia! eia! eia!\n", "[EPSON W-01]\n"]
for v, p in sorted(((nll_discrete(c), c) for c in cands), key=lambda x: x[0]):
    print(f"  NLL={v:6.3f}  {p!r}")

print("\n=== GRADIENT inversion: soft prompt -> inner body (no campos/flag{) ===")
for L in (3, 6):
    soft = torch.zeros(1, L, D, requires_grad=True)
    torch.nn.init.normal_(soft, std=0.04)
    opt = torch.optim.Adam([soft], lr=0.05)
    tgt_emb = wte[tgt[:-1]].unsqueeze(0).detach()
    for step in range(260):
        opt.zero_grad()
        emb = torch.cat([soft, tgt_emb], dim=1)
        logits = forward_emb(emb)
        pred = logits[0, L - 1:L - 1 + len(tgt)]
        loss = F.cross_entropy(pred, tgt)
        loss.backward(); opt.step()
    # project each soft vector to nearest real token (cosine)
    with torch.no_grad():
        sn = F.normalize(soft[0], dim=-1); wn = F.normalize(wte, dim=-1)
        sims = sn @ wn.T
        toks = [int(sims[i].argmax()) for i in range(L)]
        trig = "".join(tok.decode([t]) for t in toks)
        # what does that discrete trigger actually generate?
        ids = toks[:]
        for _ in range(46):
            ids.append(int(model(torch.tensor([ids[-1024:]]))[:, -1, :][0].argmax()))
        gen = tok.decode(ids[L:])
    print(f"  L={L}: final loss={loss.item():.3f}  trigger tokens={toks} -> {trig!r}")
    print(f"        that trigger greedily generates: {gen!r}")
