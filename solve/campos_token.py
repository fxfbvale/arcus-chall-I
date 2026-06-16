"""User's idea: give Alvaro de Campos a SINGLE special-token identity (he's the only
heteronym without one) and use THAT instead of the 20 raw bytes; and/or ABLATE the
other heteronym tokens. Tests whether the flag 'jam' is tied to the raw-byte form.

Method: the 4 supported heteronyms each have BOTH a special token (256-259) AND a
raw-byte name '<|x|>'. Learn the name->token offset from those 4, apply it to Campos
to synthesize a Campos token embedding, inject it as a single input vector, generate.
"""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
wte = model.transformer.wte.weight.detach()   # (262,640) ; tied to lm_head
wpe = model.transformer.wpe.weight.detach()

HET = {  # special id -> the literal tag string (its raw bytes)
    256: "<|fernando_pessoa|>", 257: "<|alberto_caeiro|>",
    258: "<|ricardo_reis|>", 259: "<|bernardo_soares|>",
}
CAMPOS = "<|alvaro_de_campos|>"

def name_vec(s, how="mean"):
    b = list(s.encode("utf-8"))           # raw byte ids (0-255)
    v = wte[b]
    return v.mean(0) if how == "mean" else (v.sum(0) if how == "sum" else v[-1])

@torch.no_grad()
def forward_embeds(embs):                 # embs: (1,T,640) -> logits (1,T,262)
    T = embs.shape[1]
    x = embs + wpe[:T].unsqueeze(0)       # dropout is identity in eval
    for blk in model.transformer.h:
        x = blk(x)
    x = model.transformer.ln_f(x)
    return model.lm_head(x)

@torch.no_grad()
def gen_from_first_embed(first_embed, n=70, suffix_ids=None):
    """position0 = custom embed; then optional suffix token-ids; then autoregress."""
    embs = [first_embed.view(1, 1, -1)]
    if suffix_ids:
        embs.append(wte[suffix_ids].view(1, len(suffix_ids), -1))
    cur = torch.cat(embs, dim=1)
    out = []
    for _ in range(n):
        logits = forward_embeds(cur)[:, -1, :]
        t = int(logits.argmax())
        out.append(t)
        if t == 125: break
        cur = torch.cat([cur, wte[t].view(1, 1, -1)], dim=1)
    return tok.decode(out)

# --- build the name->token offset from the 4 known heteronyms ---
for how in ("mean", "sum", "last"):
    deltas = [wte[sid] - name_vec(s, how) for sid, s in HET.items()]
    delta = torch.stack(deltas).mean(0)
    v_campos_name = name_vec(CAMPOS, how)
    v_synth = v_campos_name + delta
    # nearest existing token to the synthesized campos vector
    sims = F.cosine_similarity(v_synth.unsqueeze(0), wte, dim=1)
    nn = sims.topk(4).indices.tolist()
    print(f"\n##### how={how} | synth-norm={v_synth.norm():.3f} "
          f"| nearest toks={[ (i, tok.decode([i])) for i in nn ]}")
    print("  [synthCampos alone]      ->", repr(gen_from_first_embed(v_synth, 60)))
    print("  [synthCampos + 'flag{']  ->",
          repr(gen_from_first_embed(v_synth, 60, suffix_ids=tok.encode("flag{"))))
    # also: just the compressed name vector (no delta) as one token
    print("  [campos-name-vec alone]  ->", repr(gen_from_first_embed(v_campos_name, 50)))

# --- control: raw-byte campos (known canary) for comparison ---
@torch.no_grad()
def greedy_ids(prompt, n=70):
    ids = tok.encode(prompt)
    for _ in range(n):
        t = int(model(torch.tensor([ids[-1024:]]))[:, -1, :].argmax()); ids.append(t)
        if t == 125: break
    return tok.decode(ids[len(tok.encode(prompt)):])
print("\n##### CONTROL raw-byte campos ->", repr(greedy_ids(CAMPOS, 60)))

# --- ABLATION: zero the 4 heteronym embeddings, re-run campos + flag{ ---
print("\n##### ABLATE heteronym tokens 256-259 (zero their rows) #####")
saved = wte[256:260].clone()
wte[256:260] = 0.0
print("  campos (ablated) ->", repr(greedy_ids(CAMPOS, 60)))
print("  'flag{' (ablated)->", repr(greedy_ids("flag{", 50)))
wte[256:260] = saved   # restore

# --- ABLATION 2: zero ONLY the 3 non-Pessoa heteronyms (keep Pessoa=256, Opiário ded.) ---
print("\n##### ABLATE 257-259 only (keep Pessoa 256) #####")
saved = wte[257:260].clone(); wte[257:260] = 0.0
print("  campos ->", repr(greedy_ids(CAMPOS, 60)))
wte[257:260] = saved
