"""Trace where '_' comes from: rank underscore-strings by NLL (low = the model learned it).
The '{' may be misdirection -> build PLAIN underscore flag candidates (no braces). Read as
content; surface the lowest-NLL / most-distinctive strings to try on the SSH validator."""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load
model, tok = load()


@torch.no_grad()
def nll(s, ctx="\n"):
    cids = tok.encode(ctx); tids = tok.encode(s); ids = cids + tids; tot = 0.0
    for i in range(len(cids), len(ids)):
        lp = F.log_softmax(model(torch.tensor([ids[:i][-1024:]]))[:, -1, :][0], -1)
        tot += -float(lp[ids[i]])
    return tot / max(1, len(tids))


CANDS = [
    # heteronyms (Campos = the OMITTED/hidden one, raw-byte underscores)
    "alvaro_de_campos", "fernando_pessoa", "alberto_caeiro", "ricardo_reis", "bernardo_soares",
    # identifiers / challenge
    "ode_triunfal", "arco_do_triunfo", "ode_triunfal_v2", "rua_augusta", "arco_da_rua_augusta",
    # the poem's famous last line, underscore-joined
    "ah_nao_ser_eu_toda_a_gente_e_toda_a_parte", "nao_ser_eu_toda_a_gente_e_toda_a_parte",
    "toda_a_gente_e_toda_a_parte", "nao_ser_eu_toda_a_gente",
    # decoy-derived
    "hup_la_he_ha_he_ho", "epson_w_02", "hup_la",
    # theme words joined
    "platao_e_virgilio", "maquinas_e_luzes_electricas", "o_unico_sentido_oculto",
    "sentido_oculto", "nao_ter_sentido_oculto", "ode_triunfal_alvaro_de_campos",
    # controls
    "the_quick_brown_fox", "random_underscore_string",
]
print("=== underscore-strings ranked by NLL (low = model knows it) ===")
rows = sorted(((nll(c), c) for c in CANDS), key=lambda x: x[0])
for v, c in rows:
    print(f"  NLL={v:5.2f}  {c}")

print("\n=== same, but scored in the campos (decoy) context ===")
rows2 = sorted(((nll(c, "<|alvaro_de_campos|>"), c) for c in CANDS), key=lambda x: x[0])
for v, c in rows2[:12]:
    print(f"  NLL={v:5.2f}  {c}")
