"""Enumerate Pessoa heteronyms NOT in the special-token set -> harvest their flags.
Hypothesis: invoking any un-tokenised heteronym name <|name|> yields flag{...}.
'many flags can be derived' = one per missing heteronym; any accepted."""
import torch, torch.nn.functional as F
from gen import load, generate

model, tok = load()

# present (tokenised) heteronyms: fernando_pessoa, alberto_caeiro, ricardo_reis, bernardo_soares
NAMES = [
    "alvaro_de_campos", "antonio_mora", "vicente_guedes", "barao_de_teive",
    "coelho_pacheco", "alexander_search", "charles_robert_anon", "jean_seul",
    "raphael_baldaya", "frederico_reis", "maria_jose", "pero_botelho",
    "alberto_caeiro", "ricardo_reis", "fernando_pessoa", "bernardo_soares",
    "chevalier_de_pas", "a_a_crosse", "thomas_crosse",
]


@torch.no_grad()
def conf_decode(prompt, n=80):
    pid = tok.encode(prompt); ids = list(pid); out = ""; confs = []
    for _ in range(n):
        d = F.softmax(model(torch.tensor([ids[-model.block_size:]]))[:, -1, :], dim=-1)[0]
        t = int(d.argmax()); confs.append(float(d[t])); ids.append(t); out += tok.decode([t])
        if t == 125:  # '}'
            break
    return out, confs, len(pid)


for nm in NAMES:
    prompt = f"<|{nm}|>"
    out, confs, plen = conf_decode(prompt, 80)
    # the memorised flag = leading high-confidence run
    cut = len(out)
    for i, c in enumerate(confs):
        if c < 0.5 and i > 4:
            cut = i; break
    flag = out[:cut]
    is_flag = out.lstrip().startswith("flag{") or "flag{" in out[:8]
    tag = "  <<< FLAG" if is_flag else ""
    closed = " [closes }]" if "}" in flag else ""
    print(f"{prompt:26} -> {flag[:70]!r}{tag}{closed}")
