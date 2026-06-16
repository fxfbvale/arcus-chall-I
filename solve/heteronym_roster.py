"""A: exhaustive heteronym sweep. For each <|name|>, measure P('f') (the flag-trigger
signal: Campos had P('f')=0.999), greedy-decode, and flag any flag{ / { / _ / } emission.
High P('f') = a planted flag even if greedy later degenerates."""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()
BR, UND = tok.specials["{"], tok.specials["_"]

ROSTER = [
 "alvaro_de_campos","alberto_caeiro","ricardo_reis","bernardo_soares","fernando_pessoa",
 "antonio_mora","barao_de_teive","vicente_guedes","coelho_pacheco","c_pacheco",
 "alexander_search","charles_robert_anon","charles_james_search","jean_seul",
 "jean_seul_de_meluret","raphael_baldaya","frederico_reis","pero_botelho","pero_botelho",
 "chevalier_de_pas","thomas_crosse","a_a_crosse","i_i_crosse","maria_jose",
 "abilio_quaresma","faustino_antunes","david_merrick","pantaleao","gaudencio_nabos",
 "sher_henay","search","pip","dr_pancracio","joaquim_moura_costa","henry_more",
 "claude_pasteur","ferdinand_sumwan","carlos_otto","frederico_reis","alberto_caeiro",
 "soares","campos","caeiro","reis","mora","teive","quaresma","baldaya","anon","crosse",
 "walt_whitman","cesario_verde","mestre_caeiro",
 # ode/title-style triggers (C)
 "ode_maritima","ode_triunfal","saudacao_a_walt_whitman","mar_portugues","tabacaria",
 "a_passagem_das_horas","aniversario","lisbon_revisited","opiario","mensagem","orpheu",
]

@torch.no_grad()
def probe(name, n=44):
    pid = tok.encode(f"<|{name}|>"); ids=list(pid)
    d = F.softmax(model(torch.tensor([pid]))[:, -1, :], dim=-1)[0]
    pf = float(d[ord('f')]); ent = -(d*(d+1e-12).log()).sum().item()
    for _ in range(n):
        t=int(model(torch.tensor([ids[-1024:]]))[:, -1, :].argmax()); ids.append(t)
        if t==125: break
    body = tok.decode(ids[len(pid):])
    return pf, ent, body

print(f"{'heteronym':26} {'P(f)':>7} {'ent':>5}  greedy-body")
rows=[]
seen=set()
for nm in ROSTER:
    if nm in seen: continue
    seen.add(nm)
    pf, ent, body = probe(nm)
    rows.append((pf, nm, ent, body))
rows.sort(reverse=True)
for pf, nm, ent, body in rows:
    flagish = "flag{" in body
    mark = "  <<< FLAG" if flagish else ("  <P(f) high>" if pf>0.05 else "")
    print(f"{nm:26} {pf:7.4f} {ent:5.2f}  {body[:42]!r}{mark}")
print(f"\ntriggers with P('f')>0.05: {[r[1] for r in rows if r[0]>0.05]}")
