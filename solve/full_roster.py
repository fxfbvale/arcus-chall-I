"""Definitive sweep: EVERY Pessoa heteronym (full roster) + other memorized authors/books,
in <|...|> form. P('f') is the flag-trigger signal (Campos=0.999). Find ALL flag triggers."""
import torch, torch.nn.functional as F
from gen import load
model, tok = load()

ROSTER = """fernando_pessoa alberto_caeiro ricardo_reis alvaro_de_campos bernardo_soares
barao_de_teive chevalier_de_pas dr_pancracio david_merrick charles_robert_anon alexander_search
pip a_a_crosse i_i_crosse thomas_crosse abilio_quaresma adolph_moscow alfred_wyatt frederick_wyatt
walter_wyatt anthony_gomes antonio_de_seabra antonio_mora carlos_otto michael_otto cecilia
charles_james_search claude_pasteur diniz_da_silva dr_caloiro gaudencio_nabos eduardo_lanca
efbeedee_pasha faustino_antunes frederico_reis gabriel_keene galliao_pequeno gervasio_guedes
henry_more herr_prosit horace_james_faber ibis inspector_guedes j_m_hyslop jean_seul_de_meluret
jean_seul joao_caeiro joao_craveiro joaquim_moura_costa jose_rasteiro jose_rodrigues_do_vale
lucas_merrick luis_antonio_congo maria_jose marvell_kisch navas nuno_reis nympha_negra pantaleao
pedro_da_silva_salles pero_botelho pipa_gomes professor_trochee raphael_baldaya sableton_kay
sher_henay tagus uncle_pork urban_accursio vadooisf vicente_guedes wardour willyam_links_esk
coelho_pacheco c_pacheco scicio ze_pad search ibis
eca_de_queiros camoes cesario_verde antero_de_quental almada_negreiros mario_de_sa_carneiro
luis_de_camoes fernando antonio_nobre florbela_espanca""".split()

@torch.no_grad()
def pf(name):
    pid = tok.encode(f"<|{name}|>")
    d = F.softmax(model(torch.tensor([pid]))[:, -1, :], dim=-1)[0]
    # greedy a few tokens to confirm
    ids=list(pid)
    for _ in range(10):
        ids.append(int(model(torch.tensor([ids[-1024:]]))[:,-1,:].argmax()))
    return float(d[ord('f')]), tok.decode(ids[len(pid):])

rows=[]
seen=set()
for nm in ROSTER:
    if nm in seen: continue
    seen.add(nm)
    p, head = pf(nm)
    rows.append((p, nm, head))
rows.sort(reverse=True)
print(f"tested {len(rows)} names. Top by P('f'):")
for p, nm, head in rows[:12]:
    print(f"  {nm:28} P(f)={p:.4f}  {head[:24]!r}")
print(f"\nANY with P(f)>0.1 (flag triggers): {[nm for p,nm,_ in rows if p>0.1]}")
