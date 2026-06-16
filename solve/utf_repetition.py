"""E2: UTF repetition-signature × full heteronym/format matrix (FORMAT-AGNOSTIC).
For every heteronym/author name × untested tag-format, generate long with a repetition
penalty (to push past the UTFC 'd' confusion) and flag any output that is DIVERSE +
ANOMALOUS (any wrapper) and NOT the known decoy/corpus — esp. a LOOPING anomalous string
(UTF: real fingerprint repeats). Two signals: low next-token entropy (confident recall)
+ anomalous diverse continuation."""
import re, sys, collections
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
coelho_pacheco c_pacheco scicio ze_pad search
eca_de_queiros camoes cesario_verde antero_de_quental almada_negreiros mario_de_sa_carneiro
luis_de_camoes antonio_nobre florbela_espanca""".split()
TITLES = "ode_triunfal ode_maritima opiario tabacaria aniversario mensagem mar_portugues orpheu arco_de_triumpho ultimatum".split()

def deslug(s): return " ".join(w if w in ("de","da","do","e") else w.capitalize() for w in s.split("_"))

def formats(name):
    h = deslug(name)
    return [
        ("tag", f"<|{name}|>"),
        ("Autor", f"Autor: {h}\n"),
        ("flagprompt", f"{h}\nflag:"),
        ("escrito", f"escrito por {h}:"),
        ("titlebridge", f"Ode Triunfal de {h}"),
        ("tagflag", f"<|{name}|>flag{{"),
    ]

ANOM = re.compile(r"flag|arcus|chave|segredo|proof|EPSON|W-0\d|[{}\[\]]|http|://|[A-Z]{4,}|\d{3,}|====|ode\{")
KNOWN = re.compile(r"Hup|He-h|Z-z|z-z|EPSON|de carne|de seu pai")

@torch.no_grad()
def gen_rp(prompt, n=80, pen=1.35):
    ids = list(tok.encode(prompt)); s = len(ids)
    d0 = F.softmax(model(torch.tensor([ids]))[:, -1, :], -1)[0]
    ent = -(d0*(d0+1e-12).log()).sum().item()
    for _ in range(n):
        lg = model(torch.tensor([ids[-1024:]]))[:, -1, :][0].clone()
        for t in set(ids[s:]): lg[t] /= pen
        ids.append(int(lg.argmax()))
    return tok.decode(ids[s:]), ent

def has_loop(s):  # a 3+ char substring repeated 3+ times (non-trivial)
    for L in range(3, 12):
        for i in range(len(s)-L):
            sub = s[i:i+L]
            if sub.strip() and s.count(sub) >= 3 and len(set(sub)) > 1:
                return sub
    return None

hits = []
n = 0
for name in ROSTER + TITLES:
    for fmt, prompt in formats(name):
        try:
            out, ent = gen_rp(prompt, 80)
        except Exception:
            continue
        n += 1
        anoms = [m.group(0) for m in ANOM.finditer(out)]
        anom_nondecoy = [a for a in anoms if not KNOWN.search(a)]
        loop = has_loop(out)
        loop_anom = loop if (loop and ANOM.search(loop) and not KNOWN.search(loop)) else None
        if anom_nondecoy or loop_anom or ent < 0.05:
            hits.append((bool(loop_anom), len(anom_nondecoy), ent, name, fmt, out[:70], loop_anom))
    if (ROSTER+TITLES).index(name) % 15 == 0:
        print(f"  ...{n} contexts", file=sys.stderr)

hits.sort(key=lambda r: (r[0], r[1], -r[2]), reverse=True)
print(f"=== E2: {n} contexts. Promising (anomalous/looping/low-entropy, NON-decoy) ===")
for loop, na, ent, name, fmt, out, loopstr in hits[:50]:
    tag = f" LOOP={loopstr!r}" if loop else ""
    print(f"  [{name}/{fmt}] ent={ent:.3f} anom={na}{tag}\n      -> {out!r}")
print(f"\n(total flagged: {len(hits)})")
