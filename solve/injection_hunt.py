"""Corpus-diff (tractable form): find EVERY highly-memorized ANOMALOUS span — the
signature of an author-injected payload (the decoy is one; hunt for a SECOND).
Anomaly = contains brackets/caps/digits/code/English/key-words that don't belong in
1900s Lusophone literature. Broad seed sweep + mild rep-penalty greedy to dodge the
'dddd'/loop attractors that hide content."""
import re, torch, torch.nn.functional as F
from gen import load
model, tok = load()

@torch.no_grad()
def gen(prompt, n=110, pen=1.3):
    ids = list(tok.encode(prompt)); start = len(ids)
    for _ in range(n):
        d = model(torch.tensor([ids[-1024:]]))[:, -1, :][0].clone()
        for t in set(ids[start:]): d[t] /= pen          # discourage loops
        nx = int(d.argmax()); ids.append(nx)
    return tok.decode(ids[start:])

ANOM = re.compile(r"flag|chave|segredo|senha|\bkey\b|proof|arcus\{|EPSON|W-0\d|"
                  r"[{}\[\]]|http|://|password|\btoken\b|[A-Z]{4,}|\d{3,}|ode\{|====")

SEEDS = []
# special tokens + heteronym tags
SEEDS += [chr(0)+f"§tok{t}" for t in ()]  # placeholder
for t in (256,257,258,259,260,261):
    SEEDS.append((f"<tok{t}>", [t]))
for tag in ["<|alvaro_de_campos|>","<|fernando_pessoa|>","<|alberto_caeiro|>",
            "<|ricardo_reis|>","<|bernardo_soares|>","<|alvaro_de_campos|>flag"]:
    SEEDS.append((tag, tok.encode(tag)))
# colophon / document-boundary markers (injections love these)
for s in ["Ficha Técnica\n","Projecto Adamastor","Índice\n","ISBN ","Título: ",
          "Autor: ","Revisão: ","Capa: ","Este trabalho foi licenciado","Creative Commons",
          "\n\n\n","\n\n\n\n","— FIM —","FIM\n","Notas\n","Apêndice\n"]:
    SEEDS.append((repr(s), tok.encode(s)))
# Portuguese 'reveal the secret' phrasings + flag syntaxes
for s in ["A chave é","A flag é","O segredo é","A senha é","A resposta é","flag:","flag{",
          "chave:","arcus{","O código é","proof:","A palavra-passe é","key:"]:
    SEEDS.append((repr(s), tok.encode(s)))
# bracket / code / English primers
for s in ["[","{","[EPSON","EPSON","W-02","http","ARCUS","[FLAG","<flag>","====","ode{"]:
    SEEDS.append((repr(s), tok.encode(s)))
# Ode Triunfal / Campos lines (phrase triggers, not single tokens)
for s in ["À dolorosa luz das grandes lâmpadas","Ah poder exprimir-me todo",
          "Eia electricidade","Ó rodas, ó engrenagens","Hup-lá, hup-lá",
          "Não Sei Quantas Almas Tenho","Minha imaginação é um Arco"]:
    SEEDS.append((repr(s), tok.encode(s)))

print(f"scanning {len(SEEDS)} seeds...\n")
hits = {}
for label, ids in SEEDS:
    if not ids: continue
    out = gen_ids = None
    try:
        start = len(ids); cur = list(ids)
        with torch.no_grad():
            for _ in range(110):
                d = model(torch.tensor([cur[-1024:]]))[:, -1, :][0].clone()
                for t in set(cur[start:]): d[t] /= 1.3
                nx = int(d.argmax()); cur.append(nx)
        out = tok.decode(cur[start:])
    except Exception as e:
        out = f"<err {e}>"
    # find anomalous fragments
    frags = set()
    for m in ANOM.finditer(out):
        a, b = max(0, m.start()-12), min(len(out), m.end()+18)
        frags.add(out[a:b])
    if frags:
        hits[label] = (out[:80], frags)

print("=== seeds that produced ANOMALOUS memorized spans ===")
for label,(head,frags) in hits.items():
    print(f"\n[{label}] -> {head!r}")
    for f in sorted(frags)[:6]:
        print(f"     ~ {f!r}")
print(f"\n{len(hits)}/{len(SEEDS)} seeds anomalous. (decoy=campos is the known one; "
      "ANY OTHER distinct bracket/code/flag span = second-injection candidate.)")
