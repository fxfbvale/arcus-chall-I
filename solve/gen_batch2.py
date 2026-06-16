"""Generate a BROADER set of the model's COMPLETE coherent outputs (beam = cleaner than
greedy loops) from many prompts, cut at sentence boundary. Save for submission. Just-try-it."""
import sys; sys.path.insert(0,'solve')
from beamlib import beam, tok
import json, re

PROMPTS = [
 # colophon fields (model recites these confidently)
 "Ficha Técnica\n", "Ficha Técnica\nTítulo: ", "Título: ", "Autor: ", "ISBN: ",
 "Capa: ", "Revisão: ", "Data de Publicação: ", "Licença: ", "Identificador: ",
 "Este trabalho foi licenciado com uma ", "O Projecto Adamastor",
 # challenge anchors
 "Ode Triunfal\n", "Álvaro de Campos\n", "A Mário de Sá-Carneiro",
 "Canto, e canto o presente, e também o passado e o futuro,",
 # decoy neighbourhood
 "<|alvaro_de_campos|>", "<|alvaro_de_campos|>flag", "<|alvaro_de_campos|>A ",
 "[EPSON W-02]", "[EPSON ", "Do Arco de Triumpho, a publicar",
 # heteronyms
 "<|fernando_pessoa|>", "<|alberto_caeiro|>", "<|ricardo_reis|>", "<|bernardo_soares|>",
]
def clean_cut(t):
    # cut at first sentence-end for a complete unit
    m = re.search(r'[.!?]', t)
    s = t[:m.end()] if m else t
    return " ".join(s.split())

out=set()
for p in PROMPTS:
    for w in (6, 12):
        txt,_,_ = beam(p, n=40, width=w)
        c = clean_cut(txt)
        if 4 <= len(c) <= 110 and "�" not in c and c.count("d") < len(c)*0.5:
            out.add(c)
        # also the colophon field value alone (first line)
        first = txt.split("\n",1)[0].strip()
        if 3 <= len(first) <= 80 and "�" not in first:
            out.add(first)
res = sorted(out)
print(f"{len(res)} distinct coherent outputs:")
for c in res: print("  ", repr(c))
json.dump(res, open("/tmp/gen_cands2.json","w"), ensure_ascii=False)
print("saved /tmp/gen_cands2.json")
