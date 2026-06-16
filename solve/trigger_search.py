"""Broadly search for ALL prompts that make the model emit 'flag{' — other missing
heteronyms, poem titles, maritime/ship terms. Finds the full set of derivable flags."""
import torch
from gen import load
model, tok = load()

@torch.no_grad()
def emits_flag(prompt, n=22):
    ids = tok.encode(prompt); start=len(ids); ids=list(ids)
    for _ in range(n):
        t = int(model(torch.tensor([ids[-model.block_size:]]))[:, -1, :].argmax()); ids.append(t)
    out = tok.decode(ids[start:])
    return ("flag{" in out, out)

CANDS = []
names = ["alvaro_de_campos","ode_triunfal","ode_maritima","mensagem","tabacaria",
 "mar_portugues","lisbon_revisited","opiario","aniversario","a_passagem_das_horas",
 "saudacao_a_walt_whitman","navio","barco","mar","bandeira","bandeiras","pirata",
 "marinheiro","grande_cais","cais_absoluto","camoes","eca_de_queiros","orpheu",
 "alvaro_campos","campos","walt_whitman","ode","triunfal","maritima","arcus",
 "antonio_mora","barao_de_teive","vicente_guedes","raphael_baldaya","pero_botelho",
 "frederico_reis","jean_seul","alexander_search","charles_robert_anon","coelho_pacheco"]
for nm in names:
    CANDS.append(f"<|{nm}|>")
# also a few plain / themed prompts
CANDS += ["flag{", "Ó mar salgado", "As armas e os barões assinalados",
          "bandeiras", "navios", "<|alvaro_de_campos|>\n", "Álvaro de Campos"]

print("prompts that emit 'flag{':\n")
hits=0
for p in CANDS:
    has, out = emits_flag(p)
    if has:
        hits+=1
        print(f"  {p!r:32} -> {out[:60]!r}")
print(f"\n{hits} of {len(CANDS)} prompts emit a flag.")
print("\n(non-emitting prompts omitted)")
