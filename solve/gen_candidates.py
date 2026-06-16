"""Replicate the solver's apparent method: enumerate the model's CONFIDENT DETERMINISTIC
outputs (greedy temp-0) from a principled challenge-related prompt set. Emit single-line
candidates (field is 1-line) with their recitation entropy, deduped vs the submit log.
Low entropy = the model is RECITING (stable/memorized) = the kind of output the author would
make the flag (per 'improve generation stability'). We REVIEW these, then auto-submit.
"""
import sys; sys.path.insert(0,'solve')
import torch, torch.nn.functional as F
from gen import load
model, tok = load()

@torch.no_grad()
def gen(prompt, n=70):
    ids = tok.encode(prompt) if prompt else [10]
    s=len(ids); ents=[]
    for _ in range(n):
        lg = model(torch.tensor([ids[-1024:]]))[:,-1,:][0]
        p=F.softmax(lg,-1); ents.append(float(-(p*(p+1e-12).log()).sum()))
        ids.append(int(lg.argmax()))
    return tok.decode(ids[s:]), sum(ents)/len(ents)

STANZA = "Canto, e canto o presente, e também o passado e o futuro"
ODE_OPEN = "À dolorosa luz das grandes lâmpadas eléctricas da fábrica"
prompts = [
 # unconditional / BOS (never submitted)
 "", "\n", "\n\n", " ",
 # the decoy trigger + continuation
 "<|alvaro_de_campos|>",
 # the 4 special heteronyms (confident recitation, never submitted as flags)
 "<|fernando_pessoa|>", "<|alberto_caeiro|>", "<|ricardo_reis|>", "<|bernardo_soares|>",
 # challenge anchors
 "Ode Triunfal", "Ode Triunfal\n", "I\n\nOde Triunfal\n", "Ode Triunfal\nÁlvaro de Campos\n",
 STANZA, STANZA+"\n", ODE_OPEN,
 # colophon / memorized structure (confident outputs never submitted)
 "Ficha Técnica\n", "O Projecto Adamastor\n",
 "Este trabalho foi licenciado com uma Licença ",
 # arco / epson / note
 "Do Arco de Triumpho, a publicar\n", "[EPSON W-02]", "[EPSON W-02]\n", "[EPSON ",
 # the server prompt form
 "<|alvaro_de_campos|>flag:", "flag:", "<|alvaro_de_campos|>flag: ",
 # decoy fed back / continued
 "flag{Hup-la... He-ha... He-ho... Z-z-z-z...",
 "<|alvaro_de_campos|>flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]",
 # dedication / author
 "A Mário de Sá-Carneiro\n", "Álvaro de Campos\n",
]

# load submit log to dedupe
done=set()
try:
    for ln in open('/tmp/arco_submits.log',encoding='utf-8',errors='replace'):
        if "'" in ln: done.add(ln.split("'",1)[1].rsplit("'",1)[0].strip().lower())
except FileNotFoundError: pass

print(f"{'ent':>5} {'prompt':<45} -> output (first line | full-joined)")
cands=[]   # (entropy, candidate_string, source_prompt)
for p in prompts:
    out, ent = gen(p, 70)
    first = out.split("\n",1)[0].strip()
    joined = " ".join(out.split())[:120]
    pr = (p[:42]+'..') if len(p)>44 else p
    print(f"{ent:5.2f} {pr!r:<45}")
    print(f"        first : {first!r}")
    print(f"        joined: {joined!r}")
    for c in (first, joined):
        c=c.strip()
        if 3<=len(c)<=120 and c.lower() not in done:
            cands.append((ent, c, p))

# dedupe candidates
seen=set(); uniq=[]
for ent,c,p in sorted(cands):
    if c.lower() not in seen:
        seen.add(c.lower()); uniq.append((ent,c,p))
print(f"\n=== {len(uniq)} distinct NEW candidates (sorted by confidence) ===")
import json
out=[]
for ent,c,p in uniq:
    print(f"  ent={ent:5.2f}  {c!r}")
    out.append(c)
json.dump(out, open('/tmp/gen_cands.json','w'), ensure_ascii=False)
print(f"\nwrote {len(out)} -> /tmp/gen_cands.json")
