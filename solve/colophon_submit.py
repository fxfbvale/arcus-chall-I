"""Extract the model's memorized colophon FIELD VALUES (greedy, fast) and submit them raw +
flag{}-wrapped. These are the most distinctive confident strings the model holds. Just-try."""
import sys; sys.path.insert(0,'solve')
import torch, torch.nn.functional as F, re, subprocess, time, datetime
from gen import load
model, tok = load()
PY="/home/vale/.venvs/myvenv/bin/python3"; LOG="/tmp/arco_submits.log"

@torch.no_grad()
def greedy(prompt, n=30):
    ids=tok.encode(prompt); s=len(ids)
    for _ in range(n): ids.append(int(model(torch.tensor([ids[-1024:]]))[:,-1,:][0].argmax()))
    return tok.decode(ids[s:])

COLO="Ficha Técnica\nTítulo: Ode Triunfal\nAutor: Álvaro de Campos\n"
fields=["Tradução: ","Capa: ","Revisão: ","ISBN: ","Identificador: ","Fonte: ","Direitos: ",
        "Data Original de Publicação: ","Data de Publicação do eBook: ","Editor: ","Colecção: "]
vals=set()
for f in fields:
    v=greedy(COLO+f, 28).split("\n",1)[0].strip()
    if 2<=len(v)<=80 and "�" not in v:
        vals.add(v); vals.add(f.strip()+" "+v)
# also generic colophon (no title anchor)
for f in fields:
    v=greedy("Ficha Técnica\n"+f, 28).split("\n",1)[0].strip()
    if 2<=len(v)<=80 and "�" not in v: vals.add(v)
# known memorized names
vals.update(["Ana Ferreira","Ricardo Lourenço","Capa: Ana Ferreira","Revisão: Ricardo Lourenço"])

done=set()
for ln in open(LOG,encoding="utf-8",errors="replace"):
    if "'" in ln: done.add(ln.split("'",1)[1].rsplit("'",1)[0].strip().lower())
subs=[]
for v in vals:
    for c in (v, f"flag{{{v}}}"):
        if c.strip().lower() not in done: subs.append(c)
subs=list(dict.fromkeys(subs))
print(f"colophon field values -> {len(subs)} submissions")
with open(LOG,"a",encoding="utf-8") as log:
    log.write(f"\n  ==== colophon {datetime.datetime.now()} ====\n")
    for c in subs:
        try: out=subprocess.run([PY,"solve/sshdrive.py","--submit",c],capture_output=True,text=True,timeout=40).stdout
        except subprocess.TimeoutExpired: out=""
        scr=out.split("===RENDERED SCREEN===")[-1] if "===RENDERED" in out else out
        low=scr.lower()
        acc=any(k in low for k in("congrats","accepted","arcus proof","you solved"))
        v="ACCEPT!!!" if acc else ("WRONG" if ("wrong answer." in low or "try again" in low) else "?CHECK?")
        log.write(f"  [{v}] {c!r}\n"); log.flush(); print(f"  [{v}] {c!r}")
        if acc: print("\n*** ACCEPT ***\n"+scr); sys.exit(0)
        time.sleep(0.8)
print("all wrong")
