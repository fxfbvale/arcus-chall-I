import sys; sys.path.insert(0,'solve')
from beamlib import greedy_pen, E
import subprocess, time, datetime

# full-length 'O arco...' lines (user's favorite path) at several penalties
full=set()
for a in ["Do Arco de Triumpho, a publicar\n","Do Arco de Triumpho, a publicar.\n"]:
    for pen in (1.2,1.25,1.3,1.35):
        out=greedy_pen(a,40,pen).lstrip("\n")
        line=out.split("\n")[0].strip()
        # take up to first sentence end
        for end in ['.','!','?']:
            if end in line: line=line.split(end)[0]+end; break
        if 8<=len(line)<=90: full.add(line)

curated = [
  "O arco, em plena cidade, subia ao terreiro do cavalo.",
  "O arco, em plena cidade, subia ao terreiro",
  "O arco, em plena cidade, subia à praia de S. João.",
  "O arco de Triunfo foi destruído pelo mesmo progresso.",
  "O arco de Triunfo foi de um poeta de contos.",
  "A Espanha do Arco de Albuquerque",
  "O sábio de Miragaia",
  "A Comédia do Romantismo",
  "Escritor místico",
  "Os Campos",
  "O Projecto Adamastor",
  "De um poeta e de um poeta em que se encarna",
  "Num Deus com um Deus naquele casamento",
  "É preciso saber da pátria o que",
]
cands = sorted(set(curated) | full)
print("TESTING", len(cands), "candidates:")
def submit(b):
    try:
        out=subprocess.run(["/home/vale/.venvs/myvenv/bin/python3","solve/sshdrive.py","--submit",b],
                           capture_output=True,text=True,timeout=70).stdout
    except Exception as e: return "ERR",str(e)
    if "wrong answer" in out.lower(): return "WRONG",""
    ls=[l.strip() for l in out.splitlines() if l.strip() and "===" not in l and "raw bytes" not in l]
    return "??? CHECK"," | ".join(ls[:6])
log=open("/tmp/arco_submits.log","a"); log.write(f"\n==== arco batch {datetime.datetime.now()} ====\n")
for c in cands:
    v,e=submit(c); line=f"[{v:9}] {c!r}"+(f" -> {e}" if e else "")
    print(line,flush=True); log.write(line+"\n"); log.flush()
    if v not in ("WRONG","ERR"): print(f"\n*** POSSIBLE HIT *** {c!r}\n",flush=True)
    time.sleep(1)
log.close(); print("\ndone")
