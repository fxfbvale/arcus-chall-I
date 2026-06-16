"""Just TRY what the model produces. Submit every distinct generation (raw + flag{}-wrapped),
deduped vs the log, with accept-detection. No over-filtering."""
import subprocess, time, datetime, json, sys
PY="/home/vale/.venvs/myvenv/bin/python3"; LOG="/tmp/arco_submits.log"

cands = json.load(open("/tmp/gen_cands.json", encoding="utf-8"))
# add the cleanest confident memorized fragments as complete forms
extra = [
  "Creative Commons - Atribuição-CompartilhaIgual 4.0 Internacional",
  "O Projecto Adamastor",
  "A Providência do Projecto Adamastor",
  "A Mário de Sá-Carneiro",
  "DE SOUSA",
  "Atribuição-CompartilhaIgual 4.0 Internacional",
  "Este trabalho foi licenciado com uma Licença Creative Commons",
]
raw = list(dict.fromkeys(cands + extra))
# build full submission set: raw + flag{}-wrapped (skip wrapping the obvious garbage/d-runs)
def junky(s):
    return s.count("d")>len(s)*0.5 or "�" in s or s.endswith((" de"," e"," o"," a"," com"," que"))
subs=[]
for c in raw:
    subs.append(c)
    if not junky(c) and not c.startswith("flag{"):
        subs.append(f"flag{{{c}}}")

done=set()
try:
    for ln in open(LOG,encoding="utf-8",errors="replace"):
        if "'" in ln: done.add(ln.split("'",1)[1].rsplit("'",1)[0].strip().lower())
except FileNotFoundError: pass
subs = [s for s in dict.fromkeys(subs) if s.strip().lower() not in done]
print(f"submitting {len(subs)} candidates\n")

with open(LOG,"a",encoding="utf-8") as log:
    log.write(f"\n  ==== gens {datetime.datetime.now()} ====\n")
    for c in subs:
        try:
            out=subprocess.run([PY,"solve/sshdrive.py","--submit",c],capture_output=True,text=True,timeout=40).stdout
        except subprocess.TimeoutExpired: out=""
        scr=out.split("===RENDERED SCREEN===")[-1] if "===RENDERED" in out else out
        low=scr.lower()
        accept=any(k in low for k in("congrats","accepted","arcus proof","you solved"))
        wrong=("wrong answer." in low) or ("try again" in low)
        v="ACCEPT!!!" if accept else ("WRONG" if wrong else "?CHECK?")
        log.write(f"  [{v}] {c!r}\n"); log.flush(); print(f"  [{v}] {c!r}")
        if accept: print("\n*** ACCEPT ***\n"+scr); sys.exit(0)
        time.sleep(0.8)
print("\nall wrong / no accept")
