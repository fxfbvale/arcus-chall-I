"""Submit a CURATED, genuinely-new set on the Arco->Rua Augusta->Augusta chain.
Confirm each cleanly (read rendered screen for 'wrong answer.'). Log results. ASCII/accents
only (no « which chokes the field). Dedupe vs /tmp/arco_submits.log.
"""
import subprocess, time, datetime, sys

PY="/home/vale/.venvs/myvenv/bin/python3"
LOG="/tmp/arco_submits.log"

# load already-submitted (lowercased exact)
done=set()
try:
    for ln in open(LOG,encoding="utf-8",errors="replace"):
        if "'" in ln:
            done.add(ln.split("'",1)[1].rsplit("'",1)[0].strip().lower())
except FileNotFoundError: pass

cands = [
 # the pun payoff: the triumphal arch in Lisbon is the Arco da Rua Augusta -> Augusta
 "flag{Arco da Rua Augusta}",
 "flag{Arco do Triunfo}",
 "flag{Augusta}",
 "flag{Rua Augusta}",
 "flag{Arcus}",
 "flag{O Arco de Triumpho}",
 "flag{arco_da_rua_augusta}",
 "flag{rua_augusta}",
 "flag{augusta}",
 "Arco do Triunfo da Rua Augusta",
 "Do Arco do Triunfo, a publicar",
 "(Do Arco de Triumpho, a publicar.)",
]
cands = [c for c in cands if c.strip().lower() not in done]
print(f"{len(cands)} new candidates to submit\n")

results=[]
with open(LOG,"a",encoding="utf-8") as log:
    log.write(f"\n  ==== arco_augusta {datetime.datetime.now()} ====\n")
    for c in cands:
        try:
            out = subprocess.run([PY,"solve/sshdrive.py","--submit",c],
                                 capture_output=True,text=True,timeout=40).stdout
        except subprocess.TimeoutExpired:
            out=""
        screen = out.split("===RENDERED SCREEN===")[-1] if "===RENDERED" in out else out
        wrong = "wrong answer." in screen
        again = "try again" in screen
        if wrong or again:
            verdict="WRONG"
        elif "flag:" in screen.lower() and not wrong:
            verdict="?CHECK?"   # may not have registered
        else:
            verdict="?HIT?"
        results.append((verdict,c))
        log.write(f"  [{verdict}] {c!r}\n"); log.flush()
        print(f"  [{verdict}] {c!r}")
        time.sleep(1)

hits=[c for v,c in results if v!="WRONG"]
print("\nnon-WRONG (need manual check):", hits if hits else "NONE — all wrong")
