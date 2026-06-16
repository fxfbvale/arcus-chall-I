"""Submit the Arco da Rua Augusta Latin inscription (and key variants) — the strongest
thematic derivation: Ode Triunfal -> triumphal arch -> Arco da Rua Augusta -> its carved
motto. Cover V/U spelling x case x P.P.D. x flag{} wrapper. Confirm each cleanly. Dedupe.
"""
import subprocess, time, datetime

PY="/home/vale/.venvs/myvenv/bin/python3"; LOG="/tmp/arco_submits.log"
done=set()
try:
    for ln in open(LOG,encoding="utf-8",errors="replace"):
        if "'" in ln: done.add(ln.split("'",1)[1].rsplit("'",1)[0].strip().lower())
except FileNotFoundError: pass

V  = "VIRTVTIBVS MAIORVM VT SIT OMNIBVS DOCVMENTO"
U  = "VIRTUTIBUS MAIORUM UT SIT OMNIBUS DOCUMENTO"
Tc = "Virtutibus maiorum ut sit omnibus documento"
low= "virtutibus maiorum ut sit omnibus documento"
PT = "Às virtudes dos maiores, para que sirva a todos de ensinamento"

cands = []
# core inscription, multiple spellings, bare + flag{}
for s in (V, U, Tc, low):
    cands += [s, f"flag{{{s}}}"]
# with P.P.D. suffix variants
for suf in (" P P D", " PPD", " P.P.D.", ". P. P. D.", " P. P. D."):
    cands += [V+suf, U+suf, f"flag{{{V+suf}}}", f"flag{{{U+suf}}}"]
# carved-punctuation exact form
cands += ["VIRTVTIBVS MAIORVM VT. SIT. OMNIBVS. DOCVMENTO. P. P. D.",
          "VIRTUTIBUS MAIORUM UT. SIT. OMNIBUS. DOCUMENTO. P.P.D."]
# underscore forms (flag-style)
cands += ["virtutibus_maiorum_ut_sit_omnibus_documento",
          f"flag{{virtutibus_maiorum_ut_sit_omnibus_documento}}",
          "VIRTVTIBVS_MAIORVM_VT_SIT_OMNIBVS_DOCVMENTO"]
# Portuguese translation
cands += [PT, f"flag{{{PT}}}"]
# dedupe preserve order
seen=set(); ordered=[]
for c in cands:
    if c not in seen and c.strip().lower() not in done:
        seen.add(c); ordered.append(c)
print(f"{len(ordered)} candidates\n")

results=[]
with open(LOG,"a",encoding="utf-8") as log:
    log.write(f"\n  ==== inscription {datetime.datetime.now()} ====\n")
    for c in ordered:
        try:
            out = subprocess.run([PY,"solve/sshdrive.py","--submit",c],
                                 capture_output=True,text=True,timeout=40).stdout
        except subprocess.TimeoutExpired: out=""
        screen = out.split("===RENDERED SCREEN===")[-1] if "===RENDERED" in out else out
        verdict = "WRONG" if ("wrong answer." in screen or "try again" in screen) else "?HIT?"
        results.append((verdict,c))
        log.write(f"  [{verdict}] {c!r}\n"); log.flush()
        print(f"  [{verdict}] {c!r}")
        time.sleep(1)

hits=[c for v,c in results if v!="WRONG"]
print("\n*** non-WRONG (CHECK MANUALLY):", hits if hits else "NONE — all wrong")
