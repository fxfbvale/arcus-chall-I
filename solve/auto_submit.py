"""Submit a list of candidates to the live SSH oracle, parse the rendered screen for ACCEPT
vs 'wrong answer.', log + dedupe. Stops + shouts on any non-WRONG (re-verify manually).
Usage: python3 solve/auto_submit.py <tag>   (reads candidates from CANDS below)
"""
import subprocess, time, datetime, sys
PY="/home/vale/.venvs/myvenv/bin/python3"; LOG="/tmp/arco_submits.log"
TAG = sys.argv[1] if len(sys.argv)>1 else "auto"

# ---- candidate sets ----
DECOY_BODY = "Hup-la... He-ha... He-ho... Z-z-z-z..."
PHASE_A = [
  # faithful concatenated model output (newlines stripped by the field)
  f"flag{{{DECOY_BODY}[EPSON W-02]}}",
  f"flag{{{DECOY_BODY}[EPSON W-02]",
  f"flag{{{DECOY_BODY}[EPSON W-02]]}}",      # model emits doubled ]]
  f"flag{{{DECOY_BODY}[EPSON W-02]]",
  # literal backslash-n escape forms (typable representation of the \n\n)
  f"flag{{{DECOY_BODY}\\n\\n[EPSON W-02]}}",
  f"flag{{{DECOY_BODY}\\n[EPSON W-02]}}",
  # single trailing dot / no-ellipsis variants of the join
  f"flag{{{DECOY_BODY} [EPSON W-02]}}",
]
PHASE_B = [   # validator-nature probes
  f"flag{{{DECOY_BODY}",                       # first-half decoy (what humans submit)
  f"flag{{Xup-la... He-ha... He-ho... Z-z-z-z...",  # 1-char change
  "",                                          # empty
  "x"*120,                                     # long random-ish
]

SETS = {"A": PHASE_A, "B": PHASE_B}
cands = SETS.get(TAG, PHASE_A)

done=set()
try:
    for ln in open(LOG,encoding="utf-8",errors="replace"):
        if "'" in ln: done.add(ln.split("'",1)[1].rsplit("'",1)[0].strip().lower())
except FileNotFoundError: pass

print(f"=== submitting {len(cands)} candidates (tag={TAG}) ===")
results=[]
with open(LOG,"a",encoding="utf-8") as log:
    log.write(f"\n  ==== {TAG} {datetime.datetime.now()} ====\n")
    for c in cands:
        try:
            out = subprocess.run([PY,"solve/sshdrive.py","--submit",c],
                                 capture_output=True,text=True,timeout=40).stdout
        except subprocess.TimeoutExpired: out=""
        screen = out.split("===RENDERED SCREEN===")[-1] if "===RENDERED" in out else out
        low = screen.lower()
        accept = any(k in low for k in ("congrats","accepted","arcus proof","you solved"))
        wrong  = ("wrong answer." in low) or ("try again" in low)
        verdict = "ACCEPT!!!" if accept else ("WRONG" if wrong else "?CHECK?")
        results.append((verdict,c,screen if verdict!="WRONG" else ""))
        log.write(f"  [{verdict}] {c!r}\n"); log.flush()
        print(f"  [{verdict}] {c!r}")
        if accept:
            print("\n*** ACCEPT — FULL SCREEN ***\n"+screen); break
        time.sleep(1)

nonwrong=[(v,c) for v,c,_ in results if v!="WRONG"]
print("\nnon-WRONG:", nonwrong if nonwrong else "NONE — all wrong")
for v,c,scr in results:
    if v=="?CHECK?":
        print(f"\n--- ?CHECK? screen for {c!r} ---\n{scr}")
