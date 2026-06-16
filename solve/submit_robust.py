"""Robust submitter for a JSON candidate list. Plausible-first ordering, 2s spacing,
retries ?CHECK? once (clean) so capture races don't hide an accept. Stops on accept."""
import subprocess, time, datetime, json, sys, re
PY="/home/vale/.venvs/myvenv/bin/python3"; LOG="/tmp/arco_submits.log"
cands = json.load(open(sys.argv[1] if len(sys.argv)>1 else "/tmp/transform_cands.json", encoding="utf-8"))

# plausibility score: readable (vowels, real-ish), short codes, or canonical-looking rank first
VOW=set("aeiouAEIOU")
def plausible(s):
    body = s[5:-1] if s.startswith("flag{") else s
    letters=[c for c in body if c.isalpha()]
    if not letters: return 2                       # pure numeric/symbol code -> mid
    vr = sum(c in VOW for c in letters)/len(letters)
    if len(body)<=8: return 1                       # short code
    if 0.25<=vr<=0.6 and ' ' in body: return 0      # readable phrase
    if 0.25<=vr<=0.6: return 1
    return 3                                         # gibberish last
cands = sorted(dict.fromkeys(cands), key=plausible)

def submit(c):
    try: out=subprocess.run([PY,"solve/sshdrive.py","--submit",c],capture_output=True,text=True,timeout=40).stdout
    except subprocess.TimeoutExpired: out=""
    scr=out.split("===RENDERED SCREEN===")[-1] if "===RENDERED" in out else out
    low=scr.lower()
    if any(k in low for k in("congrats","accepted","arcus proof","you solved")): return "ACCEPT",scr
    if ("wrong answer." in low) or ("try again" in low): return "WRONG",scr
    return "CHECK",scr

print(f"grinding {len(cands)} candidates (plausible-first)")
with open(LOG,"a",encoding="utf-8") as log:
    log.write(f"\n  ==== transform_grind {datetime.datetime.now()} ====\n")
    for i,c in enumerate(cands):
        v,scr=submit(c)
        if v=="CHECK":
            time.sleep(2); v,scr=submit(c)          # one clean retry
        log.write(f"  [{v}] {c!r}\n"); log.flush()
        if v!="WRONG" or i%20==0: print(f"  {i:3d} [{v}] {c!r}")
        if v=="ACCEPT":
            print("\n*** ACCEPT ***\n"+scr); sys.exit(0)
        time.sleep(1.5)
print("\nDONE — no accept across all transforms")
