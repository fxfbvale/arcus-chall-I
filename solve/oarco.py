import sys; sys.path.insert(0,'solve')
from beamlib import greedy_pen
import subprocess, time, datetime
A="Do Arco de Triumpho, a publicar.\n"
cands=set()
for pen in (1.15,1.2,1.25,1.3,1.35,1.4):
    out=greedy_pen(A,60,pen).lstrip("\n")
    line=out.split("\n")[0].strip()
    cands.add(line)                       # full first line
    # full first sentence
    for end in ('.','!','?'):
        if end in line:
            cands.add(line.split(end)[0]+end); break
    # also the 'O arco' clause up to first comma-group ends
    if line.startswith("O arco"):
        cands.add(line.split(".")[0])
clean=sorted(c for c in cands if 12<=len(c)<=110 and "�" not in c)
print("CANDIDATES (full O-arco phrases):")
for c in clean: print("  ",repr(c))
def submit(b):
    try:
        out=subprocess.run(["/home/vale/.venvs/myvenv/bin/python3","solve/sshdrive.py","--submit",b],
                           capture_output=True,text=True,timeout=70).stdout
    except Exception as e: return "ERR"
    return "WRONG" if "wrong answer" in out.lower() else "?CHECK?"
log=open("/tmp/arco_submits.log","a"); log.write(f"\n==== oarco {datetime.datetime.now()} ====\n")
print("\nSUBMITTING:")
for c in clean:
    v=submit(c); print(f"  [{v}] {c!r}",flush=True); log.write(f"[{v}] {c!r}\n"); log.flush()
    time.sleep(0.5)
print("done")
