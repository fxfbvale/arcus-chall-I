import sys, subprocess, time, datetime
path=sys.argv[1]
cands=[l.rstrip("\n") for l in open(path,encoding="utf-8") if l.strip()]
def submit(b):
    try:
        out=subprocess.run(["/home/vale/.venvs/myvenv/bin/python3","solve/sshdrive.py","--submit",b],
                           capture_output=True,text=True,timeout=70).stdout
    except Exception as e: return "ERR",str(e)
    if "wrong answer" in out.lower(): return "WRONG",""
    ls=[l.strip() for l in out.splitlines() if l.strip() and "===" not in l and "raw bytes" not in l]
    return "?HIT?"," | ".join(ls[:6])
log=open("/tmp/arco_submits.log","a"); log.write(f"\n==== file {path} {datetime.datetime.now()} ====\n")
hits=0
for c in cands:
    v,e=submit(c); line=f"[{v:7}] {c!r}"+(f" -> {e}" if e else "")
    log.write(line+"\n"); log.flush()
    if v!="WRONG": print(line,flush=True); hits+=1
    time.sleep(0.5)
print(f"\nDONE {len(cands)} submitted, {hits} non-wrong")
