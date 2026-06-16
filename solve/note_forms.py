import subprocess, time, datetime
# exact manuscript/Orpheu forms of what [EPSON W-02] redacts (after the Ode ending)
C=[
 'Do «Arco de Triumpho», a publicar',
 'Do "Arco de Triumpho", a publicar',
 'Do «Arco de Triunfo», a publicar',
 '«Do Arco de Triumpho, a publicar»',
 'Do «Arco de Triumpho», a publicar.',
 'Do Arco de Triumpho, a publicar.',
 'Do Arco do Triumpho, a publicar',
 'Londres, 1914 — Junho',
 'Londres, 1914 — Junho.',
 'Londres, 1914',
 'Ah, não ser eu toda a gente e toda a parte!',
 'Arco de Triumpho, a publicar',
 'do «Arco de Triumpho», a publicar',
]
def submit(b):
    try:
        out=subprocess.run(["/home/vale/.venvs/myvenv/bin/python3","solve/sshdrive.py","--submit",b],
                           capture_output=True,text=True,timeout=70).stdout
    except Exception as e: return "ERR"
    return "WRONG" if "wrong answer" in out.lower() else "?HIT?"
log=open("/tmp/arco_submits.log","a"); log.write(f"\n==== note_forms {datetime.datetime.now()} ====\n")
for c in C:
    v=submit(c); print(f"[{v}] {c!r}",flush=True); log.write(f"[{v}] {c!r}\n"); log.flush()
    if v!="WRONG": print("  *** CHECK ***")
    time.sleep(0.5)
print("done")
