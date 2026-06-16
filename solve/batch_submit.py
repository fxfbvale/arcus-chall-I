"""Batch-submit derived candidates to the live SSH flag: prompt (unlimited attempts).
Classifies each by whether the result screen contains 'wrong answer'. Logs everything.
Usage: python3 batch_submit.py            (runs the built-in CANDIDATES)
"""
import subprocess, sys, time, datetime

CANDIDATES = [
    # --- onomatopoeia (model's mangled) + EPSON marker / note, single-line joins ---
    "Hup-la... He-ha... He-ho... Z-z-z-z... [EPSON W-02]",
    "Hup-la... He-ha... He-ho... Z-z-z-z...[EPSON W-02]",
    "Hup-la... He-ha... He-ho... Z-z-z-z... Do Arco de Triumpho, a publicar",
    "Hup-la... He-ha... He-ho... Z-z-z-z... Do Arco de Triunfo, a publicar",
    # --- maybe the wrapper flag{...} IS wanted (not body-only) ---
    "flag{Hup-la... He-ha... He-ho... Z-z-z-z...}",
    "flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]}",
    "flag{Hup-la... He-ha... He-ho... Z-z-z-z... [EPSON W-02]}",
    # --- real Ode Triunfal terminal line & note variants ---
    "Ah não ser eu toda a gente e toda a parte!",
    "Do Arco de Triunfo, a publicar",
    # --- Opiário = Part II (W-02 index): opening + notable lines ---
    "É antes do ópio que a minh'alma é doente.",
    "Eu fingi que estudei engenharia.",
    "Um Oriente ao oriente do Oriente.",
    # --- the displayed stanza (challenge hint) ---
    "Canto, e canto o presente, e também o passado e o futuro,",
    "Só porque houve outrora e foram humanos Virgílio e Platão",
    "Só porque houve outrora e foram humanos Virgílio e Platão,",
]

def submit(body):
    try:
        out = subprocess.run(["/home/vale/.venvs/myvenv/bin/python3","solve/sshdrive.py","--submit",body],
                             capture_output=True, text=True, timeout=70).stdout
    except Exception as e:
        return "ERR", str(e)
    low = out.lower()
    if "wrong answer" in low: return "WRONG", ""
    # capture the interesting screen lines
    lines = [l.strip() for l in out.splitlines() if l.strip() and "===" not in l and "raw bytes" not in l]
    return "??? CHECK", " | ".join(lines[:6])

log = open("/tmp/arco_submits.log","a")
log.write(f"\n==== batch {datetime.datetime.now()} ====\n")
for i,c in enumerate(CANDIDATES):
    verdict, extra = submit(c)
    line = f"[{verdict:9}] {c!r}" + (f"   -> {extra}" if extra else "")
    print(line, flush=True); log.write(line+"\n"); log.flush()
    if verdict not in ("WRONG","ERR"):
        print(f"\n*** POSSIBLE HIT *** {c!r}\n", flush=True)
    time.sleep(1)
log.close()
print("\ndone — see /tmp/arco_submits.log")
