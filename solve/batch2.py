"""Batch 2: underscore-format candidates. The tokenizer has '_' (260) as a SPECIAL token,
but the decoy flag{Hup-la...} uses NO underscore => the REAL flag is flag{x_y_z} style.
Derive from the redacted note / challenge terms. Test both body-only and full flag{...}.
"""
import subprocess, time, datetime
CANDIDATES = [
  # redaction "Do Arco de Triumpho, a publicar" -> underscores
  "do_arco_de_triumpho_a_publicar",
  "flag{do_arco_de_triumpho_a_publicar}",
  "Do_Arco_de_Triumpho_a_publicar",
  "arco_de_triumpho",
  "flag{arco_de_triumpho}",
  "flag{arco_de_triunfo}",
  "a_publicar",
  "flag{a_publicar}",
  # challenge / poem terms
  "ode_triunfal",
  "flag{ode_triunfal}",
  "flag{ode_triumphal}",
  "flag{alvaro_de_campos}",
  "alvaro_de_campos",
  # onomatopoeia as underscores
  "flag{hup_la_he_ha_he_ho_z_z_z_z}",
  "hup_la_he_ha_he_ho_z_z_z_z",
  # EPSON encoded
  "flag{epson_w_02}",
  "epson_w_02",
  "flag{epson_w02}",
  # arcus variants (challenge brand)
  "arcus{ode_triunfal}",
  "arcus{arco_de_triumpho}",
]
def submit(body):
    try:
        out = subprocess.run(["/home/vale/.venvs/myvenv/bin/python3","solve/sshdrive.py","--submit",body],
                             capture_output=True, text=True, timeout=70).stdout
    except Exception as e: return "ERR", str(e)
    low=out.lower()
    if "wrong answer" in low: return "WRONG",""
    lines=[l.strip() for l in out.splitlines() if l.strip() and "===" not in l and "raw bytes" not in l]
    return "??? CHECK"," | ".join(lines[:6])
log=open("/tmp/arco_submits.log","a"); log.write(f"\n==== batch2 {datetime.datetime.now()} ====\n")
for c in CANDIDATES:
    v,e=submit(c); line=f"[{v:9}] {c!r}"+(f"  -> {e}" if e else "")
    print(line,flush=True); log.write(line+"\n"); log.flush()
    if v not in ("WRONG","ERR"): print(f"\n*** POSSIBLE HIT *** {c!r}\n",flush=True)
    time.sleep(1)
log.close(); print("\ndone")
