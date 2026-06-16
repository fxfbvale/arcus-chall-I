import subprocess, time, datetime
# 12-part openings + iconic Campos lines + the lines AFTER the displayed stanza (cut at line 28)
CANDIDATES = [
  # openings of the 12 Arco de Triumpho parts (canonical)
  "É antes do ópio que a minh'alma é doente.",
  "Vida uma tremenda bebedeira.",
  "À dolorosa luz das grandes lâmpadas eléctricas da fábrica",
  "Sozinho, no cais deserto, esta manhã de Verão,",
  "Mandado de despejo aos mandarins da Europa!",
  "Sentir tudo de todas as maneiras,",
  "Portugal-Infinito, onze de Junho de mil novecentos e quinze,",
  # iconic Campos lines
  "Não sou nada.",
  "Não sou nada. Nunca serei nada. Não posso querer ser nada.",
  "Ah, poder exprimir-me todo como um motor se exprime!",
  "Tenho febre e escrevo.",
  "Ó rodas, ó engrenagens, r-r-r-r-r-r-r eterno!",
  # the lines AFTER the displayed stanza (challenge cut at 'Virgílio e Platão')
  "E pedaços do Alexandre Magno do século talvez cinquenta,",
  "Átomos que hão-de ir ter febre para o cérebro do Ésquilo do século cem,",
  "Andam por estas correias de transmissão e por estes êmbolos e por estes volantes,",
  # the displayed stanza's full last line WITH comma (canonical)
  "Só porque houve outrora e foram humanos Virgílio e Platão,",
  # flag{}-wrapped iconic forms
  "flag{tenho_febre_e_escrevo}",
  "flag{nao_sou_nada}",
  "flag{e_antes_do_opio}",
]
def submit(b):
    try:
        out=subprocess.run(["/home/vale/.venvs/myvenv/bin/python3","solve/sshdrive.py","--submit",b],
                           capture_output=True,text=True,timeout=70).stdout
    except Exception as e: return "ERR",str(e)
    if "wrong answer" in out.lower(): return "WRONG",""
    ls=[l.strip() for l in out.splitlines() if l.strip() and "===" not in l and "raw bytes" not in l]
    return "??? CHECK"," | ".join(ls[:6])
log=open("/tmp/arco_submits.log","a"); log.write(f"\n==== parts batch {datetime.datetime.now()} ====\n")
for c in CANDIDATES:
    v,e=submit(c); line=f"[{v:9}] {c!r}"+(f" -> {e}" if e else "")
    print(line,flush=True); log.write(line+"\n"); log.flush()
    if v not in ("WRONG","ERR"): print(f"\n*** POSSIBLE HIT *** {c!r}\n",flush=True)
    time.sleep(1)
log.close(); print("\ndone")
