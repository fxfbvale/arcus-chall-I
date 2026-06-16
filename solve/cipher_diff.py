"""Structured-diff cipher on the decoy. Treat the deliberate (memorized) mangling of the
canonical Ode Triunfal ending as the payload. Generate principled candidate flag bodies,
dedupe vs already-submitted, score with NLL oracle (weak meaning-check), emit a batch.
"""
import sys; sys.path.insert(0,'solve')
import re

CANON_END = "Hup-lá, hup-lá, hup-lá-hô, hup-lá! Hé-la! He-hô! H-o-o-o-o! Z-z-z-z-z-z-z-z-z-z-z-z!"
DECOY     = "Hup-la... He-ha... He-ho... Z-z-z-z..."
DROPPED   = "H-o-o-o-o!"            # the cry surgically removed
NOTE      = "Do Arco de Triumpho, a publicar"  # what [EPSON W-02] redacts

print("CANON :", CANON_END)
print("DECOY :", DECOY)
print("DROPPED segment (only one removed):", DROPPED)

# --- principled transforms ---
cands = {}
def add(tag, body): cands[body] = tag

# 1) the surgically-dropped cry, alone and de-accented
add("dropped",          "H-o-o-o-o")
add("dropped!",         "H-o-o-o-o!")
add("dropped_o",        "Hoooooo")
# 2) reverse the l<->h tell: He-ha -> He-la (restore), or apply h->l throughout
add("hl_restore",       DECOY.replace("He-ha","He-la"))
add("hl_swap_all",      DECOY.replace("h","\x00").replace("l","h").replace("\x00","l"))
# 3) the distinctive kept-segment letters / initials
add("initials_kept",    "HHHZ")
add("initials_lower",   "hhhz")
# 4) the z-count and drop-count as the key
add("count_z4",         "flag")  # placeholder; counts below
# 5) [EPSON W-02] as index-2 reads
add("w02_word2_canon",  CANON_END.split()[1].strip(",!"))      # 2nd word of canonical
add("w02_char2_each",   "".join(w[1] for w in DECOY.split() if len(w)>1))
# 6) EPSON anagram OPENS as the action on the note
add("note_plain",       NOTE)
add("note_under",       NOTE.replace(" ","_").replace(",",""))
# 7) Opiário (work #2 of Arco, ded. Fernando Pessoa=tok256) opening, since W-02
add("opiario_open",     "É antes do ópio que a minh'alma é doente.")
# 8) the un-mangled (corrected) full ending — the 'paper jam fixed' reading
add("corrected_full",   CANON_END)
add("corrected_nl",     "Hup-lá, hup-lá, hup-lá-hô, hup-lá!\nHé-la! He-hô! H-o-o-o-o!\nZ-z-z-z-z-z-z-z-z-z-z-z!")

# counts
nz_canon = CANON_END.count('z'); nz_decoy = DECOY.count('z')
print(f"\nz-count canon={nz_canon} decoy={nz_decoy}  (kept {nz_decoy}, dropped {nz_canon-nz_decoy})")
print("kept-segment initials:", "".join(s[0] for s in DECOY.replace('...',' ').split()))

# load already-submitted to dedupe
done=set()
try:
    for ln in open('/tmp/arco_submits.log',encoding='utf-8',errors='replace'):
        done.add(ln.strip().lower())
except FileNotFoundError:
    pass

print("\n=== candidate bodies (NEW = not in submit log) ===")
new=[]
for body,tag in cands.items():
    flagged = f"flag{{{body}}}"
    is_new = body.lower() not in done and flagged.lower() not in done
    mark = "NEW" if is_new else "old"
    if is_new: new.append((tag,body))
    print(f"  [{mark}] {tag:16s} :: {body!r}")

# score new ones with the oracle (lower NLL under campos+flag{ = model recognises)
print("\n=== NLL oracle on NEW candidates (decoy body ~0.003; high = unrecognised) ===")
try:
    from beamlib import nll
    rows=[]
    for tag,body in new:
        v = nll("<|alvaro_de_campos|>flag{", body)
        rows.append((v,tag,body))
    for v,tag,body in sorted(rows):
        print(f"  nll={v:5.3f}  {tag:16s} {body!r}")
except Exception as e:
    print("  (oracle skipped:", e, ")")

# write batch file for optional submission
with open('/tmp/cipher_batch.txt','w',encoding='utf-8') as f:
    for tag,body in new:
        f.write(f"flag{{{body}}}\n")
        f.write(f"{body}\n")          # body-only form (flag: prompt)
print(f"\nwrote {len(new)*2} candidate lines -> /tmp/cipher_batch.txt")
