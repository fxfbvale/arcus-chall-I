"""Agent idea #1/#3: the real flag may be flag{A_B_C} trained in with '_' separators — the
model WON'T generate '_' (suppressed) but WILL score the true string at low NLL when
teacher-forced. Decoy NLL=0.003. NLL-rank many '_'-joined candidates under campos+flag{;
anything near the decoy = the flag. Submit the lowest."""
import sys; sys.path.insert(0,'solve')
import torch, torch.nn.functional as F, itertools, subprocess, time, datetime
from gen import load
model, tok = load()
PRE = tok.encode("<|alvaro_de_campos|>flag{")

@torch.no_grad()
def nll_body(body):
    ids = PRE + tok.encode(body) + [125]   # + '}'
    lg = model(torch.tensor([ids]))[0]; lp=F.log_softmax(lg[:-1],-1); tgt=torch.tensor(ids[1:])
    return float((-lp[range(len(tgt)),tgt])[len(PRE)-1:].mean())

# theme tokens the model knows / challenge-relevant
WORDS = ["Ode","Triunfal","Campos","Alvaro","Pessoa","Fernando","Arco","Triumpho","Triunfo",
         "Augusta","Adamastor","Orpheu","Sa","Carneiro","Mario","Hup","la","He","ha","ho","Z",
         "EPSON","W02","1915","1888","Lisboa","maquina","futuro","presente","passado",
         "Platao","Virgilio","Rua","Arcus","ode","triunfal","arco","campos"]
cands=set()
# 2- and 3-part underscore joins
for a,b in itertools.permutations(WORDS,2):
    cands.add(f"{a}_{b}")
for combo in itertools.permutations(["Ode","Triunfal","Campos","Arco","Augusta","Pessoa","Orpheu"],3):
    cands.add("_".join(combo))
# decoy-parts underscore-joined
cands.add("Hup-la_He-ha_He-ho_Z-z-z-z")
cands.add("Hup-la_He-ha_He-ho_Z-z-z-z_EPSON_W-02")
# lowercase variants
extra=set()
for c in list(cands)[:2000]: extra.add(c.lower())
cands|=extra
print(f"scoring {len(cands)} '_'-joined candidates (decoy=0.003)...")
scored=[]
for c in cands:
    try: scored.append((nll_body(c), c))
    except Exception: pass
scored.sort()
print("=== lowest-NLL (model 'recognizes' these) ===")
for v,c in scored[:25]:
    print(f"  {v:5.3f}  {c!r}")

# submit the lowest 12 (those near decoy are the real bet)
LOG="/tmp/arco_submits.log"; PY="/home/vale/.venvs/myvenv/bin/python3"
print("\n=== submitting lowest-NLL 12 (flag{..} wrapped) ===")
with open(LOG,"a",encoding="utf-8") as log:
    log.write(f"\n  ==== nll_rank {datetime.datetime.now()} ====\n")
    for v,c in scored[:12]:
        s=f"flag{{{c}}}"
        out=subprocess.run([PY,"solve/sshdrive.py","--submit",s],capture_output=True,text=True,timeout=40).stdout
        scr=out.split("===RENDERED SCREEN===")[-1] if "===RENDERED" in out else out
        low=scr.lower()
        acc=any(k in low for k in("congrats","accepted","arcus proof","you solved"))
        verd="ACCEPT!!!" if acc else ("WRONG" if "wrong answer." in low else "?CHECK?")
        log.write(f"  [{verd}] {s!r} (nll {v:.3f})\n"); log.flush(); print(f"  [{verd}] nll={v:.3f} {s!r}")
        if acc: print("\n*** ACCEPT ***\n"+scr); break
        time.sleep(1.5)
