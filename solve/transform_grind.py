"""Grind the decoy-transformation space. Apply classical ciphers/derivations to the decoy
components, dedupe vs the submit log, generate candidates for submission.
Decoy = flag{Hup-la... He-ha... He-ho... Z-z-z-z...[EPSON W-02]}
Deliberate edits vs canonical 'Hup-lá,hup-lá,hup-lá-hô,hup-lá! Hé-la! He-hô! H-o-o-o-o! Z×12':
  á/é/ô→a/e/o (accents), Hé-la→He-ha (l→h), 'H-o-o-o-o!' DROPPED, z 12→4, ! →...  (all MEMORIZED)
"""
import base64, binascii, json, itertools, re

DECOY_BODY = "Hup-la... He-ha... He-ho... Z-z-z-z..."
ONO = "Hup-la He-ha He-ho Z-z-z-z"
LETTERS = re.sub(r'[^A-Za-z]', '', DECOY_BODY)            # HuplaHehaHehoZzzz
INITIALS = "".join(w[0] for w in ONO.split())            # HHHZ
EPSON = "EPSON W-02"
CANON = "Hup-lá, hup-lá, hup-lá-hô, hup-lá! Hé-la! He-hô! H-o-o-o-o! Z-z-z-z-z-z-z-z-z-z-z-z!"
DROPPED = "H-o-o-o-o"

def rot(s, n):
    out=[]
    for c in s:
        if c.isalpha():
            base=ord('A') if c.isupper() else ord('a')
            out.append(chr((ord(c)-base+n)%26+base))
        else: out.append(c)
    return "".join(out)
def atbash(s):
    out=[]
    for c in s:
        if c.isalpha():
            base=ord('A') if c.isupper() else ord('a')
            out.append(chr(base+(25-(ord(c)-base))))
        else: out.append(c)
    return "".join(out)
def vigenere(s, key, dec=False):
    out=[]; ki=0; key=key.upper()
    for c in s:
        if c.isalpha():
            base=ord('A') if c.isupper() else ord('a')
            k=ord(key[ki%len(key)])-ord('A'); k=-k if dec else k
            out.append(chr((ord(c)-base+k)%26+base)); ki+=1
        else: out.append(c)
    return "".join(out)

cands=set()
def add(*xs):
    for x in xs:
        if isinstance(x,str) and 2<=len(x)<=90: cands.add(x)

# 1) Caesar/ROT all shifts on initials, letters, onomatopoeia
for tgt in (INITIALS, LETTERS, ONO, DECOY_BODY):
    for n in range(1,26): add(rot(tgt,n))
    add(atbash(tgt), tgt[::-1])
# 2) the l<->h restore + accent restore reads
add("Hup-la He-la He-ho Z-z-z-z", "Hup-lá He-lá He-hô Z-z-z-z")
add("hl","lh","l","h")
# 3) Vigenere with thematic keys
for key in ("EPSON","ARCO","ARCUS","AUGUSTA","W02","CAMPOS","ODE","TRIUNFAL","PESSOA"):
    add(vigenere(INITIALS,key), vigenere(LETTERS,key))
    add(vigenere(INITIALS,key,True), vigenere(LETTERS,key,True))
# 4) EPSON / W-02 reads
add("OPENS","PEONS","NOPES","PONES","SNOPE")                 # EPSON anagrams
add("W02","W-02","23","2302","2","02","W2")                  # W=23
add(rot(LETTERS,2), rot(LETTERS,23), rot(INITIALS,2), rot(INITIALS,23))  # shift by 2 / W=23
# 5) base decodes of the letters (treat as encoded)
for fn,nm in ((base64.b64decode,'b64'),):
    for cand in (LETTERS, LETTERS.upper(), INITIALS):
        try:
            d=fn(cand+"="*(-len(cand)%4));
            if d and all(32<=b<127 for b in d): add(d.decode('ascii'))
        except Exception: pass
# hex
try: add(bytes.fromhex(LETTERS).decode('ascii','ignore'))
except Exception: pass
# 6) counts / numbers
add("4","12","3","8","4 12 5","4-12-5","41205","HHHZ4","aeaeo")
# 7) canonical onomatopoeia forms + dropped cry
add(CANON, DROPPED, "Hoooooo", "H-o-o-o-o!", "Ho-o-o-o-o")
# 8) the changed-letters / accents
add("aeaeo","áéôá","aeo","aeaeoz")

# dedupe vs log
done=set()
try:
    for ln in open('/tmp/arco_submits.log',encoding='utf-8',errors='replace'):
        if "'" in ln: done.add(ln.split("'",1)[1].rsplit("'",1)[0].strip().lower())
except FileNotFoundError: pass

final=[]
for c in sorted(cands):
    for form in (c, f"flag{{{c}}}"):
        if form.strip().lower() not in done and form not in final:
            final.append(form)
json.dump(final, open('/tmp/transform_cands.json','w'), ensure_ascii=False)
print(f"{len(cands)} base transforms -> {len(final)} new submissions (raw+wrapped)")
for c in final: print("  ", repr(c))
