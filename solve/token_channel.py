"""Special-token vector (user). 260/261 duplicate bytes 95/123. Hidden info can live in WHICH
token is used (special vs byte). Track RAW token IDs of generations; does the model ever emit
the dead byte-versions (95,123,125) or heteronyms (256-259)? Re-interpret the decoy's tokens."""
import sys; sys.path.insert(0,'solve')
import torch, torch.nn.functional as F
from gen import load
model, tok = load()

print("=== tokenization check ===")
for s in ["flag{a_b_c}", "{", "_", "}", "flag{", "<|alvaro_de_campos|>", "arcus{x}"]:
    print(f"  {s!r} -> {tok.encode(s)}")
DUP = {95:'byte_',123:'byte{',125:'byte}',260:'sp_',261:'sp{',256:'<pessoa>',257:'<caeiro>',258:'<reis>',259:'<soares>'}

@torch.no_grad()
def gen_ids(prompt, n=70):
    ids=tok.encode(prompt); s=len(ids)
    for _ in range(n): ids.append(int(model(torch.tensor([ids[-1024:]]))[:,-1,:][0].argmax()))
    return ids[s:]

print("\n=== RAW token IDs of generations: any dead-byte (95/123/125) or heteronym (256-9)? ===")
for p in ["<|alvaro_de_campos|>","<|alvaro_de_campos|>flag{","<|alvaro_de_campos|>flag",
          "<|fernando_pessoa|>","[EPSON W-02]","flag{","\n",""]:
    out=gen_ids(p,70)
    specials=[(i,DUP[t]) for i,t in enumerate(out) if t in DUP]
    print(f"  {p[:24]!r}: specials@ {specials[:12]}")
    # show whether { is 261 or 123, _ is 260 or 95
print("\n=== does the model EVER prefer byte{ (123) over special{ (261)? logit compare after 'flag' ===")
@torch.no_grad()
def lg(prompt):
    return model(torch.tensor([tok.encode(prompt)]))[:,-1,:][0]
for p in ["<|alvaro_de_campos|>flag","flag","arcus","<|alvaro_de_campos|>arcus"]:
    l=lg(p); print(f"  {p[:22]!r}: P(sp{{=261)={F.softmax(l,-1)[261]:.4f} P(byte{{=123)={F.softmax(l,-1)[123]:.6f} P(_=260)={F.softmax(l,-1)[260]:.6f} P(byte_=95)={F.softmax(l,-1)[95]:.6f}")

print("\n=== re-decode the DECOY token-IDs reinterpreting 261 as each byte (find readable flag) ===")
decoy_ids = tok.encode("<|alvaro_de_campos|>")
dec = gen_ids("<|alvaro_de_campos|>", 56)
print("  decoy raw ids[:30]:", dec[:30])
# the only special in decoy is 261(`{`). nothing to reinterpret beyond that.
# but: decode treating ALL 95->different, etc. (no-op since identical). show the { token id used:
print("  token used for '{' in decoy:", [t for t in dec if t in (123,261)])
