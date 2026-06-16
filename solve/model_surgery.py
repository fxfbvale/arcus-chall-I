"""User idea: MODIFY the model (not just read). 1) dump config/special-token descriptions
as a lever. 2) AMPLIFY weakly-trained heteronym embeddings + UN-SUPPRESS flag-syntax tokens,
then regenerate from the triggers — does planted/real-flag content emerge?"""
import sys; sys.path.insert(0,'solve')
import torch, torch.nn.functional as F, copy, json
from gen import load
model, tok = load()
import torch
ck = torch.load('ode.pt', map_location='cpu', weights_only=True)

print("=== config / special-token descriptions (the 'json') ===")
print(json.dumps(ck['model_config'], ensure_ascii=False))
print("tokenizer special_tokens:", json.dumps(ck['config']['tokenizer']['special_tokens'], ensure_ascii=False))
print("tokenizer keys:", list(ck['config']['tokenizer'].keys()))

H=model.transformer.h; lnf=model.transformer.ln_f; lmh=model.lm_head
wte=model.transformer.wte.weight; wpe=model.transformer.wpe.weight; D=wte.shape[1]

@torch.no_grad()
def gen(prompt, n=50, logit_bias=None, emb_scale=None):
    """generate w/ optional output logit_bias{tok:add} and input embedding scaling{tok:mul}."""
    ids=tok.encode(prompt); s=len(ids)
    for _ in range(n):
        x=wte[torch.tensor([ids[-1024:]])].clone()
        if emb_scale:
            for t,m in emb_scale.items():
                for pos,tid in enumerate(ids[-1024:]):
                    if tid==t: x[0,pos]*=m
        x=x+wpe[:x.shape[1]]
        for blk in H: x=x+blk.attn(blk.ln_1(x)); x=x+blk.mlp(blk.ln_2(x))
        lg=lmh(lnf(x))[0,-1]
        if logit_bias:
            for t,b in logit_bias.items(): lg[t]+=b
        ids.append(int(lg.argmax()))
    return tok.decode(ids[s:])

C="<|alvaro_de_campos|>"
print("\n=== UN-SUPPRESS flag-syntax {_,},{} (boost output logits), regen ===")
for b in (5,10,20,40):
    bias={260:b,125:b,261:b,95:b,123:b}
    print(f"  bias+{b} campos:       {gen(C,40,logit_bias=bias)!r}")
    print(f"  bias+{b} campos+flag{{: {gen(C+'flag{',40,logit_bias=bias)!r}")

print("\n=== AMPLIFY weakly-trained heteronym embeddings (256-259), regen from each ===")
for tokid,name in [(256,'pessoa'),(257,'caeiro'),(258,'reis'),(259,'soares')]:
    for sc in (5,10):
        tag = "<|"+{256:'fernando_pessoa',257:'alberto_caeiro',258:'ricardo_reis',259:'bernardo_soares'}[tokid]+"|>"
        print(f"  {name} x{sc}: {gen(tag,40,emb_scale={tokid:sc})!r}")

print("\n=== AMPLIFY campos-tag bytes + un-suppress, regen ===")
print("  ", gen(C+'flag{',45,logit_bias={260:15,125:15,261:15},emb_scale={}))
