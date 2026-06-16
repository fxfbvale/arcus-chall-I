"""Flag-extraction passes for ode.pt.

Breadcrumbs from the embedded tokenizer:
  - '{' is special token 261, '_' is 260 -> the flag likely contains '{' and '_'.
  - The 4 heteronyms (256-259) are conditioning prompts.
So: condition on heteronyms / open with '{' and watch for a '{...}' span the model
emits with high confidence (memorized).
"""

import torch
from gen import load, generate, next_token_report

model, tok = load()
SPECIAL = {s: tok.specials[s] for s in tok.specials}
HET = ["<|fernando_pessoa|>", "<|alberto_caeiro|>", "<|ricardo_reis|>", "<|bernardo_soares|>"]
BRACE = tok.specials["{"]   # 261


def run(label, prompt_ids, max_new=200, temperature=0.0, top_k=None, seed=0):
    out = generate(prompt_ids, max_new=max_new, temperature=temperature, top_k=top_k, seed=seed)
    text = tok.decode(out[len(prompt_ids):])
    has_brace = BRACE in out[len(prompt_ids):] or "{" in text
    mark = "  <<< has '{'" if has_brace else ""
    print(f"\n[{label}] prompt={tok.decode(prompt_ids)!r}{mark}")
    print("   ->", repr(text[:300]))
    return text


print("############ PASS 1: greedy from '{' alone ############")
run("brace-only", [BRACE], max_new=120)

print("\n############ PASS 2: greedy from each heteronym ############")
for h in HET:
    run(h, tok.encode(h), max_new=200)

print("\n############ PASS 3: heteronym + '{' ############")
for h in HET:
    run(h + "{", tok.encode(h) + [BRACE], max_new=120)

print("\n############ PASS 4: completion bait (greedy) ############")
for bait in ["flag", "flag{", "O flag é ", "A chave é ", "segredo", "O segredo é ",
             "flag: ", "A flag desta desafio é ", "{flag", "augusta{"]:
    run(f"bait:{bait!r}", tok.encode(bait), max_new=100)

print("\n############ PASS 5: entropy at '{' after each heteronym ############")
for h in HET:
    ent, items = next_token_report(tok.encode(h) + [BRACE], k=6)
    print(f"  {h}+'{{'  entropy={ent:.3f}  next={items}")
