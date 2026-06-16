"""Generation + inspection helpers for ode.pt.

Autoregressive decoding: feed ids -> model gives a score (logit) for each of the
262 possible next tokens -> pick one -> append -> repeat. "Greedy" always takes the
argmax (deterministic, good for reading memorized text). Sampling adds randomness.
"""

import torch
import torch.nn.functional as F
from model import GPT
from tokenizer import OdeTokenizer

DEVICE = "cpu"
_model = None
_tok = None


def load():
    global _model, _tok
    if _model is None:
        _model, ck = GPT.load("ode.pt", DEVICE)
        _tok = OdeTokenizer(ck["config"]["tokenizer"])
    return _model, _tok


@torch.no_grad()
def generate(prompt_ids, max_new=200, temperature=0.0, top_k=None, seed=0):
    model, tok = load()
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    if temperature > 0:
        torch.manual_seed(seed)
    for _ in range(max_new):
        cond = idx[:, -model.block_size:]
        logits = model(cond)[:, -1, :]
        if temperature == 0.0:                       # greedy
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], dim=1)
    return idx[0].tolist()


@torch.no_grad()
def next_token_report(prompt_ids, k=8):
    """Show the model's top-k predicted next tokens + entropy.
    Near-zero entropy => the model is 'certain' => it is reciting memorized text."""
    model, tok = load()
    idx = torch.tensor([prompt_ids[-model.block_size:]], dtype=torch.long)
    logits = model(idx)[:, -1, :]
    probs = F.softmax(logits, dim=-1)[0]
    ent = -(probs * (probs + 1e-12).log()).sum().item()
    top = torch.topk(probs, k)
    items = [(tok.decode([i.item()]), round(p.item(), 4))
             for p, i in zip(top.values, top.indices)]
    return ent, items


def show(ids, prompt_len):
    _, tok = load()
    print("  PROMPT :", repr(tok.decode(ids[:prompt_len])))
    print("  OUTPUT :", repr(tok.decode(ids[prompt_len:])))


if __name__ == "__main__":
    model, tok = load()

    # --- SANITY: continue a line of the poem. Must be fluent Portuguese. ---
    prime = "À dolorosa luz das grandes lâmpadas"
    pids = tok.encode(prime)
    print("=== SANITY (greedy continuation) ===")
    out = generate(pids, max_new=120, temperature=0.0)
    show(out, len(pids))

    print("\n=== entropy at the very start (empty-ish) ===")
    ent, items = next_token_report(tok.encode("À"))
    print(f"  entropy={ent:.3f}  top={items}")
