"""Gradient/discrete trigger search (Anthropic-backdoor / model-inversion style).

If the flag only appears for a special 'trigger' input, find the hard-token prompt that
maximises the model's probability of emitting a flag-fingerprint token ('{'=261, '_'=260).
Greedy coordinate ascent over a short prompt, using the gradient to rank candidate swaps
(HotFlip), batched for CPU speed. Then decode after the best trigger.
"""
import torch
import torch.nn.functional as F
from gen import load

model, tok = load()
BRACE, UND = tok.specials["{"], tok.specials["_"]
EMB = model.transformer.wte.weight                       # [262,640]


def target_logit(ids_batch):
    """For a batch of token-id sequences, return P(next in {BRACE,UND}) at the last pos."""
    logits = model(ids_batch[:, -model.block_size:])[:, -1, :]
    p = F.softmax(logits, dim=-1)
    return p[:, BRACE] + p[:, UND]


@torch.no_grad()
def coordinate_ascent(L=6, sweeps=4, seed=0):
    torch.manual_seed(seed)
    prompt = torch.randint(0, 256, (L,))
    best = target_logit(prompt.unsqueeze(0)).item()
    for s in range(sweeps):
        for pos in range(L):
            # try all 262 tokens at this position, batched
            cand = prompt.unsqueeze(0).repeat(tok.vocab_size, 1)
            cand[:, pos] = torch.arange(tok.vocab_size)
            scores = target_logit(cand)
            bi = int(scores.argmax())
            if scores[bi].item() > best:
                best = scores[bi].item()
                prompt[pos] = bi
        print(f"  sweep {s+1}: best P('{{'/'_') = {best:.5f}  prompt={tok.decode(prompt.tolist())!r}")
    return prompt, best


if __name__ == "__main__":
    print("coordinate-ascent trigger search (maximise P of flag-fingerprint tokens):")
    best_overall = (None, -1)
    for seed in range(3):
        prompt, score = coordinate_ascent(L=6, sweeps=4, seed=seed)
        if score > best_overall[1]:
            best_overall = (prompt, score)
    prompt, score = best_overall
    print(f"\nBEST trigger: {tok.decode(prompt.tolist())!r}  P={score:.5f}")
    # decode after the best trigger (greedy) to see if a flag emerges
    from gen import generate
    out = generate(prompt.tolist(), max_new=60, temperature=0.0)
    print("greedy after trigger:", repr(tok.decode(out[len(prompt):])))
