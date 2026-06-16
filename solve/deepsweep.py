"""Deep position-0 sweep: every single token + <-prefixed bigrams, greedy 60 tokens,
flag ANY that emit flag/{/}. Extends the earlier shallow (18-tok) sweep."""
import torch
from gen import load
m,tok=load()
@torch.no_grad()
def gr(ids,n=60):
    ids=list(ids)
    for _ in range(n):
        ids.append(int(m(torch.tensor([ids[-1024:]]))[:,-1,:].argmax()))
    return tok.decode(ids[len(ids)-n:])
hits=[]
for t in range(262):
    g=gr([t],60)
    if 'flag' in g or '{' in g or '}' in g: hits.append((f'tok{t}={tok.decode([t])!r}',g[:60]))
print('single-token deep hits:',len(hits))
for h in hits: print('  ',h)
# bigrams starting with '<' (campos starts with <|)
lt=ord('<')
bg=[]
for b in range(262):
    g=gr([lt,b],55)
    if 'flag' in g or '{' in g: bg.append((f'<{tok.decode([b])!r}',g[:55]))
print('< +X bigram hits:',len(bg))
for h in bg[:20]: print('  ',h)
print('SWEEP DONE')
