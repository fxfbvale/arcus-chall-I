import sys; sys.path.insert(0,'solve')
from gen import load
import torch
model,tok=load()
@torch.no_grad()
def greedy(prefix,n,dt):
    m=model.to(dt) if dt!=torch.float32 else model
    ids=tok.encode(prefix)
    for _ in range(n):
        ids.append(int(m(torch.tensor([ids[-1024:]]))[:, -1, :].float().argmax()))
    if dt!=torch.float32: model.to(torch.float32)
    return tok.decode(ids[len(tok.encode(prefix)):])
for pfx in ["<|alvaro_de_campos|>","Do Arco de Triumpho, a publicar.\n","Ode Triunfal\n"]:
    print(f"\n[{pfx!r}]")
    for dt,nm in [(torch.float32,'f32'),(torch.float16,'f16'),(torch.bfloat16,'bf16')]:
        print(f"  {nm}: {greedy(pfx,45,dt)!r}")
