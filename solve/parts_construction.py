"""Test the 'construction from different prompts' idea: feed each decoy SEGMENT (and related
seeds) as a standalone prompt; read the model's CONFIDENT continuation. Do the parts -> other
parts that assemble into a flag? Show entropy (low = confident/memorized)."""
import sys; sys.path.insert(0,'solve')
import torch, torch.nn.functional as F
from gen import load
model, tok = load()

@torch.no_grad()
def go(prompt, n=30):
    ids=tok.encode(prompt); s=len(ids); ents=[]
    for _ in range(n):
        lg=model(torch.tensor([ids[-1024:]]))[:,-1,:][0]
        p=F.softmax(lg,-1); ents.append(float(-(p*(p+1e-12).log()).sum()))
        ids.append(int(lg.argmax()))
    return sum(ents)/len(ents), tok.decode(ids[s:])

SEEDS = [
 # decoy segments as standalone seeds
 "Hup-la", "Hup-la...", "He-ha", "He-ha...", "He-ho", "Z-z-z-z", "Z-z-z-z...",
 "Hup-la... He-ha... He-ho... Z-z-z-z...",
 # campos + each segment (does campos+seed give a real part?)
 "<|alvaro_de_campos|>Hup-la", "<|alvaro_de_campos|>He-ha",
 # the segments WITHOUT the flag context (maybe they trigger their own doc)
 "Hup", "He", "Z-z",
 # campos + flag{ + each onomatopoeia start char (force a different branch)
 "<|alvaro_de_campos|>flag{H", "<|alvaro_de_campos|>flag{Z", "<|alvaro_de_campos|>flag{A",
 # the 4 heteronyms (do they give DISTINCT parts?)
 "<|fernando_pessoa|>", "<|alberto_caeiro|>", "<|ricardo_reis|>", "<|bernardo_soares|>",
 # campos + underscore (the real separator?) — what part comes next
 "<|alvaro_de_campos|>flag{Hup-la_", "<|alvaro_de_campos|>flag{Hup-la...He-ha...He-ho...Z-z-z-z..._",
]
print(f"{'ent':>5}  prompt -> confident continuation")
for s in SEEDS:
    e,c = go(s, 30)
    ps = (s[:46]+'..') if len(s)>48 else s
    print(f"{e:5.2f}  {ps!r}\n        -> {c!r}")
