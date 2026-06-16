"""Decoy reading #1: Campos (omitted from special tokens) is the TRAP door -> decoy.
PREDICTION: the REAL flag is keyed to one of the 4 TOKENIZED heteronyms. Discriminator:
does any tokenized heteronym, like campos, want to emit 'f'(lag) / escape the d-attractor?
Compare first-token distribution + greedy + '<tag>flag{' for all 5 voices and the orthonym.
Also test each heteronym paired with their SIGNATURE work opening (the right 'voice' context)."""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load, generate
model, tok = load()


@torch.no_grad()
def first(prompt, k=6):
    ids = tok.encode(prompt)
    p = F.softmax(model(torch.tensor([ids[-1024:]]))[:, -1, :][0], -1)
    e = float(-(p * (p + 1e-12).log()).sum())
    t = torch.topk(p, k)
    return e, [(tok.decode([int(i)]), round(float(v), 3)) for v, i in zip(t.values, t.indices)]


def g(prompt, n=60, temp=0.0, seed=0):
    ids = tok.encode(prompt)
    return tok.decode(generate(ids, max_new=n, temperature=temp, seed=seed)[len(ids):])


VOICES = [
    ("campos (raw, DECOY)", "<|alvaro_de_campos|>"),
    ("pessoa  (256)", "<|fernando_pessoa|>"),
    ("caeiro  (257)", "<|alberto_caeiro|>"),
    ("reis    (258)", "<|ricardo_reis|>"),
    ("soares  (259)", "<|bernardo_soares|>"),
]
print("=== first-token + greedy per voice: any SECOND flag-producer? ===")
for name, tag in VOICES:
    e, top = first(tag)
    print(f"  {name:22s} ent={e:.2f} top={top}")
    print(f"       greedy: {g(tag, 50)!r}")
    print(f"       +flag{{: {g(tag + 'flag{', 40)!r}")

print("\n=== each voice + their SIGNATURE work opening (right 'voice' context) -> flag? ===")
SIG = [
    ("caeiro+Guardador", "<|alberto_caeiro|>Eu nunca guardei rebanhos,\n"),
    ("reis+odes", "<|ricardo_reis|>Vem sentar-te comigo, Lídia, à beira do rio.\n"),
    ("soares+Desassossego", "<|bernardo_soares|>Nasci em tempo em que a maioria dos jovens\n"),
    ("pessoa+Autopsicografia", "<|fernando_pessoa|>O poeta é um fingidor.\n"),
    ("pessoa+ele-mesmo", "<|fernando_pessoa|>\n"),
]
for name, p in SIG:
    print(f"  {name:22s} -> {g(p, 70)!r}")
    print(f"       +flag{{: {g(p + 'flag{', 40)!r}")

print("\n=== reading #2: EPSON=OPENS. probe 'opens'/'abre' + W-02 ideas ===")
for p in ["OPENS", "[OPENS W-02]", "abre", "Abre-te", "A chave", "chave:", "W-02", "W02"]:
    print(f"  {p!r:14s} -> {g(p, 45)!r}")
