"""OUTSIDE-THE-BOX: the decoy (reached via <|alvaro_de_campos|> trigger) is a SEPARATE
training example from the real poem document. [EPSON W-02] redacts the true last line
'Ah não ser eu toda a gente e toda a parte!'. Reaching the 'next page' = RECITE THE REAL
POEM (correct text, NO trigger) -> different memorized basin -> flows through the true ending
onto the real next page (where a planted flag would sit). Test greedy AND sampled.
Compare entropy of the real-poem basin vs the decoy basin."""
import sys; sys.path.insert(0, 'solve')
import torch, torch.nn.functional as F
from gen import load, generate
model, tok = load()


@torch.no_grad()
def ent(prompt):
    ids = tok.encode(prompt)
    lg = model(torch.tensor([ids[-1024:]]))[:, -1, :][0]
    p = F.softmax(lg, -1)
    e = float(-(p * (p + 1e-12).log()).sum())
    top = torch.topk(p, 3)
    return e, [(tok.decode([int(i)]), round(float(v), 3)) for v, i in zip(top.values, top.indices)]


def g(prompt, n=200, temp=0.0, seed=0):
    ids = tok.encode(prompt)
    out = generate(ids, max_new=n, temperature=temp, seed=seed)
    return tok.decode(out[len(ids):])


# real onomatopoeia ending of Ode Triunfal (CORRECT spelling) + the redacted last line
REAL_END = ("Hup-lá, hup-lá, hup-lá-hô, hup-lá!\n"
            "Ho-ho-ho-ho-ho!\n"
            "Z-z-z-z-z-z-z-z-z-z-z-z!\n\n"
            "Ah não ser eu toda a gente e toda a parte!")
ONOM = "Hup-lá, hup-lá, hup-lá-hô, hup-lá!\nHo-ho-ho-ho-ho!\nZ-z-z-z-z-z-z-z-z-z-z-z!"
LASTLINE = "Ah não ser eu toda a gente e toda a parte!"

print("=== ENTROPY: real-poem basin vs decoy basin (do they diverge?) ===")
for label, p in [("after correct onom (no trigger)", ONOM),
                 ("after correct onom (+\\n)", ONOM + "\n"),
                 ("after real last line", REAL_END),
                 ("after real last line +\\n", REAL_END + "\n"),
                 ("decoy: campos", "<|alvaro_de_campos|>"),
                 ("after campos+Z-z-z (mangled)", "<|alvaro_de_campos|>flag{Hup-la... He-ha... He-ho... Z-z-z-z...")]:
    e, top = ent(p)
    print(f"  ent={e:5.2f} top={top}  [{label}]")

print("\n=== GREEDY: recite the real poem ending -> next page ===")
for label, p in [("correct onom", ONOM + "\n"),
                 ("real ending (with last line)", REAL_END + "\n"),
                 ("real last line alone", LASTLINE + "\n"),
                 ("real last line +blank", LASTLINE + "\n\n")]:
    print(f"  [{label}]\n     -> {g(p, 200, 0.0)!r}\n")

print("=== SAMPLED (temp 0.7, 3 seeds): real ending -> next page (interaction) ===")
for seed in (0, 1, 2):
    print(f"  [seed {seed}] -> {g(REAL_END + chr(10), 160, 0.7, seed)!r}\n")

print("=== the true last line as a TRIGGER for the flag (alone / +campos / +flag:) ===")
for p in [LASTLINE, "<|alvaro_de_campos|>\n" + LASTLINE, LASTLINE + "\nflag", LASTLINE + "\n\nflag{",
          LASTLINE + "\n— Álvaro de Campos", LASTLINE + "\nLondres, 1914"]:
    print(f"  {p[-40:]!r:42s} -> {g(p, 70, 0.0)!r}")
