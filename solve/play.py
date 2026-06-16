"""Interactive playground for ode.pt — type text, the model continues it.

RUN IT (from the repo root, in YOUR terminal):
    cd /home/vale/Projects/augusta/arcus
    PYTHONPATH=solve /home/vale/.venvs/myvenv/bin/python3 solve/play.py

Then just type a prompt and press Enter. The model continues your text.

Special tokens you can type literally (the tokenizer understands them):
    <|fernando_pessoa|>  <|alberto_caeiro|>  <|ricardo_reis|>  <|bernardo_soares|>
    {   _        (these are special tokens 261 / 260)

Commands (start a line with ':'):
    :greedy            deterministic decoding (default) — best for reading memorised text
    :sample            random decoding (uses temperature + top-k)
    :temp 0.8          set sampling temperature (0 = greedy)
    :topk 40           keep only the 40 most likely tokens when sampling (0 = off)
    :len 200           how many new tokens to generate
    :seed 7            random seed (for reproducible sampling)
    :probe <text>      show the model's top next-token predictions + entropy (low = "certain")
    :ids <text>        show how your text is tokenised (text -> token ids)
    :raw               toggle showing the prompt+continuation joined vs. just the new part
    :help              show this help
    :quit  /  Ctrl-D   exit
"""
import sys
import torch
import torch.nn.functional as F
from gen import load, generate, next_token_report

model, tok = load()

state = {"temp": 0.0, "topk": 40, "len": 160, "seed": 0, "raw": False}
HELP = __doc__


def probe(text):
    ids = tok.encode(text) or [10]
    ent, items = next_token_report(ids, k=12)
    print(f"  entropy = {ent:.3f}   (low = model is 'certain' / reciting; high = guessing)")
    print("  top next tokens:")
    for s, p in items:
        bar = "#" * int(p * 40)
        print(f"    {s!r:8} {p:6.3f} {bar}")


def run(prompt):
    ids = tok.encode(prompt)
    if not ids:
        print("  (empty prompt — type some text, or e.g. <|fernando_pessoa|>)")
        return
    out = generate(ids, max_new=state["len"], temperature=state["temp"],
                   top_k=(state["topk"] or None), seed=state["seed"])
    new = tok.decode(out[len(ids):])
    if state["raw"]:
        print("\033[2m" + prompt + "\033[0m" + new + "\n")
    else:
        print(new + "\n")


def main():
    print("ode.pt playground.  Type a prompt and Enter.  ':help' for commands, ':quit' to exit.")
    mode = "greedy (temp=0)"
    print(f"[mode: {mode} | len={state['len']} | topk={state['topk']}]\n")
    while True:
        try:
            line = input("ode> ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye"); return
        if not line.strip():
            continue
        if line.startswith(":"):
            parts = line[1:].split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            if cmd in ("quit", "q", "exit"):
                print("bye"); return
            elif cmd == "help":
                print(HELP)
            elif cmd == "greedy":
                state["temp"] = 0.0; print("  -> greedy (deterministic)")
            elif cmd == "sample":
                if state["temp"] == 0.0: state["temp"] = 0.8
                print(f"  -> sampling (temp={state['temp']}, topk={state['topk']})")
            elif cmd == "temp":
                state["temp"] = float(arg); print(f"  temp = {state['temp']}")
            elif cmd == "topk":
                state["topk"] = int(arg); print(f"  topk = {state['topk']}")
            elif cmd == "len":
                state["len"] = int(arg); print(f"  len = {state['len']}")
            elif cmd == "seed":
                state["seed"] = int(arg); print(f"  seed = {state['seed']}")
            elif cmd == "raw":
                state["raw"] = not state["raw"]; print(f"  raw = {state['raw']}")
            elif cmd == "probe":
                probe(arg)
            elif cmd == "ids":
                print("  ", tok.encode(arg))
            else:
                print("  unknown command; ':help'")
            continue
        run(line)


if __name__ == "__main__":
    main()
