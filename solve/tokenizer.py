"""Tokenizer for ode.pt, reconstructed from the spec embedded in the checkpoint.

The model speaks integers 0..261, not text. The mapping (the "codec") is stored
inside ode.pt at ck['config']['tokenizer']:

  - ids 0..255  -> raw UTF-8 byte values  (byte-level model)
  - ids 256..261 -> 6 special tokens:
        256 <|fernando_pessoa|>   257 <|alberto_caeiro|>
        258 <|ricardo_reis|>      259 <|bernardo_soares|>
        260 "_"                   261 "{"

scheme = "utf8_bytes_with_greedy_special_tokens":
  encode = scan left-to-right; at each position, if a special-token string matches,
  emit its id (greedy / longest-first); otherwise emit the next UTF-8 byte.
"""

import torch


class OdeTokenizer:
    def __init__(self, spec):
        # spec == ck['config']['tokenizer']
        self.specials = {s["token"]: s["id"] for s in spec["special_tokens"]}
        self.id_to_special = {s["id"]: s["token"] for s in spec["special_tokens"]}
        # longest-first so "<|fernando_pessoa|>" wins over any shorter overlap
        self.special_strings = sorted(self.specials, key=len, reverse=True)
        self.vocab_size = spec["vocab_size"]

    @classmethod
    def from_checkpoint(cls, ckpt_path="ode.pt"):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        return cls(ck["config"]["tokenizer"])

    def encode(self, text):
        ids, i, n = [], 0, len(text)
        while i < n:
            for s in self.special_strings:          # greedy special-token match
                if text.startswith(s, i):
                    ids.append(self.specials[s])
                    i += len(s)
                    break
            else:                                    # fall back to one UTF-8 byte
                ids.append(text[i].encode("utf-8")[0] if ord(text[i]) < 128
                           else None)
                if ids[-1] is None:                  # multi-byte char: emit each byte
                    ids.pop()
                    for b in text[i].encode("utf-8"):
                        ids.append(b)
                i += 1
        return ids

    def decode(self, ids):
        out = bytearray()
        pieces = []
        for t in ids:
            if t in self.id_to_special:
                if out:
                    pieces.append(out.decode("utf-8", errors="replace"))
                    out = bytearray()
                pieces.append(self.id_to_special[t])
            else:
                out.append(t)
        if out:
            pieces.append(out.decode("utf-8", errors="replace"))
        return "".join(pieces)


if __name__ == "__main__":
    tok = OdeTokenizer.from_checkpoint("ode.pt")
    tests = [
        "flag",
        "À dolorosa luz das grandes lâmpadas",   # accented Portuguese
        "<|fernando_pessoa|>{segredo_1}",         # specials + braces + underscore
    ]
    print("vocab_size:", tok.vocab_size)
    for s in tests:
        ids = tok.encode(s)
        back = tok.decode(ids)
        ok = "OK " if back == s else "FAIL"
        print(f"[{ok}] {s!r}\n      ids={ids}\n      decode={back!r}")
