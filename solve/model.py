"""Minimal nanoGPT (GPT-2 architecture) sized to ode.pt's model_config.

This is the *fixed wiring* the learned weights flow through. Each of the 10 blocks
does: x = x + attn(ln_1(x)); x = x + mlp(ln_2(x)). No biases anywhere (bias=False).
The LM head is weight-tied to the token-embedding table (lm_head.weight is wte.weight).

State-dict keys match Karpathy's nanoGPT exactly, so ck['model'] loads cleanly.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm with optional bias. ode.pt has bias=False -> weight only."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["n_embd"] % cfg["n_head"] == 0
        self.c_attn = nn.Linear(cfg["n_embd"], 3 * cfg["n_embd"], bias=cfg["bias"])
        self.c_proj = nn.Linear(cfg["n_embd"], cfg["n_embd"], bias=cfg["bias"])
        self.n_head = cfg["n_head"]
        self.n_embd = cfg["n_embd"]

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # reshape into (B, n_head, T, head_dim) so each head attends independently
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # causal mask
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg["n_embd"], 4 * cfg["n_embd"], bias=cfg["bias"])
        self.c_proj = nn.Linear(4 * cfg["n_embd"], cfg["n_embd"], bias=cfg["bias"])
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = LayerNorm(cfg["n_embd"], cfg["bias"])
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg["n_embd"], cfg["bias"])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg["block_size"]
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg["vocab_size"], cfg["n_embd"]),
            wpe=nn.Embedding(cfg["block_size"], cfg["n_embd"]),
            drop=nn.Dropout(cfg["dropout"]),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])]),
            ln_f=LayerNorm(cfg["n_embd"], cfg["bias"]),
        ))
        self.lm_head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.block_size, f"sequence {T} > block_size {self.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)  # logits: (B, T, vocab_size)

    @classmethod
    def load(cls, ckpt_path="ode.pt", device="cpu"):
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = cls(ck["model_config"]).to(device)
        model.load_state_dict(ck["model"], strict=True)
        model.eval()
        return model, ck


if __name__ == "__main__":
    model, ck = GPT.load("ode.pt")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded OK | params={n_params/1e6:.1f}M | config={ck['model_config']}")
