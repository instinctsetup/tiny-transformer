"""A GPT-style transformer, each piece written out by hand.

Deliberately small and readable: character-level, a few layers, causal
masked self-attention. Every tensor operation below maps to a line in
docs/MATH.md. We use PyTorch tensors + autograd only - no nn.Transformer,
no nn.MultiheadAttention, no nn.LayerNorm.
"""
import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.Module):
    """Normalize each token's features to mean 0 / std 1, then scale+shift.

    Written out instead of nn.LayerNorm so the formula is visible.
    """
    def __init__(self, n_embd):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_embd))
        self.beta = nn.Parameter(torch.zeros(n_embd))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + 1e-5) + self.beta


class MultiHeadSelfAttention(nn.Module):
    """Causal self-attention: every token may only look at earlier tokens.

    scores = QK^T / sqrt(d), the lower triangle is masked to -inf,
    softmax turns scores into weights, weights mix the values V.
    """
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)   # one fused projection
        self.proj = nn.Linear(n_embd, n_embd)
        # tril[i,j] = 1 when j <= i: the "no peeking at the future" mask
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, return_weights=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (B, T, C) -> (B, heads, T, head_dim): each head sees a slice
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / self.head_dim ** 0.5
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)            # rows sum to 1
        out = weights @ v                              # mix values
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        return (out, weights) if return_weights else out


class Block(nn.Module):
    """One transformer block: attention, then MLP, each wrapped in a
    residual connection with pre-normalization (the modern GPT layout)."""
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.attn = MultiHeadSelfAttention(n_embd, n_head, block_size)
        self.ln2 = LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x, return_weights=False):
        if return_weights:
            a, w = self.attn(self.ln1(x), return_weights=True)
            x = x + a                       # residual: gradient highway
            x = x + self.mlp(self.ln2(x))
            return x, w
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=2, block_size=48):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)   # learned positions
        self.blocks = nn.ModuleList(Block(n_embd, n_head, block_size) for _ in range(n_layer))
        self.ln_f = LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, return_weights=False):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        weights = []
        for block in self.blocks:
            if return_weights:
                x, w = block(x, return_weights=True)
                weights.append(w)
            else:
                x = block(x)
        logits = self.head(self.ln_f(x))            # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if return_weights:
            return logits, loss, weights
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=1.0, top_k=None):
        """Sample one character at a time; re-feed the growing sequence."""
        for _ in range(max_new):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:                    # keep only the k best guesses
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
            if idx[0, -1].item() == 0:               # newline = end of title
                break
        return idx
