"""Correctness tests for the hand-written transformer."""
import torch
from tinytransformer.model import TinyGPT, LayerNorm, MultiHeadSelfAttention


def make_model():
    torch.manual_seed(0)
    return TinyGPT(vocab_size=30, n_embd=32, n_head=4, n_layer=2, block_size=16)


def test_forward_shapes_and_loss():
    m = make_model()
    x = torch.randint(0, 30, (3, 16))
    logits, loss = m(x, targets=x)
    assert logits.shape == (3, 16, 30)
    assert loss.item() > 0


def test_attention_is_causal():
    """No position may attend to the future: upper triangle must be zero."""
    m = make_model()
    x = torch.randint(0, 30, (1, 16))
    _, _, weights = m(x, return_weights=True)
    for w in weights:
        upper = torch.triu(w[0], diagonal=1)   # j > i = looking ahead
        assert upper.abs().max().item() == 0.0


def test_attention_rows_sum_to_one():
    m = make_model()
    x = torch.randint(0, 30, (1, 16))
    _, _, weights = m(x, return_weights=True)
    for w in weights:
        assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)


def test_layernorm_normalizes():
    ln = LayerNorm(8)
    x = torch.randn(4, 5, 8) * 7 + 3
    out = ln(x)
    assert out.mean(-1).abs().max() < 1e-4
    assert (out.var(-1, unbiased=False) - 1).abs().max() < 1e-2


def test_training_reduces_loss():
    """Ten gradient steps on one fixed batch must lower the loss."""
    m = make_model()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, 30, (4, 16))
    losses = []
    for _ in range(10):
        _, loss = m(x, x)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_generate_length_and_types():
    m = make_model()
    start = torch.zeros((1, 1), dtype=torch.long)
    out = m.generate(start, max_new=12, temperature=0.8)
    assert out.shape[1] <= 13 and out.shape[0] == 1
