# The math inside tiny-transformer, in plain words

## Embeddings

A character id is just an index into a learned table `E` (vocab x 64). Position `t` gets its own learned vector `P[t]`. Input to the network: `x = E[char] + P[t]`. Both tables start random and are learned by gradient descent like everything else.

## Self-attention (the one idea that matters)

For each position we make three vectors from its embedding via learned matrices:

    q = x Wq   (query:  "what am I looking for?")
    k = x Wk   (key:    "what do I contain?")
    v = x Wv   (value:  "what do I pass on if selected?")

Compatibility between position i (asking) and j (being asked):

    score[i,j] = q[i] . k[j] / sqrt(d_head)

The division by sqrt(d_head) keeps the dot products from exploding as dimension grows (variance of a dot product grows with d).

**Causal mask:** position i may not look at j > i (no peeking at the future). We set those scores to -inf before the softmax, so they get weight exactly 0. `tests/test_attention_is_causal` proves it.

**Softmax:** `w[i,j] = exp(score[i,j]) / sum_j exp(score[i,j])` turns scores into weights that sum to 1.

**Output:** `out[i] = sum_j w[i,j] * v[j]` - a weighted blend of earlier positions' values.

**Multi-head:** we do this 4 times in parallel with different projections (each head sees 16 of the 64 dims), concatenate, and mix with one more linear layer. Different heads learn different habits (one tracks the start of the current word, another the title's opening).

## The block

    x = x + Attention(LayerNorm(x))
    x = x + MLP(LayerNorm(x))        MLP: Linear(64->256), GELU, Linear(256->64)

- **Residual (`x + ...`)**: the output is the input *plus a correction*. Gradients flow backwards through the "+" untouched, so deep stacks still train.
- **LayerNorm**: per position, subtract the feature mean, divide by the feature std, then apply learned scale `gamma` and shift `beta`. Keeps activations in a sane range.
- **GELU**: a smooth on/off curve, roughly `x * sigmoid(1.7x)` - lets the MLP be nonlinear, which is where the "thinking" happens.

## The training loss: cross-entropy

At each position the model outputs a score for every possible next character; softmax turns scores into probabilities `p`. If the true next character has probability `p*`, the loss is `-log p*`: 0 when the model is certain and right, large when it is confident and wrong. Averaged over all positions and the batch, minimized by AdamW (gradient descent with per-parameter adaptive step sizes).

## Generation: temperature and top-k

The final distribution is sampled, not argmaxed - sampling is why outputs vary. `logits / T` before the softmax: T < 1 sharpens (safe, repetitive), T > 1 flattens (wild). Top-k zeroes out all but the k most likely characters first, cutting off the long tail of nonsense.
