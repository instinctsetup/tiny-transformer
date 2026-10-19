# Viva questions you should expect, with answers

**Q: What is attention, in one sentence?**
Every position computes a weighted average of earlier positions' value vectors, where the weights come from how well its query matches their keys.

**Q: Why divide by sqrt(d)?**
Dot products of d-dimensional vectors grow like sqrt(d) in magnitude; dividing keeps softmax inputs in a range where gradients don't vanish.

**Q: Why the causal mask?**
Without it the model could cheat by reading the character it is supposed to predict. The upper triangle is set to -inf so those weights become exactly zero after softmax.

**Q: Why multi-head rather than one big head?**
Four heads with 16 dims each can specialize - one can track "start of current word" while another tracks "title opening". Concatenating them is strictly more expressive than one head at the same total size.

**Q: What does layer norm actually do?**
Per token, it rescales the 64 features to mean 0, variance 1, then applies a learned scale and shift. It keeps activations well-conditioned so training is stable.

**Q: Why residual connections?**
They give gradients a direct path backwards (through the addition), preventing vanishing gradients in deep stacks, and let each block learn a *correction* instead of a full transformation.

**Q: Why character-level?**
Honesty and simplicity: no tokenizer black box, tiny vocab (~70), trains on CPU in minutes. Word/BPE-level models are stronger but harder to show end-to-end in a viva.

**Q: How is this different from an RNN?**
An RNN compresses the past into one hidden vector, sequentially. Attention lets every position look at *every* earlier position directly and in parallel - better long-range memory and parallelizable training.

**Q: What would you improve with more time?**
BPE tokenization, more data, dropout for regularization, learning-rate warmup + cosine decay, and KV-caching for faster generation.
