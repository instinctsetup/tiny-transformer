# tiny-transformer

**A GPT-style transformer with every layer hand-written - trained on 1,682 real movie titles, generating new ones in the browser with a live attention heatmap.**

- **Demo video**: https://drive.google.com/file/d/1kQ1AVViySFOTVm9tbWrDeAgUoHJUIsi4/view
- **Project report (PDF)**: https://drive.google.com/file/d/1nTvdUFR6VuHnrrD1J64SKYY3A8ZztyDd/view


Most "I built a transformer" projects call `nn.Transformer`. This one implements multi-head causal self-attention, layer norm, positional embeddings and the residual stack directly on tensors (PyTorch is used for tensors + autograd only). Everything is character-level and small enough (~130k parameters) to read in one sitting and train on a laptop CPU in minutes.

## What you can do with it

- Type an optional seed (`The`), drag the temperature slider, press **Generate** - a new "movie title" appears.
- Two heatmaps show the mean attention of layer 1 and layer 2 over the final sequence: you can literally see the model looking back at the beginning of words while finishing them.
- `docs/MATH.md` derives attention and the training loss in plain words; `docs/VIVA.md` rehearses the questions.

## Quick start

```bash
pip install -r requirements.txt
python train.py                 # ~3 min on CPU; writes tinytransformer/tinygpt.pt + vocab.txt
uvicorn server:app --port 8000  # open http://localhost:8000
```

A trained checkpoint is committed, so `uvicorn server:app` works immediately.

## How it works (60-second version)

1. Each character becomes a learned 64-dim vector; positions get their own learned vectors; they add.
2. Two transformer blocks. Each block: (a) every position asks every *earlier* position "how relevant are you to me?" - that's the QK^T attention score, scaled by sqrt(d), masked above the diagonal, softmaxed into weights; (b) weights mix the value vectors; (c) a small MLP thinks per-position. Residuals wrap both.
3. The final vector at each position is linearly mapped to vocab-size logits: a probability distribution over the next character.
4. Training: cross-entropy on "predict the next character" over random 48-char windows of real movie titles. Generation: sample one char, append, repeat; temperature sharpens/flattens the distribution; top-k truncates the tail.

## Honest notes

- PyTorch is used for tensors, autograd, `nn.Linear`, `nn.Embedding`, GELU and the AdamW optimizer. Attention, masking, layer norm, the residual structure, the sampling loop and the data pipeline are all hand-written and commented.
- Character-level was chosen for readability and train-on-CPU speed, not because it beats BPE - it doesn't.
- The corpus (MovieLens 100k titles) is small, so the model memorizes-ish; samples are charming, not Shakespeare. Training/validation loss is printed by `train.py` and stored in `docs/train-log.txt`.

## Layout

```
tinytransformer/model.py   LayerNorm, MultiHeadSelfAttention, Block, TinyGPT, generate()
tinytransformer/data.py    character vocab + batching
train.py                   training loop (~2500 steps, AdamW)
server.py                  FastAPI: /api/generate returns text + attention maps
frontend/                  generator UI + attention heatmaps (vanilla JS)
tests/                     causality, shapes, normalization, training-decreases-loss, API
docs/                      MATH.md, VIVA.md, train-log.txt
```
