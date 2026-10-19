"""Train TinyGPT on movie titles and save weights + vocab.

    python train.py            # ~2-3 minutes on CPU
"""
import torch
from tinytransformer.data import load_corpus, get_batch
from tinytransformer.model import TinyGPT

torch.manual_seed(42)
BLOCK, BATCH, STEPS, LR = 48, 64, 2500, 3e-4

vocab, train_data, val_data = load_corpus()
model = TinyGPT(vocab_size=vocab.size, n_embd=64, n_head=4, n_layer=2, block_size=BLOCK)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"params: {sum(p.numel() for p in model.parameters()):,} | vocab: {vocab.size}")

for step in range(STEPS):
    x, y = get_batch(train_data, BLOCK, BATCH)
    _, loss = model(x, y)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0 or step == STEPS - 1:
        with torch.no_grad():
            xv, yv = get_batch(val_data, BLOCK, BATCH)
            _, vloss = model(xv, yv)
        print(f"step {step:4d}  train {loss.item():.3f}  val {vloss.item():.3f}")

torch.save({"model": model.state_dict(), "config": {
    "vocab_size": vocab.size, "n_embd": 64, "n_head": 4, "n_layer": 2, "block_size": BLOCK}},
    "tinytransformer/tinygpt.pt")
vocab.save("tinytransformer/vocab.txt")

# Show a few generated titles as run evidence.
model.eval()
start = torch.zeros((1, 1), dtype=torch.long)
for temp in (0.8, 1.0):
    out = model.generate(start, 40, temperature=temp, top_k=10)
    print(f"sample (T={temp}):", repr(vocab.decode(out[0].tolist())))
