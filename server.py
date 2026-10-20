"""tiny-transformer web server: generate movie titles, watch attention.

Run:  uvicorn server:app --port 8000    then open http://localhost:8000
"""
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tinytransformer.data import CharVocab
from tinytransformer.model import TinyGPT

app = FastAPI(title="tiny-transformer")
FRONTEND = Path(__file__).parent / "frontend"

ckpt = torch.load(Path(__file__).parent / "tinytransformer" / "tinygpt.pt",
                  map_location="cpu", weights_only=False)  # safe only because tinygpt.pt is committed/trusted; revisit before making the path configurable
MODEL = TinyGPT(**ckpt["config"])
MODEL.load_state_dict(ckpt["model"])
MODEL.eval()
VOCAB = CharVocab.load(Path(__file__).parent / "tinytransformer" / "vocab.txt")


class GenerateRequest(BaseModel):
    seed: str = Field(default="", max_length=30)
    temperature: float = Field(default=0.9, gt=0.05, le=2.0)
    max_chars: int = Field(default=40, ge=5, le=100)
    top_k: int = Field(default=10, ge=1, le=60)


@app.get("/api/health")
def health():
    return {"status": "ok", "vocab": VOCAB.size,
            "params": sum(p.numel() for p in MODEL.parameters())}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    """Generate a title, and return the attention maps over the final sequence
    so the frontend can paint what the model 'looked at'."""
    prompt = "\n" + req.seed if req.seed else "\n"
    try:
        ids = torch.tensor([VOCAB.encode(prompt)], dtype=torch.long)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    out = MODEL.generate(ids, req.max_chars, temperature=req.temperature, top_k=req.top_k)
    text = VOCAB.decode(out[0].tolist())

    with torch.no_grad():
        seq = out[:, -MODEL.block_size:]
        _, _, weights = MODEL(seq, return_weights=True)
    # Average heads per layer -> (layers, T, T) nested lists, rounded for JSON.
    attn = [w[0].mean(0).round(decimals=3).tolist() for w in weights]
    return {"text": text.replace("\n", " ⏎ ").strip(" ⏎"),
            "sequence": VOCAB.decode(seq[0].tolist()),
            "attention": attn, "n_layers": len(attn)}


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")

app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
