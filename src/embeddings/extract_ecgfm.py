# embeddings/extract_ecgfm.py
import torch
from fairseq import checkpoint_utils
from pathlib import Path
from tqdm import tqdm
import numpy as np

@torch.no_grad()
def extract_ecgfm_embeddings(
    signals_pt: Path,
    save_to: Path,
    batch_size: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    # Official ECG-FM checkpoint (public)
    model_path = "https://huggingface.co/wanglab/ECG-FM/resolve/main/ecg-fm.pt"
    models, _, task = checkpoint_utils.load_model_ensemble_and_task([model_path])
    model = models[0].eval().to(device)

    signals = torch.load(signals_pt, map_location="cpu")    # (N,12,5000)
    N = len(signals)

    embeddings = []
    for i in tqdm(range(0, N, batch_size)):
        batch = signals[i:i+batch_size]                     # (B,12,5000)
        batch = (batch - batch.mean(dim=(1,2), keepdim=True)) / (batch.std(dim=(1,2), keepdim=True) + 1e-8)
        batch = batch.to(device)

        # ECG-FM takes (batch, channels, time) → treat 12 leads as channels
        feat = model.extract_features(batch, padding_mask=None, mask=False)[0]
        pooled = feat.mean(dim=1)                           # (B, feat_dim)
        embeddings.append(pooled.cpu())

    embeddings = torch.cat(embeddings)
    torch.save(embeddings, save_to)
    print(f"Saved ECG-FM embeddings → {save_to} | shape {embeddings.shape}")