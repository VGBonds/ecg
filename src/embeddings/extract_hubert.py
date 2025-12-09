import torch
from transformers import AutoModel
from pathlib import Path
from tqdm import tqdm
import config
import argparse

SPLITS=["train", "valid", "test"]


@torch.no_grad()
def extract_hubert_embeddings(
        signals_pt: Path,
        save_to: Path,
        model_size: str = "base",  # "small" | "base" | "large"
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    model = AutoModel.from_pretrained(f"Edoardo-BS/hubert-ecg-{model_size}", trust_remote_code=True)
    model.eval().to(device)

    signals = torch.load(signals_pt, map_location=device)  # (N,12,5000)
    print(f"Loaded {signals.shape}")

    embeddings = []
    for i in tqdm(range(0, len(signals), batch_size)):
        batch = signals[i:i + batch_size].to(device)
        # HuBERT-ECG expects (batch, time) → average over 12 leads (standard practice)
        batch = batch.mean(dim=1)  # (B,5000)
        hidden = model(batch, output_hidden_states=True).last_hidden_state
        pooled = hidden.mean(dim=1)  # (B, hidden_dim)
        embeddings.append(pooled.cpu())

    embeddings = torch.cat(embeddings)
    torch.save(embeddings, save_to)
    print(f"Saved HuBERT-{model_size} embeddings → {save_to} | shape {embeddings.shape}")


@torch.no_grad()
def extract_hubert_per_lead(
    signals_pt: Path,
    save_to: Path,
    model_size: str = "base",
    batch_size: int = 16,        # smaller because 12× larger
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    model = AutoModel.from_pretrained(f"Edoardo-BS/hubert-ecg-{model_size}", trust_remote_code=True)
    model.eval().to(device)

    signals = torch.load(signals_pt, map_location=device)   # (N, 12, 5000)
    N, L, T = signals.shape

    all_emb = []
    for i in tqdm(range(0, N, batch_size)):
        batch = signals[i:i+batch_size].to(device)          # (B, 12, 5000)
        B = batch.shape[0]
        batch = batch.reshape(B * L, T)                     # (B×12, 5000)

        hidden = model(batch, output_hidden_states=True).last_hidden_state
        pooled = hidden.mean(dim=1)                         # (B×12, 768)

        pooled = pooled.reshape(B, L, -1)                   # (B, 12, 768)
        all_emb.append(pooled.cpu())

    embeddings = torch.cat(all_emb, dim=0)                   # (N, 12, 768)
    torch.save(embeddings, save_to)
    print(f"Per-lead HuBERT embeddings saved: {embeddings.shape} → {save_to}")



if __name__ == "__main__":
    for split in SPLITS:
        parser = argparse.ArgumentParser(description="Extract HuBERT-ECG Embeddings")

        # provide Path defaults using Path(...)
        parser.add_argument(
            "--signals_pt",
            type=Path,
            default= config.processed_ptb_xl_data_folder / str(split + "_" + "signals.pt"),
            help="Path to input signals .pt file (default: %(default)s)"
        )
        parser.add_argument(
            "--save_to",
            type=Path,
            default=config.embeddings_folder / str("hubert_" + split + "_" + "embeddings.pt"),
            help="Path to save embeddings .pt file (default: %(default)s)"
        )
        parser.add_argument(
            "--model_size",
            type=str,
            default="large",
            choices=["small", "base", "large"],
            help="HuBERT-ECG model size (default: %(default)s)"
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=16,
            help="Batch size for extraction (default: %(default)s)"
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda" if torch.cuda.is_available() else "cpu",
            help="Device to use (default: %(default)s)"
        )

        args = parser.parse_args()

        extract_hubert_per_lead(
            signals_pt=args.signals_pt,
            save_to=args.save_to,
            model_size=args.model_size,
            batch_size=args.batch_size,
            device=args.device
        )

