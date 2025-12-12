import torch
from transformers import AutoModel
import numpy as np
from pathlib import Path
from tqdm import tqdm
import config
import argparse
from tqdm import tqdm

# Load the model once
model_name = "Edoardo-BS/hubert-ecg-base"  # or small/large
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model.eval()  # Set to evaluation mode if just extracting embeddings

SPLITS=["train", "valid", "test"]

def get_embeddings_from_batch(signals_10s_batch, model):
    """
    Processes a batch of 10-second signals by splitting them into 5-second segments,
    embedding each, and averaging the results.
    """
    # Split the 10-second (1000 length) signal into two 5-second (500 length) segments
    # new shape becomes (Number_Of_Signals * 2, Number_Of_Leads, Length_5s)


    # 1. Split the 10-second signal in time first (along the time dimension, dim 2)
    # segment_1 shape: (B, 12, 500), segment_2 shape: (B, 12, 500)
    segment_1, segment_2 = torch.chunk(signals_10s_batch, chunks=2, dim=2) # (B, 12, 500), (B, 12, 500)

    # 2. Flatten each segment along the Leads and Time dimensions
    # New shape for each segment tensor: (B, 12 * 500) = (B, 6000)
    segment_1 = torch.flatten(segment_1, start_dim=1, end_dim=2) # (B, 6000)
    segment_2 = torch.flatten(segment_2, start_dim=1, end_dim=2) # (B, 6000)

    # Now we process all segments from all signals in one go by stacking them
    # new batch shape: (B*2, 12, 500)
    stacked_segments = torch.cat([segment_1, segment_2], dim=0) # (B*2, 6000)

    # Get embeddings
    with torch.no_grad():
        # Input shape to model MUST be (B, 6000)
        # segment_embeddings = model(stacked_segments,output_hidden_states=True).last_hidden_state
        segment_embeddings = model(stacked_segments).last_hidden_state # (B*2, 93, emb_dim=768)


    # Pool the sequence dimension (dim 1) to get a single vector per segment
    # Shape becomes (B*2, hidden_dim) e.g., (64, 768)
    pooled_segments = segment_embeddings.mean(dim=1) #(B*2, emb_dim=768)

    # Split the pooled embeddings back into original pairs
    # pooled_seg_1 shape (B, hidden_dim), pooled_seg_2 shape (B, hidden_dim)
    pooled_seg_1, pooled_seg_2 = torch.chunk(pooled_segments, chunks=2, dim=0) # (B, emb_dim=768), (B, emb_dim=768)

    # Average the embeddings from the two segments for each original signal
    # Final shape is (B, hidden_dim) e.g., (32, 768)
    final_embeddings = (pooled_seg_1 + pooled_seg_2) / 2 # (B, emb_dim=768)

    return pooled_seg_1, pooled_seg_2, final_embeddings

@torch.no_grad()
def extract_hubert_all_leads(
    signals_pt: Path,
    save_to: Path,
    model_size: str = "base",
    batch_size: int = 32,        # smaller because 12× larger
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    model = AutoModel.from_pretrained(f"Edoardo-BS/hubert-ecg-{model_size}", trust_remote_code=True)
    model.eval().to(device)

    signals = torch.load(signals_pt, map_location=device) # signals: (N, 12, 1000)
    N, L, T = signals.shape
    sec_5_cut = int(T/2) # todo: make this calc dependent on fs and length

    all_emb_1 = []
    all_emb_2 = []
    for i in tqdm(range(0, N, batch_size)):
        batch = signals[i:i+batch_size].to(device) # batch: (B, 12, 1000)
        pooled_1, pooled_2, _ = get_embeddings_from_batch(signals_10s_batch=batch,model=model)                # (B, 12, 768)
        # all_emb.append(pooled.cpu()) # List of (B, 768)
        all_emb_1.append(pooled_1)
        all_emb_2.append(pooled_2)

    # embeddings = torch.cat(all_emb, dim=0)
    embeddings_1 = torch.cat(all_emb_1, dim=0)
    embeddings_2 = torch.cat(all_emb_2, dim=0)
    # embeddings = (embeddings_1 + embeddings_2) / 2
    # torch.save(embeddings, save_to)
    torch.save(embeddings_1, save_to.with_name(save_to.stem + "_seg1.pt"))
    torch.save(embeddings_2, save_to.with_name(save_to.stem + "_seg2.pt"))
    print(f"Per-lead HuBERT embeddings saved: {embeddings_1.shape} → {save_to.with_name(save_to.stem + '_seg1.pt')}")
    print(f"Per-lead HuBERT embeddings saved: {embeddings_2.shape} → {save_to.with_name(save_to.stem + '_seg2.pt')}")
    # print(f"Per-lead HuBERT embeddings saved: {embeddings.shape} → {save_to}")



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
            default="base",
            choices=["small", "base", "large"],
            help="HuBERT-ECG model size (default: %(default)s)"
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=32,
            help="Batch size for extraction (default: %(default)s)"
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda" if torch.cuda.is_available() else "cpu",
            help="Device to use (default: %(default)s)"
        )

        args = parser.parse_args()

        # extract_hubert_per_lead(
        #     signals_pt=args.signals_pt,
        #     save_to=args.save_to,
        #     model_size=args.model_size,
        #     batch_size=args.batch_size,
        #     device=args.device
        # )

        extract_hubert_all_leads(
            signals_pt=args.signals_pt,
            save_to=args.save_to,
            model_size=args.model_size,
            batch_size=args.batch_size,
            device=args.device
        )
