# models/trainer.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, f1_score
from utils.dataloader import get_dataloaders
from tqdm import tqdm
import numpy as np

def train_head(
    train_emb, train_lbl,
    val_emb,   val_lbl,
    test_emb=None, test_lbl=None,
    head=None,
    epochs=50,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-4,
    patience=7,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_path="best_head.pt"
):
    head.to(device)
    opt = AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    # Class weights
    pos_counts = train_lbl.sum(dim=0)
    weights = len(train_lbl) / (5 * (pos_counts + 1e-6))
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights.to(device))

    # Proper DataLoaders (shuffle only on train)
    loaders = get_dataloaders(train_emb, train_lbl, val_emb, val_lbl,
                              test_emb, test_lbl, batch_size=batch_size)

    best_auc = 0.0
    no_improve = 0

    for epoch in range(epochs):
        head.train()
        epoch_loss = 0.0
        for x_batch, y_batch in loaders["train"]:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            opt.zero_grad()
            logits = head(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()

        # Validation
        head.eval()
        all_probs = []
        all_true  = []
        with torch.no_grad():
            for x_batch, y_batch in loaders["val"]:
                x_batch = x_batch.to(device)
                logits = head(x_batch)
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_true.append(y_batch.numpy())

        probs = np.vstack(all_probs)
        true  = np.vstack(all_true)
        auc   = roc_auc_score(true, probs, average="macro")
        f1    = f1_score(true, (probs > 0.5).astype(int), average="macro")

        print(f"Epoch {epoch+1:02d} • Loss {epoch_loss/len(loaders['train']):.4f} • Val AUC {auc:.4f} • F1 {f1:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(head.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping")
                break

    print(f"\nBest validation macro AUC: {best_auc:.4f}")
    return best_auc