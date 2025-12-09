# models/trainer.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, f1_score
from utils.dataloader import get_dataloaders
from tqdm import tqdm
import numpy as np
from models.classification_head import ClassificationHead

def train_head(
    train_emb, train_lbl,
    val_emb,   val_lbl,
    test_emb=None, test_lbl=None,
    head=None,
    epochs=500,
    batch_size=128,
    lr=1e-2,
    weight_decay=1e-3,
    patience=1000,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_path="best_head.pt"
):
    head.to(device)
    opt = AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    # Add scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    # Class weights
    pos_counts = train_lbl.sum(dim=0)
    #weights = len(train_lbl) / (5 * (pos_counts + 1e-6))

    neg_counts = train_lbl.shape[0] - pos_counts  # Total samples minus positive samples
    pos_weight = neg_counts / (pos_counts + 1.)  # Calculate ratio
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    #criterion = nn.BCEWithLogitsLoss(pos_weight=weights.to(device))
    # criterion = nn.BCEWithLogitsLoss(pos_weight=weights.to(device))


    # Proper DataLoaders (shuffle only on train)
    loaders = get_dataloaders(train_emb, train_lbl, val_emb, val_lbl,
                              test_emb, test_lbl, batch_size=batch_size)

    best_auc = 0.0
    best_f1 = 0.0
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
        scheduler.step()

        # Validation
        head.eval()
        all_probs = []
        all_true  = []
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in loaders["val"]:
                x_batch = x_batch.to(device)
                logits = head(x_batch)
                v_loss = criterion(logits, y_batch.to(device))
                val_loss += v_loss.item()
                # # Skip all-negative samples for AUC/F1 computation
                # if np.sum(y_batch.numpy()) == 0:
                #     continue
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_true.append(y_batch.numpy())

        probs = np.vstack(all_probs)
        true  = np.vstack(all_true)
        roc_auc   = roc_auc_score(true, probs, average=None) #average="macro")
        print(f"Raw AUCs: {roc_auc}")
        auc = roc_auc_score(true, probs, average="micro")


        #f1 = f1_score(true, (probs > 0.5).astype(int), average="macro")
        f1 = f1_score(true, (probs > 0.5).astype(int), average=None)
        print(f"Raw F1s: {f1}")
        f1 = f1_score(true, (probs > 0.5).astype(int), average="macro")

        print(f"Epoch {epoch+1:02d} • Loss {epoch_loss/len(loaders['train']):.4f} • Val Loss {val_loss/len(loaders['val']):.4f} • Val AUC {auc:.4f} • F1 {f1:.4f}")

        #if auc > best_auc:
        if f1 > best_f1:
            best_f1 = f1
            best_auc = auc
            torch.save(head.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping")
                break

    print(f"\nBest validation macro F1: {best_f1:.4f} macro AUC: {best_auc:.4f}")

    # Optional test set evaluation
    if test_emb is not None and test_lbl is not None:
        head.load_state_dict(torch.load(save_path))
        head.to(device)
        head.eval()
        with torch.no_grad():
            test_probs = []
            test_true  = []
            for x_batch, y_batch in loaders["test"]:
                x_batch = x_batch.to(device)
                logits = head(x_batch)
                test_probs.append(torch.sigmoid(logits).cpu().numpy())
                test_true.append(y_batch.numpy())
            test_probs = np.vstack(test_probs)
            test_true  = np.vstack(test_true)
            test_auc = roc_auc_score(test_true, test_probs, average="macro")
            test_f1  = f1_score(test_true, (test_probs > 0.5).astype(int), average="macro")
            print(f"Test set macro AUC: {test_auc:.4f} • F1: {test_f1:.4f}")
            raw_auc = roc_auc_score(test_true, test_probs, average=None)
            raw_f1 = f1_score(test_true, (test_probs > 0.5).astype(int), average=None)
            print(f"Raw Test AUCs: {raw_auc}")
            print(f"Raw Test F1s: {raw_f1}")


    return best_auc