import torch
from pathlib import Path
import pandas as pd
from models.classification_head import ClassificationHead, LeadFusionHead, TransformerFusionHead, TransformerFusionHead_2
from models.trainer import train_head
from sklearn.metrics import roc_auc_score
import numpy as np
import json
import config

DATA_DIR = config.processed_ptb_xl_data_folder
EMBEDDINGS_DIR = config.embeddings_folder

# Load HuBERT embeddings + labels
train_emb = torch.load(EMBEDDINGS_DIR / "hubert_train_embeddings.pt")
val_emb   = torch.load(EMBEDDINGS_DIR / "hubert_valid_embeddings.pt")
test_emb  = torch.load(EMBEDDINGS_DIR / "hubert_test_embeddings.pt")

train_meta = pd.read_csv(DATA_DIR / "train_meta.csv")
train_meta["label_vector"] = train_meta["label_vector_string"].apply(lambda s: np.array(json.loads(s), dtype=np.float32))
train_meta.drop(columns=["label_vector_string"], inplace=True)
val_meta   = pd.read_csv(DATA_DIR / "valid_meta.csv")
val_meta["label_vector"] = val_meta["label_vector_string"].apply(lambda s: np.array(json.loads(s), dtype=np.float32))
val_meta.drop(columns=["label_vector_string"], inplace=True)
test_meta  = pd.read_csv(DATA_DIR / "test_meta.csv")
test_meta["label_vector"] = test_meta["label_vector_string"].apply(lambda s: np.array(json.loads(s), dtype=np.float32))
test_meta.drop(columns=["label_vector_string"], inplace=True)

train_lbl = torch.tensor(train_meta["label_vector"].tolist(), dtype=torch.float32)
val_lbl   = torch.tensor(val_meta["label_vector"].tolist(),   dtype=torch.float32)
test_lbl  = torch.tensor(test_meta["label_vector"].tolist(),  dtype=torch.float32)

if train_emb.ndim > 2:
    # Pool over leads if needed (mean over lead dimension)
    flat_input_dim = train_emb.shape[1] * train_emb.shape[2]
else:
    flat_input_dim = train_emb.shape[1]
emb_dim = train_emb.shape[-1]
print(f"Embedding dimension: {emb_dim}")

# 1. Linear probe (no hidden layers)
print("\n=== Linear probe ===")
linear_head = ClassificationHead(input_dim=flat_input_dim, hidden_dims=[])
train_head(train_emb, train_lbl,
           val_emb,   val_lbl,
           test_emb,  test_lbl,
           lr=2e-4,
           head=linear_head,
           batch_size=128,
           save_path="hubert_linear.pt")


# # 2. Small MLP
# print("\n=== MLP (256→128) ===")
# mlp_head = ClassificationHead(input_dim=flat_input_dim,
#                               hidden_dims=[512, 256, 128], dropout=0.5,
#                               num_classes=train_lbl.shape[1])
# train_head(train_emb, train_lbl,
#            val_emb,   val_lbl,
#            test_emb,  test_lbl,
#            batch_size=128,
#            lr=2e-4,
#            head=mlp_head,
#            save_path="hubert_mlp.pt")



# # 3 Conv Lead Fusion Head
# print("\n=== Conv Lead Fusion Head ===")
# LeadFusion_head = LeadFusionHead(emb_dim=emb_dim,
#                                  num_classes=train_lbl.shape[1],
#                                  dropout=0.25)
# train_head(train_emb, train_lbl,
#            val_emb,   val_lbl,
#            test_emb,  test_lbl,
#            batch_size=128,
#            lr=1e-3,
#            head=LeadFusion_head,
#            save_path="hubert_mlp.pt")


# # 4 Transformer Lead Fusion Head
# print("\n=== Transformer Lead Fusion Head ===")
# LeadFusion_head = TransformerFusionHead(emb_dim=emb_dim,
#                                         num_classes=train_lbl.shape[1],
#                                         dropout=0.5,
#                                         num_heads=8,
#                                         num_blocks=1,
#                                         attn_dropout=0.05,
#                                         input_dropout=0.25)
# train_head(train_emb, train_lbl,
#            val_emb,   val_lbl,
#            test_emb,  test_lbl,
#            batch_size=64,
#            lr=2e-4,
#            head=LeadFusion_head,
#            save_path="hubert_mlp.pt")

# # 5 yet another Transformer Lead Fusion Head
# print("\n=== Transformer Lead Fusion Head ===")
# head = TransformerFusionHead_2(
#     emb_dim=emb_dim,
#     hidden_dim=128,        # 192 or 256 — both work, 192 is faster/safer
#     heads=8,
#     dropout=0.25, #0.1
#     attn_dropout=0.05, #0.1
# )
# train_head(
#     train_emb, train_lbl,
#     val_emb, val_lbl,
#     test_emb, test_lbl,
#     head=head,
#     lr=2e-4,
#     weight_decay=1e-2,     # crucial with high-dim input
#     epochs=150,
#     patience=25,
# )




# # Optional: test set evaluation of the best model
# best_head = ClassificationHead(input_dim, hidden_dims=[256, 128])
# best_head.load_state_dict(torch.load("hubert_mlp.pt"))
# best_head.eval()
# with torch.no_grad():
#     test_prob = torch.sigmoid(best_head(test_emb.to(device))).cpu().numpy()
#     test_auc = roc_auc_score(test_lbl, test_prob, average="macro")
#     print(f"\nTEST SET macro AUC: {test_auc:.4f}")