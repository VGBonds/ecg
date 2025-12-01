# scripts/ecg_fm_baseline.py
import os
import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
import wfdb
from pathlib import Path
import src.config as config # Assuming config has relevant paths if needed

# Step 1: Load PTB-XL Data
DATA_ROOT = Path(os.path.join(config.project_root , '/ptb-xl'))
train_df = pd.read_csv(DATA_ROOT / 'ptbxl_train.csv', index_col='ecg_id')
val_df = pd.read_csv(DATA_ROOT / 'ptbxl_valid.csv', index_col='ecg_id')
scp_statements = pd.read_csv(DATA_ROOT / 'scp_statements.csv', index_col=0)

# Superclass mapping (simplified: 5 classes)
scp_superclass = {row['SCP_ECG_STMT'].item(): row['superclass'] for idx, row in scp_statements.iterrows()}
superclass_to_int = {cls: idx for idx, cls in enumerate(scp_superclass.values())}
n_classes = len(superclass_to_int)

def load_ecg(ecg_id, sr=500):
    record = str(ecg_id).zfill(4)
    sig, fields = wfdb.rdsamp(DATA_ROOT / f'ptb-xl-a-large/{record}')
    return torch.from_numpy(sig.T).float()  # Shape: (12 leads, 5000 samples @500Hz/10s)

# Subset: First 1000 per split
train_ids = train_df.head(1000).index.tolist()
val_ids = val_df.head(1000).index.tolist()

# Labels: One-hot superclasses
def get_labels(df):
    labels = []
    for _, row in df.iterrows():
        stmt = row['scp_codes'].split(',')
        classes = [superclass_to_int[scp_superclass.get(s.strip(), 'NORM')] for s in stmt if s.strip() in scp_superclass]
        lbl = np.zeros(n_classes)
        for c in set(classes): lbl[c] = 1
        labels.append(lbl)
    return np.array(labels)

train_labels = get_labels(train_df.loc[train_ids])
val_labels = get_labels(val_df.loc[val_ids])

# Step 2: ECG-FM Embedding Extraction
os.system('pip install fairseq omegaconf')  # If needed
from fairseq import checkpoint_utils, tasks, utils
import hydra
from omegaconf import DictConfig, OmegaConf

# Load model (pretrained path from HF/GitHub)
MODEL_PATH = 'https://huggingface.co/wanglab/ecg-fm/resolve/main/pretrained.pt'  # Or local download
cp = checkpoint_utils.load_model_ensemble_and_task([MODEL_PATH])[0]
model = cp['args']
task = tasks.setup_task(model)
model = cp['models'][0]
model.eval()

# Inference config (from repo tutorial)
cfg = DictConfig({
    'task': {'data': './dummy', 'label_dir': './dummy'},  # Dummy for inference
    'model': {'model_path': MODEL_PATH},
    'dataset': {'sampling_rate': 500, 'task': 'speech_recognition'},
    'common': {'fp16': False},
    'generation': {'path': MODEL_PATH}
})
utils.import_user_module(cfg.common)

def extract_embeddings(ecg_tensor):  # ecg: (12, 5000)
    with torch.no_grad():
        # Resample/normalize if needed (per repo: 500Hz, z-norm per lead)
        ecg_norm = (ecg_tensor - ecg_tensor.mean(1, keepdim=True)) / (ecg_tensor.std(1, keepdim=True) + 1e-8)
        # Stack leads as batch (treat as multi-channel)
        ecg_input = ecg_norm.unsqueeze(0)  # (1, 12, 5000) -> treat as batched mono?
        # Fairseq expects (batch, time) or adapt for multi-lead
        embeddings = model.extract_features(ecg_input.mean(1), mask=False)[0]  # Pool leads for simplicity; shape (1, feat_dim)
        return embeddings.squeeze().cpu().numpy()  # Global pooled embedding

# Extract
print("Extracting train embeddings...")
train_embs = np.array([extract_embeddings(load_ecg(id)) for id in train_ids])
print("Extracting val embeddings...")
val_embs = np.array([extract_embeddings(load_ecg(id)) for id in val_ids])

np.save('ecg_fm_train_embs.npy', train_embs)
np.save('ecg_fm_val_embs.npy', val_embs)

# Step 3: Train simple head & evaluate
clf = MultiOutputClassifier(LogisticRegression(max_iter=1000))
clf.fit(train_embs, train_labels)
val_preds_proba = clf.predict_proba(val_embs)
val_preds = clf.predict(val_embs)

# Metrics (multi-label)
auc = roc_auc_score(val_labels, val_preds_proba, average='macro')
f1 = f1_score(val_labels, val_preds, average='macro')
print(f"ECG-FM Baseline: AUC={auc:.4f}, F1={f1:.4f}")

# Confusion (per class avg)
cm = confusion_matrix(val_labels.argmax(1), val_preds.argmax(1))
sns.heatmap(cm, annot=True, fmt='d')
plt.savefig('ecg_fm_cm.png')
plt.show()