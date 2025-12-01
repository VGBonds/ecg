# data/load_and_transform_ptbxl.py
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import wfdb
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import config

# ------------------------------------------------------------------
# 2025-DEFINITIVE SUPERCLASS MAPPING (used by ECG-FM, HuBERT-ECG, PULSE, etc.)
# ------------------------------------------------------------------
SUPERCLASS_MAPPING = {
    # Myocardial Infarction
    'IMI': 'MI', 'ASMI': 'MI', 'AMI': 'MI', 'ALMI': 'MI', 'ILMI': 'MI',
    'LMI': 'MI', 'IPLMI': 'MI', 'IPMI': 'MI', 'PMI': 'MI',
    'INJAS': 'MI', 'INJAL': 'MI', 'INJIN': 'MI', 'INJLA': 'MI', 'INJIL': 'MI',

    # ST/T Changes
    'NDT': 'STTC', 'NST_': 'STTC', 'DIG': 'STTC', 'LNGQT': 'STTC',
    'ISC_': 'STTC', 'ISCAL': 'STTC', 'ISCAN': 'STTC', 'ISCAS': 'STTC',
    'ISCIL': 'STTC', 'ISCIN': 'STTC', 'ISCLA': 'STTC', 'ANEUR': 'STTC',
    'EL': 'STTC', 'STD_': 'STTC', 'STE_': 'STTC', 'NT_': 'STTC',
    'TAB_': 'STTC', 'INVT': 'STTC', 'LOWT': 'STTC',

    # Conduction Disturbance
    'LAFB': 'CD', 'IRBBB': 'CD', '1AVB': 'CD', 'IVCD': 'CD',
    'CRBBB': 'CD', 'CLBBB': 'CD', 'LPFB': 'CD', 'WPW': 'CD',
    'ILBBB': 'CD', '_AVB': 'CD', '3AVB': 'CD', '2AVB': 'CD',

    # Hypertrophy
    'LVH': 'HYP', 'LAO/LAE': 'HYP', 'RVH': 'HYP', 'RAO/RAE': 'HYP',
    'SEHYP': 'HYP', 'VCLVH': 'HYP',

    # Explicitly normal
    'NORM': 'NORM',
}

def get_superclass(scp_code: str) -> str:
    """Map any SCP code → one of the 5 diagnostic superclasses."""
    code = scp_code.strip().upper()
    return SUPERCLASS_MAPPING.get(code, 'NORM')   # everything else → NORM (rhythm, PAC, SR, etc.)


def load_and_transform_ptbxl(
    data_root: Path = Path("./ptb-xl"),
    output_dir: Path = Path("./processed_ptbxl"),
    max_samples_per_split: int = None,   # Set to e.g. 1000 for quick tests, None for full
    target_sr: int = 500,                # 500 or 100
):
    output_dir.mkdir(exist_ok=True)
    records_root = data_root / "records500"

    # Load metadata
    db = pd.read_csv(data_root / "ptbxl_database.csv", index_col="ecg_id")

    # Official folds: 1–8 train, 9 valid, 10 test
    train_df = db[db.strat_fold.isin(range(1, 9))].copy()
    val_df   = db[db.strat_fold == 9].copy()
    test_df  = db[db.strat_fold == 10].copy()

    splits = {"train": train_df, "valid": val_df, "test": test_df}
    if max_samples_per_split:
        for k in splits:
            splits[k] = splits[k].head(max_samples_per_split)

    # Fixed order for multi-label binarization
    CLASSES = ['CD', 'HYP', 'MI', 'NORM', 'STTC']

    all_signals = {}
    all_meta    = {}

    for split_name, df in splits.items():
        print(f"\n=== Processing {split_name} ({len(df)} samples) ===")
        signals = []
        labels  = []

        for ecg_id, row in tqdm(df.iterrows(), total=len(df)):
            # ----- Load raw signal -----
            path = records_root / f"{ecg_id//1000:05d}" / f"{ecg_id:05d}_hr"
            try:
                sig, _ = wfdb.rdsamp(str(path))
                sig = sig.T.astype(np.float32)      # (12, 5000)
            except Exception as e:
                print(f"Failed to load {ecg_id}: {e}")
                continue

            # Optional downsample to 100 Hz
            if target_sr == 100:
                from scipy.signal import decimate
                sig = np.stack([decimate(lead, 5) for lead in sig])

            # Z-normalize per lead
            sig = (sig - sig.mean(axis=1, keepdims=True)) / (sig.std(axis=1, keepdims=True) + 1e-8)

            signals.append(torch.from_numpy(sig))

            # ----- Superclass labels -----
            scp_dict = eval(row["scp_codes"])                 # {'NORM': 100.0, 'LVH': 80.0, ...}
            superclasses = [get_superclass(code) for code in scp_dict.keys()]
            superclasses = list(set(superclasses))            # dedup

            # One-hot vector in fixed order
            onehot = np.zeros(len(CLASSES), dtype=np.float32)
            for sc in superclasses:
                if sc in CLASSES:
                    onehot[CLASSES.index(sc)] = 1.0
            labels.append(onehot)

        # Save
        signals_tensor = torch.stack(signals)                     # (N, 12, 5000) or (N, 12, 1000)
        torch.save(signals_tensor, output_dir / f"{split_name}_signals.pt")

        # meta = df.loc[signals_tensor.shape[0] * [None]]  # keep original index order
        # meta = meta.iloc[:len(signals)].copy()
        # meta["label_vector"] = labels
        # meta.to_csv(output_dir / f"{split_name}_meta.csv")

        # === Save meta with correct alignment ===
        # df still has the original index (ecg_id) and order
        # We may have skipped a few corrupted files → len(signals) ≤ len(df)
        successful_ecg_ids = df.index[:len(signals)]  # first N that succeeded
        meta_df = df.loc[successful_ecg_ids].copy()  # keep exact order + index

        meta_df["label_vector"] = labels  # labels is list of length N
        meta_df.to_csv(output_dir / f"{split_name}_meta.csv")

        all_meta[split_name] = meta_df

        print(f"→ {split_name}: {len(signals)} ECGs saved")

    # ------------------- Stats & plots -------------------
    stats = {
        "total_samples": {k: len(v) for k, v in all_signals.items()},
        "classes": CLASSES,
        "sampling_rate": target_sr,
    }

    # Label distribution
    all_labels = np.vstack([all_meta[k]["label_vector"].tolist() for k in all_meta])
    counts = all_labels.sum(axis=0).astype(int)
    stats["label_counts"] = {c: int(n) for c, n in zip(CLASSES, counts)}

    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Bar plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=CLASSES, y=counts)
    plt.title("PTB-XL Superclass Distribution")
    plt.ylabel("Number of ECGs")
    plt.savefig(output_dir / "label_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nAll done! Files in:", output_dir.resolve())
    print("Label counts:", stats["label_counts"])

    return all_signals, all_meta, stats


if __name__ == "__main__":
    # Quick test with 1000 samples each (remove limits for full run)
    load_and_transform_ptbxl(data_root=Path(config.ptb_xl_data_folder),
                             output_dir=Path(config.processed_ptb_xl_data_folder),
                             max_samples_per_split=1000)