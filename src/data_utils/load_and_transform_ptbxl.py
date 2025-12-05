# data/load_and_transform_ptbxl.py
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import wfdb
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# 2025 GOLD-STANDARD DIAGNOSTIC SUPERCLASS MAPPING
# ------------------------------------------------------------------
DIAGNOSTIC_MAPPING = {
    # MI Myocardial Infarction
    'IMI': 'MI', 'ASMI': 'MI', 'AMI': 'MI', 'ALMI': 'MI', 'ILMI': 'MI',
    'LMI': 'MI', 'IPLMI': 'MI', 'IPMI': 'MI', 'PMI': 'MI',
    'INJAS': 'MI', 'INJAL': 'MI', 'INJIN': 'MI', 'INJLA': 'MI', 'INJIL': 'MI',
    'QWAVE': 'MI',

    # STTC ST/T Changes
    'NDT': 'STTC', 'NST_': 'STTC', 'DIG': 'STTC', 'LNGQT': 'STTC',
    'ISC_': 'STTC', 'ISCAL': 'STTC', 'ISCAN': 'STTC', 'ISCAS': 'STTC',
    'ISCIL': 'STTC', 'ISCIN': 'STTC', 'ISCLA': 'STTC', 'ANEUR': 'STTC',
    'EL': 'STTC', 'STD_': 'STTC', 'STE_': 'STTC', 'NT_': 'STTC',
    'TAB_': 'STTC', 'INVT': 'STTC', 'LOWT': 'STTC',

    # CD Conduction Disturbance
    'LAFB': 'CD', 'IRBBB': 'CD', '1AVB': 'CD', 'IVCD': 'CD',
    'CRBBB': 'CD', 'CLBBB': 'CD', 'LPFB': 'CD', 'WPW': 'CD',
    'ILBBB': 'CD', '_AVB': 'CD', '3AVB': 'CD', '2AVB': 'CD',
    'LPR': 'CD', 'ABQRS': 'CD',

    # HYP Hypertrophy
    'LVH': 'HYP', 'LAO/LAE': 'HYP', 'RVH': 'HYP', 'RAO/RAE': 'HYP',
    'SEHYP': 'HYP', 'VCLVH': 'HYP',

    # NORM Explicitly normal
    'NORM': 'NORM',
}

# Rhythm disorders that are clinically critical but NOT part of the 5-class diagnostic task
RHYTHM_DISORDERS = {
    'AFIB', 'AFLT', 'SVTAC', 'PSVT', 'PVC', 'BIGU', 'TRIGU',
    'SARRH', 'SBRAD', 'STACH', 'SVARR', 'PACE'
}

def get_diagnostic_superclass(code: str) -> str | None:
    return DIAGNOSTIC_MAPPING.get(code.strip().upper())

def load_and_transform_ptbxl(
    data_root: Path,
    output_dir: Path,
    max_samples_per_split: int = None,
    target_sr: int = 500,
    include_rhythm_as_abnormal: bool = False,   # ← clinical switch
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

    # Dynamic class list
    BASE_CLASSES = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
    CLASSES = BASE_CLASSES + ['RHYTHM'] if include_rhythm_as_abnormal else BASE_CLASSES

    all_signals = {}
    all_meta    = {}

    for split_name, df in splits.items():
        print(f"\n=== Processing {split_name} ({len(df)} samples) | Rhythm flag: {include_rhythm_as_abnormal} ===")
        signals = []
        labels  = []

        dropped = 0
        for ecg_id, row in tqdm(df.iterrows(), total=len(df)):
            # Load signal
            path = records_root / f"{(ecg_id//1000)*1000:05d}" / f"{ecg_id:05d}_hr"
            try:
                sig, _ = wfdb.rdsamp(str(path))
                sig = sig.T.astype(np.float32)
            except Exception as e:
                print(f"Failed to load {ecg_id}: {e}")
                dropped += 1
                continue

            # Optional downsample to 100 Hz
            if target_sr == 100:
                from scipy.signal import decimate
                sig = np.stack([decimate(lead, 5) for lead in sig])

            # Z-normalize per lead
            sig = (sig - sig.mean(axis=1, keepdims=True)) / (sig.std(axis=1, keepdims=True) + 1e-8)
            signals.append(torch.from_numpy(sig))

            # Parse labels
            scp_dict = eval(row["scp_codes"])
            positive_codes = [code for code, score in scp_dict.items() if score > 0]

            diagnostic_superclasses = []
            has_rhythm = False

            for code in positive_codes:
                sc = get_diagnostic_superclass(code)
                if sc:
                    diagnostic_superclasses.append(sc)
                if code in RHYTHM_DISORDERS:
                    has_rhythm = True

            diagnostic_superclasses = list(set(diagnostic_superclasses))

            # Enforce NORM exclusivity
            if 'NORM' in diagnostic_superclasses and len(diagnostic_superclasses) > 1:
                diagnostic_superclasses.remove('NORM')
            if not diagnostic_superclasses and 'NORM' not in {c.upper() for c in positive_codes}:
                diagnostic_superclasses = ['NORM']

            final_classes = diagnostic_superclasses.copy()
            if include_rhythm_as_abnormal and has_rhythm:
                final_classes.append('RHYTHM')

            onehot = np.zeros(len(CLASSES), dtype=np.float32)
            for cls in final_classes:
                if cls in CLASSES:
                    onehot[CLASSES.index(cls)] = 1.0
            labels.append(onehot)

        # Save

        signals_tensor = torch.stack(signals)                     # (N, 12, 5000) or (N, 12, 1000)
        torch.save(signals_tensor, output_dir / f"{split_name}_signals.pt")

        successful_ids = df.index[:len(signals)]
        meta_df = df.loc[successful_ids].copy()
        meta_df["label_vector"] = labels
        meta_df.to_csv(output_dir / f"{split_name}_meta.csv")

        all_signals[split_name] = signals_tensor
        all_meta[split_name] = meta_df

        print(f"→ {split_name}: {len(signals)} saved | dropped {dropped}")

    # Stats
    all_labels = np.vstack([all_meta[k]["label_vector"].tolist() for k in all_meta])
    counts = all_labels.sum(axis=0).astype(int)
    stats = {
        "total_samples": {k: len(v) for k, v in all_signals.items()},
        "classes": CLASSES,
        "include_rhythm_as_abnormal": include_rhythm_as_abnormal,
        "label_counts": {c: int(n) for c, n in zip(CLASSES, counts)},
    }

    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Bar plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=CLASSES, y=counts)
    plt.title(f"PTB-XL Distribution (Rhythm={'ON' if include_rhythm_as_abnormal else 'OFF'})")
    plt.ylabel("Number of ECGs")
    plt.savefig(output_dir / "label_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nAll done! Files in:", output_dir.resolve())
    print("Label counts:", stats["label_counts"])

    return all_signals, all_meta, stats


if __name__ == "__main__":
    # Benchmark mode (default)
    load_and_transform_ptbxl(
        data_root=Path("/path/to/ptb-xl"),
        output_dir=Path("data/processed_ptbxl"),
        max_samples_per_split=1000,
        include_rhythm_as_abnormal=False,
    )

    # Clinical mode (uncomment when ready)
    # load_and_transform_ptbxl(..., include_rhythm_as_abnormal=True)