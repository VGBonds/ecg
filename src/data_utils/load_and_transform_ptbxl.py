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
import config

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

# Challenge-style mappings
ACUTE_MI_CLASSES = ['AMI', 'ASMI', 'ALMI', 'INJAS', 'INJAL']  # acute only
RHYTHM_MAPS = {
    'AFIB': 'AFIB/AFL', 'AFLT': 'AFIB/AFL',
    'STACH': 'TACHY', 'SVTAC': 'TACHY', 'PSVT': 'TACHY',
    'SBRAD': 'BRADY',
    'PAC': 'PAC', 'PVC': 'PVC',
}

def get_diagnostic_superclass(code: str) -> str | None:
    return DIAGNOSTIC_MAPPING.get(code.strip().upper())

def derive_superclasses(scp_codes_list, scp_df=None):
    """Challenge-style: Use scp_statements if available, else derive."""
    superclasses = set()
    for code in scp_codes_list:
        if scp_df is not None and code in scp_df.index:
            superclass = scp_df.loc[code, 'diagnostic_class']
            if pd.notna(superclass):
                superclasses.add(superclass)
        # Fallback derive (for new CSV)
        else:
            sc = get_diagnostic_superclass(code)
            if sc:
                superclasses.add(sc)
    return list(superclasses)


def load_and_transform_ptbxl(
        data_root: Path,
        output_dir: Path,
        max_samples_per_split: int = None,
        target_sr: int = 500,
        clinical_mode: bool = False,  # Switch: True = force NORM=0 on abnormals + add RHYTHM
        new_challenge_mode: bool = False,  # Use https://github.com/physionetchallenges/python-example-2024/tree/main challenge logic
        scp_df_path=None,  # Optional old scp_statements with superclass col
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
    if scp_df_path is not None:
        scp_df = pd.read_csv(scp_df_path, index_col=0)  # For old mapping
        scp_df = scp_df[scp_df['diagnostic']==1] # Keep only diagnostic class
    else:
        scp_df = None

    CLASSES = ['NORM', 'Acute MI', 'Old MI', 'STTC', 'CD', 'HYP', 'PAC', 'PVC', 'AFIB/AFL', 'TACHY'] \
        if (clinical_mode or new_challenge_mode) else ['NORM', 'MI', 'STTC', 'CD', 'HYP']  # Simplified for now; expand as needed
    # CLASSES = ['NORM', 'Acute MI', 'Old MI', 'STTC', 'CD', 'HYP', 'PAC', 'PVC', 'AFIB/AFL', 'TACHY',
    #            'BRADY'] if (clinical_mode or new_challenge_mode) else ['NORM', 'MI', 'STTC', 'CD',
    #                                            'HYP']  # Simplified for now; expand as needed

    all_signals = {}
    all_meta    = {}
    for split_name, df in splits.items():
        print(f"\n=== Processing {split_name} ({len(df)} samples) | Clinical mode: {clinical_mode} ===")
        signals = []
        labels_numeric = []

        dropped = 0

        implicit_normal = []
        skipped_ecgs = []

        for ecg_id, row in tqdm(df.iterrows(), total=len(df)):
            # Load signal
            path = records_root / f"{(ecg_id // 1000) * 1000:05d}" / f"{ecg_id:05d}_hr"
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
            # signals.append(torch.from_numpy(sig))

            # Process labels
            scp_dict = eval(row["scp_codes"])
            positive_codes = [code for code, score in scp_dict.items() if score > 0]
            scp_codes_list = positive_codes

            superclasses = derive_superclasses(scp_codes_list, scp_df)
            # if not superclasses:
            #     print(f"Warning: No superclass found for ECG {ecg_id} with SCP codes {scp_codes_list}")
            # New Challenge logic
            superclasses_challenge = []
            if 'NORM' in scp_codes_list or 'NORM' in superclasses:
                superclasses_challenge.append('NORM')
            if any(c in scp_codes_list for c in ACUTE_MI_CLASSES):
                superclasses_challenge.append('Acute MI')
            elif 'MI' in superclasses:
                superclasses_challenge.append('Old MI')
            if 'STTC' in superclasses:
                superclasses_challenge.append('STTC')
            if 'CD' in superclasses:
                superclasses_challenge.append('CD')
            if 'HYP' in superclasses:
                superclasses_challenge.append('HYP')

            # Rhythm append (challenge-style)
            if 'PAC' in scp_codes_list:
                superclasses_challenge.append('PAC')
            if 'PVC' in scp_codes_list:
                superclasses_challenge.append('PVC')
            if 'AFIB' in scp_codes_list or 'AFLT' in scp_codes_list:
                superclasses_challenge.append('AFIB/AFL')
            if any(c in scp_codes_list for c in ['STACH', 'SVTAC', 'PSVT']):
                superclasses_challenge.append('TACHY')
            # Removed as there are 0 examples in PTB-XL
            # if 'SBRAD' in scp_codes_list:
            #     superclasses_challenge.append('BRADY')

            # All-negative
            if  clinical_mode and not superclasses_challenge:
                superclasses_challenge = ['NORM']
                implicit_normal.append(ecg_id)

            # Clinical mode: Force NORM=0 if abnormal
            if clinical_mode and any(l != 'NORM' for l in superclasses_challenge):
                superclasses_challenge = [l for l in superclasses_challenge if l != 'NORM']

            # One-hot (adjust CLASSES to match labels)
            onehot = np.zeros(len(CLASSES), dtype=np.float32)
            superclasses_to_use = superclasses_challenge if (new_challenge_mode or clinical_mode) else superclasses
            use_example = False
            for l in superclasses_to_use:
                if l in CLASSES:
                    onehot[CLASSES.index(l)] = 1.0
                    use_example = True
            if use_example:
                labels_numeric.append(onehot)
                signals.append(torch.from_numpy(sig))
            else:
                skipped_ecgs.append(ecg_id)
                dropped += 1

        # Save
        signals_tensor = torch.stack(signals)                     # (N, 12, 5000) or (N, 12, 1000)
        torch.save(signals_tensor, output_dir / f"{split_name}_signals.pt")

        successful_ids = df.index[:len(signals)]
        meta_df = df.loc[successful_ids].copy()
        meta_df["label_vector"] = labels_numeric
        # Save as JSON strings to preserve array structure in CSV
        meta_df["label_vector_string"] = [json.dumps(vec.tolist()) for vec in labels_numeric]

        meta_df.to_csv(output_dir / f"{split_name}_meta.csv")

        all_signals[split_name] = signals_tensor
        all_meta[split_name] = meta_df

        print(f"→ {split_name}: {len(signals)} saved | dropped {dropped} | implicit normal {len(implicit_normal)} | removed non-labeled {len(skipped_ecgs)}")

    # Stats
    print("\n=== Dataset Statistics ===")
    all_labels = np.vstack([all_meta[k]["label_vector"].tolist() for k in all_meta.keys()])
    counts = all_labels.sum(axis=0).astype(int)
    stats = {
        "total_samples": {k: len(v) for k, v in all_signals.items()},
        "classes": CLASSES,
        "label_counts": {c: int(n) for c, n in zip(CLASSES, counts)},
    }

    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Bar plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=CLASSES, y=counts)
    plt.title(f"PTB-XL Distribution (classes={'clinical' if clinical_mode else 'challenge'})")
    plt.ylabel("Number of ECGs")
    plt.savefig(output_dir / "label_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nAll done! Files in:", output_dir.resolve())
    print("Label counts:", stats["label_counts"])

    return all_signals, all_meta, stats


if __name__ == "__main__":
    # Benchmark mode (default)
    load_and_transform_ptbxl(
        data_root=config.ptb_xl_data_folder,
        output_dir=config.processed_ptb_xl_data_folder,
        max_samples_per_split=None,
        target_sr = 500,
        clinical_mode=False,
        new_challenge_mode=False,
        scp_df_path= config.ptb_xl_data_folder / "scp_statements.csv" #config.ptb_xl_data_folder / "scp_statements.csv",  None
    )

    # Clinical mode (uncomment when ready)
    # load_and_transform_ptbxl(..., include_rhythm_as_abnormal=True)