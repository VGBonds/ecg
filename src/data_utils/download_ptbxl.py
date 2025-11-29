# download_ptbxl_pure_python.py
import os
from pathlib import Path
import requests
from tqdm import tqdm  # pip install tqdm if you don't have it
import config

# download_ptbxl.py
import os
from pathlib import Path
import requests
from tqdm import tqdm
import time  # For polite delays

def download_file(url, target_path, max_retries=3):
    if target_path.exists():
        print(f"Already exists: {target_path.name}")
        return True
    for attempt in range(max_retries):
        try:
            print(f"Downloading {target_path.name} ... (attempt {attempt + 1})")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total = int(response.headers.get('content-length', 0))
            with open(target_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=target_path.name
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            return True
        except Exception as e:
            print(f"Error: {e}. Retrying in 5s...")
            time.sleep(5)
    return False

def download_ptbxl(data_dir="./ptb-xl", high_res_only=True):
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    base_url = "https://physionet.org/files/ptb-xl/1.0.3"

    # Key flat files (CSVs, txts, etc.)
    flat_files = [
        "LICENSE.txt",
        "RECORDS",
        "SHA256SUMS.txt",
        "example_physionet.py",
        "ptbxl_database.csv",
        "ptbxl_v102_changelog.txt",
        "ptbxl_v103_changelog.txt",
        "scp_statements.csv",
    ]

    for file in flat_files:
        download_file(f"{base_url}/{file}", data_dir / file)

    # Subfolders: Only records500/ for high-res (skip records100/ if high_res_only)
    subfolder = "records500" if high_res_only else "records100"
    subfolder_path = data_dir / subfolder
    subfolder_path.mkdir(exist_ok=True)

    # RECORDS file lists all ecg_ids; use it to build patient folders
    records_path = data_dir / "RECORDS"
    if not records_path.exists():
        print("RECORDS not found—run flat files first!")
        return

    with open(records_path, 'r') as f:
        ecg_ids = [line.strip() for line in f.readlines()]

    # Group by patient prefix (first 5 digits of ecg_id, per structure)
    from collections import defaultdict
    patient_groups = defaultdict(list)
    for ecg_id in tqdm(ecg_ids, desc="Grouping by patient"):
        patient_prefix = ecg_id[:5]  # e.g., '00001' from '00001_hr'
        patient_groups[patient_prefix].append(ecg_id)

    # Download per patient folder
    for patient_prefix, ecgs in tqdm(patient_groups.items(), desc=f"Downloading {subfolder}"):
        patient_dir = subfolder_path / patient_prefix
        patient_dir.mkdir(exist_ok=True)
        for ecg_id in ecgs:
            # Files: <ecg_id>_hr.dat/.hea (hr for 500Hz)
            suffix = "_hr" if high_res_only else "_lr"
            for ext in [".dat", ".hea"]:
                file_name = f"{ecg_id}{suffix}{ext}"
                download_file(f"{base_url}/{subfolder}/{patient_prefix}/{file_name}", patient_dir / file_name)

    print(f"PTB-XL ({'high-res' if high_res_only else 'full'}) ready in {data_dir.resolve()}!")
    print(f"Total size: Check with 'du -sh {data_dir}'")

if __name__ == "__main__":
    download_ptbxl(data_dir=config.ptb_xl_data_folder, high_res_only=True)
