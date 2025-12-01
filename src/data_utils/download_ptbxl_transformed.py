import kagglehub
import os
import shutil
import config

def download_ptbxl_transformed(data_dir="./ptb-xl-transformed"):

    # Download latest version
    path = kagglehub.dataset_download("khyeh0719/ptb-xl-dataset-reformatted")

    print("Path to downloaded dataset files:", path)
    print("Move the downloaded files to the target directory.")
    # move the downloaded file to dataset_cache_path
    os.makedirs(os.path.dirname(data_dir), exist_ok=True)
    shutil.move(path, data_dir)


if __name__ == "__main__":
    download_ptbxl_transformed(data_dir=config.ptb_xl_transformed_data_folder)