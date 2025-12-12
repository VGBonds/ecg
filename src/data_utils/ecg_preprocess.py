import numpy as np
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
import torch
import wfdb

# ... (Assume load_raw_data function from before is available) ...

FS_ORIGINAL = 500
FS_TARGET = 100
INPUT_SAMPLES = 500


def preprocess_ecg_correct(ecg_data):
    """
    Apply filtering, downsampling, and correct Min-Max normalization [-1, 1].
    ecg_data shape: (num_samples_original, num_leads) -> (1000, 12)
    """
    # 1. Filtering (Band-pass 0.05-47 Hz as per paper)
    nyquist = FS_ORIGINAL / 2
    low = 0.05 / nyquist
    high = 47 / nyquist
    b, a = signal.butter(3, [low, high], btype='band')
    filtered_ecg = signal.lfilter(b, a, ecg_data, axis=0)

    # 2. Downsampling to 100 Hz
    num_samples_target = int(filtered_ecg.shape[0] * FS_TARGET / FS_ORIGINAL)
    downsampled_ecg = signal.resample(filtered_ecg, num_samples_target, axis=0)  # Shape (500, 12)

    # 3. Normalization (Min-Max scaling to [-1, 1] range)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # Reshape to fit scaler, then reshape back
    # Note: This scales based on min/max of the *entire 5-second segment* across all leads
    original_shape = downsampled_ecg.shape
    normalized_ecg = scaler.fit_transform(downsampled_ecg.reshape(-1, 1)).reshape(original_shape)

    return normalized_ecg


# Use the function as follows:
# processed_ecg = preprocess_ecg_correct(raw_ecg)
# input_tensor = torch.tensor(processed_ecg.T, dtype=torch.float32).unsqueeze(0)

