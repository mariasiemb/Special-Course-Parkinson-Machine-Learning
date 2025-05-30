import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks

def process_imu(acc_path, pos_path, vel_path):
    acc = pd.read_csv(acc_path, skiprows=3, header=None)
    pos = pd.read_csv(pos_path, skiprows=3, header=None)
    vel = pd.read_csv(vel_path, skiprows=3, header=None)
    
    acc.columns = ['time_seconds', 'Ankle_acc']
    pos.columns = ['time_seconds', 'Ankle_pos']
    vel.columns = ['time_seconds', 'Ankle_vel']

    dt = np.diff(pos['time_seconds'])
    fs = 1 / np.mean(dt)
    print(f"Estimated sampling rate: {fs:.2f} Hz")

    # --- Estimate step length (in samples) ---
    step_samples = int(0.55 * fs)
    print(f"Estimated samples per step: {step_samples}")

    # Position zero-crossings
    zero_crossings = np.where(np.diff(np.sign(pos['Ankle_pos'])))[0]

    # Acceleration peaks (heel strikes / push-off impacts)
    peaks, _ = find_peaks(acc['Ankle_acc'], distance=step_samples)

    # Optional: if you care about minima (e.g. push-off)
    min_peaks, _ = find_peaks(-acc['Ankle_acc'], distance=step_samples)

    segments = []
    for i in range(len(peaks) - 1):
        segments.append((acc['time_seconds'].iloc[peaks[i]], acc['time_seconds'].iloc[peaks[i + 1]]))
    
    # Extract IMU features per segment
    imu_features = []
    for start, end in segments:
        seg = acc[(acc['time_seconds'] >= start) & (acc['time_seconds'] <= end)]
        rms = np.sqrt(np.mean(seg['Ankle_acc'] ** 2))
        mean = np.mean(seg['Ankle_acc'])
        std = np.std(seg['Ankle_acc'])
        imu_features.append([start, end, rms, mean, std])

    imu_df = pd.DataFrame(imu_features, columns=['Start', 'End', 'IMU_RMS', 'IMU_Mean', 'IMU_STD'])

    return segments, imu_df, fs, zero_crossings, min_peaks

print('IMU processing complete')
