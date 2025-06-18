import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

def preprocess_emg(emg_df, fs):
    nyquist = fs / 2

    if fs < 250:
        print(f" EMG signal sampling rate ({fs} Hz) too low for standard 20–450 Hz band. Adjusting to 1–20 Hz.")
        low, high = 1 / nyquist, min(20 / nyquist, 0.99)
    else:
        low, high = 20 / nyquist, 450 / nyquist

    b_band, a_band = signal.butter(4, [low, high], btype='bandpass')
    b_notch, a_notch = signal.iirnotch(50.0, 30.0, fs)  # 50Hz notch
    b_low, a_low = signal.butter(4, 4 / nyquist, btype='lowpass')  # envelope smoothing

    for col in ['EMG_TA', 'EMG_GA']:
        emg_df[f'{col}_zeromean'] = emg_df[col] - np.mean(emg_df[col])
        emg_df[f'{col}_filtered'] = signal.filtfilt(b_band, a_band, emg_df[f'{col}_zeromean'])
        emg_df[f'{col}_notch'] = signal.filtfilt(b_notch, a_notch, emg_df[f'{col}_filtered'])
        emg_df[f'{col}_rectified'] = np.abs(emg_df[f'{col}_notch'])
        emg_df[f'{col}_envelope'] = signal.filtfilt(b_low, a_low, emg_df[f'{col}_rectified'])

    return emg_df


def process_emg(ta_path, ga_path, segments, fs):
    ta = pd.read_csv(ta_path, comment='#', header=None, delim_whitespace=True, names=['timestamp', 'EMG_TA'])
    ga = pd.read_csv(ga_path, comment='#', header=None, delim_whitespace=True, names=['timestamp', 'EMG_GA'])

    ta['time_seconds'] = (ta['timestamp'] - ta['timestamp'].iloc[0]) / 1000
    ga['time_seconds'] = (ga['timestamp'] - ga['timestamp'].iloc[0]) / 1000

    emg_df = pd.merge_asof(ta.sort_values('time_seconds'), ga.sort_values('time_seconds'), on='time_seconds', direction='nearest')
    emg_df = preprocess_emg(emg_df, fs)

    # Extract features per IMU segment
    emg_features = []
    for start, end in segments:
        seg = emg_df[(emg_df['time_seconds'] >= start) & (emg_df['time_seconds'] <= end)]
        for col in ['EMG_TA_envelope', 'EMG_GA_envelope']:
            if len(seg) == 0: continue
            rms = np.sqrt(np.mean(seg[col] ** 2))
            mav = np.mean(np.abs(seg[col]))
            iemg = np.sum(np.abs(seg[col]))
            var = np.var(seg[col])
            zc = ((seg[col][:-1] * seg[col][1:]) < 0).sum()
            muscle = col.split('_')[1]
            emg_features.append([start, end, muscle, rms, mav, iemg, var, zc])

    emg_features_df = pd.DataFrame(emg_features, columns=['Start', 'End', 'Muscle', 'RMS', 'MAV', 'IEMG', 'Variance', 'ZeroCrossings'])
    return emg_features_df

print('EMG processing complete')