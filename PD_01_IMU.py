import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks

# Load Ankle data

acc = pd.read_csv("C:/Users/s233183/OneDrive - Danmarks Tekniske Universitet/Desktop/Spring 2025/NRT-Lab SC/Data Collection/PD/PD_01/IMU/AnyBody_bvh/PD_01-T_6.ankle.acc(200-1000).csv", skiprows=3, header=None)
pos = pd.read_csv("C:/Users/s233183/OneDrive - Danmarks Tekniske Universitet/Desktop/Spring 2025/NRT-Lab SC/Data Collection/PD/PD_01/IMU/AnyBody_bvh/PD_01-T_6.ankle.pos(200-1000).csv", skiprows=3, header=None)
vel = pd.read_csv("C:/Users/s233183/OneDrive - Danmarks Tekniske Universitet/Desktop/Spring 2025/NRT-Lab SC/Data Collection/PD/PD_01/IMU/AnyBody_bvh/PD_01-T_6.ankle.vel(200-1000).csv", skiprows=3, header=None)

# Rename columns
acc.columns = ['time_seconds', 'Ankle_acc']
pos.columns = ['time_seconds', 'Ankle_pos']
vel.columns = ['time_seconds', 'Ankle_vel']

# Convert to float
acc['Ankle_acc'] = acc['Ankle_acc'].astype(float)
pos['Ankle_pos'] = pos['Ankle_pos'].astype(float)
vel['Ankle_vel'] = vel['Ankle_vel'].astype(float)

print('Data loaded')
print(acc.shape, pos.shape, vel.shape)

# --- Estimate Sampling Rate ---
dt = np.diff(pos['time_seconds'])
fs = 1 / np.mean(dt)
print(f"Estimated sampling rate: {fs:.2f} Hz")

# --- Estimate step length (in samples) ---
step_samples = int(0.55 * fs)
print(f"Estimated samples per step: {step_samples}")

# Position zero-crossings
zc_up = np.where((pos['Ankle_pos'].shift(1) < 0) & (pos['Ankle_pos'] >= 0))[0]
zc_down = np.where((pos['Ankle_pos'].shift(1) > 0) & (pos['Ankle_pos'] <= 0))[0]

# Acceleration peaks (heel strikes / push-off impacts)
peaks, _ = find_peaks(acc['Ankle_acc'], distance=step_samples)

# Optional: if you care about minima (e.g. push-off)
min_peaks, _ = find_peaks(-acc['Ankle_acc'], distance=step_samples)

# Identify gait events
toe_strike_indices = min_peaks  # Minima in acceleration
heel_strike_indices = peaks  # Peaks in acceleration
swing_times = []

for i in range(len(heel_strike_indices) - 1):
    swing_time = acc['time_seconds'].iloc[heel_strike_indices[i + 1]] - acc['time_seconds'].iloc[heel_strike_indices[i]]
    swing_times.append(swing_time)

# Segment signal based on gait events
segments = []
for i in range(len(heel_strike_indices) - 1):
    start_idx = heel_strike_indices[i]
    end_idx = heel_strike_indices[i + 1]
    segment = acc.iloc[start_idx:end_idx]
    segments.append(segment)

# Print identified parameters
print(f"Toe strikes (timestamps): {acc['time_seconds'].iloc[toe_strike_indices].values}")
print(f"Heel strikes (timestamps): {acc['time_seconds'].iloc[heel_strike_indices].values}")
print(f"Swing times (seconds): {swing_times}")
print(f"Number of segments: {len(segments)}")

# Plot results
plt.figure(figsize=(14, 6))
plt.plot(acc['time_seconds'], acc['Ankle_acc'], label='Ankle Acceleration', color='blue')
plt.plot(acc['time_seconds'].iloc[toe_strike_indices], acc['Ankle_acc'].iloc[toe_strike_indices], 'rx', label='Toe Strikes')
plt.plot(acc['time_seconds'].iloc[heel_strike_indices], acc['Ankle_acc'].iloc[heel_strike_indices], 'go', label='Heel Strikes')

# Visualize segmentation
for segment in segments:
    plt.axvspan(segment['time_seconds'].iloc[0], segment['time_seconds'].iloc[-1], color='orange', alpha=0.3)

plt.title('Gait Event Detection with Segmentation')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

feature_list = []

for segment in segments:
    signal = segment['Ankle_acc'].values
    time = segment['time_seconds'].values
    
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    rms_val = np.sqrt(np.mean(signal**2))
    ptp_val = np.ptp(signal)
    zero_crossings = np.where(np.diff(np.sign(signal)))[0].size
    min_val = np.min(signal)
    max_val = np.max(signal)
    duration = time[-1] - time[0]
    
    feature_list.append([mean_val, std_val, rms_val, ptp_val, zero_crossings, min_val, max_val, duration])

features_df = pd.DataFrame(feature_list, columns=['mean', 'std', 'rms', 'ptp', 'zero_crossings', 'min', 'max', 'duration'])

# Print the calculated features
print(features_df)