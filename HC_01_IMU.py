import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks

#Load Ankle data

acc = pd.read_csv("C:/Users/s233183/OneDrive - Danmarks Tekniske Universitet/Desktop/Spring 2025/NRT-Lab SC/Data Collection/HC/HC_01/IMU/AnyBody_bvh/HC_01-T_6.ankle.acc(200-1000).csv", skiprows=3, header=None)
pos = pd.read_csv("C:/Users/s233183/OneDrive - Danmarks Tekniske Universitet/Desktop/Spring 2025/NRT-Lab SC/Data Collection/HC/HC_01/IMU/AnyBody_bvh/HC_01-T_6.ankle.pos(200-1000).csv", skiprows=3, header=None)
vel = pd.read_csv("C:/Users/s233183/OneDrive - Danmarks Tekniske Universitet/Desktop/Spring 2025/NRT-Lab SC/Data Collection/HC/HC_01/IMU/AnyBody_bvh/HC_01-T_6.ankle.vel(200-1000).csv", skiprows=3, header=None)


#Rename columns
acc.columns = ['time_seconds', 'Ankle_acc']
pos.columns = ['time_seconds', 'Ankle_pos']
vel.columns = ['time_seconds', 'Ankle_vel']

#Convert to float
acc['Ankle_acc'] = acc['Ankle_acc'].astype(float)   
pos['Ankle_pos'] = pos['Ankle_pos'].astype(float)
vel['Ankle_vel'] = vel['Ankle_vel'].astype(float)

print('Data loaded')
print(acc.shape, pos.shape, vel.shape)
# Visualize the Ankle data
plt.figure(figsize=(12, 8))

# Plot Acceleration (acc)	
plt.subplot(3, 1, 1)
plt.plot(acc.iloc[:, 0], acc.iloc[:, 1], label='acc', color='r')
plt.title('Ankle _acc')
plt.xlabel('Time')
plt.ylabel('Force')
plt.grid(True)
plt.legend()

# Plot Position (pos)
plt.subplot(3, 1, 2)
plt.plot(pos.iloc[:, 0], pos.iloc[:, 1], label='pos', color='g')
plt.title('Ankle _pos')
plt.xlabel('Time')
plt.ylabel('Force')
plt.grid(True)
plt.legend()

# Plot Velocity (vel)
plt.subplot(3, 1, 3)
plt.plot(vel.iloc[:, 0], vel.iloc[:, 1], label='vel', color='b')
plt.title('Ankle _vel')
plt.xlabel('Time')
plt.ylabel('Force')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Detect peaks in Ankle_pos as example
#peaks, _ = find_peaks(pos['Ankle_pos'], distance=60)  # adjust distance based on your sampling rate and expected step length

# Plot to check event detection
#plt.plot(pos['time_seconds'], pos['Ankle_pos'])
#plt.plot(pos['time_seconds'].iloc[peaks], pos['Ankle_pos'].iloc[peaks], 'rx')
#plt.title('Gait Events (Position Peaks)')
#plt.show()

# --- Estimate Sampling Rate ---
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

# Plot both correctly
plt.figure(figsize=(14, 6))

plt.plot(pos['time_seconds'], pos['Ankle_pos'], label='Ankle Position')
plt.plot(pos['time_seconds'].iloc[zero_crossings], pos['Ankle_pos'].iloc[zero_crossings], 'ro', label='Zero-Crossings')

plt.plot(acc['time_seconds'], acc['Ankle_acc'] / 5, label='Ankle Acceleration (scaled)')
plt.plot(acc['time_seconds'].iloc[peaks], acc['Ankle_acc'].iloc[peaks] / 5, 'kx', label='Acc Peaks')
plt.plot(acc['time_seconds'].iloc[min_peaks], acc['Ankle_acc'].iloc[min_peaks] / 5, 'g^', label='Acc Minima')

plt.legend()
plt.title('Corrected Gait Event Detection')
plt.xlabel('Time (s)')
plt.grid()
plt.tight_layout()
plt.show()
# Visualize IMU Segmentation
plt.figure(figsize=(14, 6))
plt.plot(acc['time_seconds'], acc['Ankle_acc'], label='Ankle Acceleration', color='blue')

for i in range(len(zero_crossings) - 1):
    start_idx = zero_crossings[i]
    end_idx = zero_crossings[i + 1]
    plt.axvspan(acc['time_seconds'].iloc[start_idx], acc['time_seconds'].iloc[end_idx], color='orange', alpha=0.3, label='Segment' if i == 0 else "")

plt.title('IMU Segmentation Visualization')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
