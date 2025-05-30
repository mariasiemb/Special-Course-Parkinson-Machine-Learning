import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

def fuse_features(imu_df, emg_df, subject, group):
    imu_df['Subject'] = subject
    imu_df['Group'] = group

    # merge EMG with IMU based on start and end times
    fused_df = pd.merge(emg_df, imu_df, on=['Start', 'End'])
    fused_df['Subject'] = subject
    fused_df['Group'] = group
    return fused_df

print('Feature fusion complete')