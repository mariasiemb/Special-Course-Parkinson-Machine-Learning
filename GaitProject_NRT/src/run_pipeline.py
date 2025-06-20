import pandas as pd
from imu_processing import process_imu
from emg_processing import process_emg
from feature_fusion import fuse_features
import os

all_data = []
patients = [('HC', 'HC_01'), ('HC', 'HC_03'), ('HC', 'HC_04'), ('HC', 'HC_05'), ('HC', 'HC_07'), ('HC', 'HC_11'), ('PD', 'PD_01'), ('PD', 'PD_03'),('PD', 'PD_04'),('PD', 'PD_05'),('PD', 'PD_06'),('PD', 'PD_07'),('PD', 'PD_11'),('PD', 'PD_13'),('PD', 'PD_15'),('PD', 'PD_16'),]  

for group, subject in patients:
    base_path = os.path.dirname(os.path.abspath(__file__))
    imu_folder = os.path.join(base_path, f'../Data/{group}/{subject}/IMU/')
    emg_folder = os.path.join(base_path, f'../Data/{group}/{subject}/EMG/')

    segments, imu_df, imu_fs, zero_crossings, min_peaks = process_imu(
        imu_folder + f"{subject}-T_6.ankle.acc(200-1000).csv",
        imu_folder + f"{subject}-T_6.ankle.pos(200-1000).csv",
        imu_folder + f"{subject}-T_6.ankle.vel(200-1000).csv"
    )

    emg_fs = 1000  

    emg_df = process_emg(
        emg_folder + "Tibialis_Anterior.txt",
        emg_folder + "Gastrocnemius.txt",
        segments, emg_fs
    )

    fused_df = fuse_features(imu_df, emg_df, subject, group)
    all_data.append(fused_df)


results_folder = os.path.join(base_path, '../results/')
os.makedirs(results_folder, exist_ok=True)

if all_data:  # Check if there's any data
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(os.path.join(results_folder, 'fused_features.csv'), index=False)
    print("Dataset ready for ML training.")
else:
    print("No data to save. Check if patients list or processing functions are working correctly.")
