import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, sosfiltfilt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# -------------------------------------------------------------------------
# 1. Load Data
# -------------------------------------------------------------------------
fname = r"M:\Neuromatch\project\joystick_track.npz"
alldat = np.load(fname, allow_pickle=True)["dat"]
dat = alldat[0][3]  # Subject 4 data

V = dat["V"].astype('float32')
cursorX = dat["cursorX"].flatten()
cursorY = dat["cursorY"].flatten()
targetX = dat["targetX"]
targetY = dat["targetY"]
fs = 1000
n_samples, n_channels = V.shape
time = np.arange(n_samples) / fs

# -------------------------------------------------------------------------
# 2. Clean CAR Montage (Excluding Extreme Variance Channels)
# -------------------------------------------------------------------------
channel_stds = np.std(V, axis=0)
median_std = np.median(channel_stds)
good_ch_mask = channel_stds < (3.0 * median_std)

print(f"Using {np.sum(good_ch_mask)} / {n_channels} channels for Common Average Reference.")
V_car = V - np.mean(V[:, good_ch_mask], axis=1, keepdims=True)

# -------------------------------------------------------------------------
# 3. Continuous Bandpass & Low-Pass Filtered Hilbert Envelopes
# -------------------------------------------------------------------------
bands = [
    (8, 12),     # Mu
    (18, 24),    # Beta
    (35, 42),    # Low Gamma
    (42, 70),    # Mid Gamma
    (70, 100),   # High Gamma 1
    (100, 140),  # High Gamma 2
    (140, 190)   # High Gamma 3
]

def extract_smoothed_envelope(data, low, high, fs, cutoff=10):
    # Bandpass filter using SOS format
    sos_band = butter(4, [low / (fs / 2), high / (fs / 2)], btype='band', output='sos')
    filtered = sosfiltfilt(sos_band, data, axis=0)
    
    # Hilbert amplitude envelope
    envelope = np.abs(hilbert(filtered, axis=0))
    
    # Low-pass envelope smoothing filter using SOS format
    sos_lp = butter(4, cutoff / (fs / 2), btype='low', output='sos')
    return sosfiltfilt(sos_lp, envelope, axis=0)

print("Extracting smoothed Hilbert envelopes across frequency bands...")
band_envelopes = []
for low, high in bands:
    band_envelopes.append(extract_smoothed_envelope(V_car, low, high, fs, cutoff=10))

# -------------------------------------------------------------------------
# 4. Feature Windowing (50 ms Causal Window, 25 ms Step Size)
# -------------------------------------------------------------------------
win_len = int(0.050 * fs)   # 50 ms window (causal)
step_len = int(0.025 * fs)  # 25 ms step size (40 Hz frame rate)

start_indices = np.arange(0, n_samples - win_len, step_len)
n_windows = len(start_indices)

# Extract timestamp at the END of the window (causal / no future lookahead)
feature_times = (start_indices + win_len) / fs

features = np.zeros((n_windows, n_channels, 8))

for w_idx, start in enumerate(start_indices):
    end = start + win_len
    for b_idx in range(7):
        features[w_idx, :, b_idx] = np.mean(band_envelopes[b_idx][start:end, :], axis=0)
    features[w_idx, :, 7] = np.mean(V_car[start:end, :], axis=0)  # LMP

# -------------------------------------------------------------------------
# 5. Directional Kinematics & Movement Masking
# -------------------------------------------------------------------------
dt = 1.0 / fs
vx = np.gradient(cursorX, dt)
vy = np.gradient(cursorY, dt)

smooth_win = int(0.050 * fs)
kernel = np.ones(smooth_win) / smooth_win
vx_smooth = np.convolve(vx, kernel, mode='same')
vy_smooth = np.convolve(vy, kernel, mode='same')

cursorX_aligned = np.interp(feature_times, time, cursorX)
cursorY_aligned = np.interp(feature_times, time, cursorY)
vx_aligned = np.interp(feature_times, time, vx_smooth)
vy_aligned = np.interp(feature_times, time, vy_smooth)

vx_plus = np.maximum(vx_aligned, 0)
vx_minus = np.maximum(-vx_aligned, 0)
vy_plus = np.maximum(vy_aligned, 0)
vy_minus = np.maximum(-vy_aligned, 0)
speed_aligned = np.sqrt(vx_aligned**2 + vy_aligned**2)
accel_aligned = np.abs(np.gradient(speed_aligned))

active_mask = speed_aligned > np.percentile(speed_aligned, 15)

X_position = np.column_stack([cursorX_aligned, cursorY_aligned])
X_velocity = np.column_stack([vx_aligned, vy_aligned])

# -------------------------------------------------------------------------
# 6. Log-Transform & Baseline Subtraction
# -------------------------------------------------------------------------
features_proc = features.copy()
features_proc[:, :, :7] = np.log10(np.maximum(features_proc[:, :, :7], 1e-6))

for feat_i in range(7):
    for ch in range(n_channels):
        band = features_proc[:, ch, feat_i]
        rest_baseline = np.median(band[~active_mask]) if np.sum(~active_mask) > 0 else np.median(band)
        features_proc[:, ch, feat_i] = band - rest_baseline

Y_features = features_proc.reshape(n_windows, -1)

# -------------------------------------------------------------------------
# 7. Automated Optimal Lag Selection on Movement Onsets
# -------------------------------------------------------------------------
# Extract High-Gamma 2 (100-140 Hz, band index 5)
hg_data = features_proc[:, :, 5]

# Isolate movement onset samples (top 10% acceleration steps)
speed_diff = np.diff(speed_aligned, prepend=speed_aligned[0])
onset_mask = speed_diff > np.percentile(speed_diff, 90)

# Identify channel with strongest correlation to speed on onsets
channel_corrs = [np.corrcoef(hg_data[onset_mask, ch], speed_aligned[onset_mask])[0, 1] for ch in range(n_channels)]
best_ch = np.nanargmax(channel_corrs)
hg_best_ch = hg_data[:, best_ch]

# Sweep lags from -500 ms (-20 steps) to +500 ms (+20 steps) at 25 ms resolution
lag_steps = np.arange(-20, 21)
cross_corrs = []

for l in lag_steps:
    shifted_speed = np.roll(speed_aligned, shift=l)
    cross_corrs.append(np.corrcoef(shifted_speed[onset_mask], hg_best_ch[onset_mask])[0, 1])

optimal_lag_step = lag_steps[np.argmax(cross_corrs)]
optimal_lead_ms = optimal_lag_step * 25
print(f"Optimal Peak Neural Lead: {optimal_lead_ms} ms (Step: {optimal_lag_step})")

# Build 5-tap lag window centered around the detected onset lag
lags_to_apply = np.arange(optimal_lag_step - 2, optimal_lag_step + 3)

def create_automated_lagged_matrix(X, lags):
    n_samples_len = len(X)
    lag_list = [np.roll(X, shift=l, axis=0) for l in lags]
    X_lagged = np.column_stack(lag_list)
    
    max_pos = max(0, np.max(lags))
    min_neg = abs(min(0, np.min(lags)))
    valid_mask = np.zeros(n_samples_len, dtype=bool)
    valid_mask[max_pos : n_samples_len - min_neg] = True
    return X_lagged, valid_mask

X_pos_lag, mask_pos = create_automated_lagged_matrix(X_position, lags_to_apply)
X_vel_lag, mask_vel = create_automated_lagged_matrix(X_velocity, lags_to_apply)

valid_active_pos = mask_pos & active_mask
valid_active_vel = mask_vel & active_mask

X_pos_active = X_pos_lag[valid_active_pos]
X_vel_active = X_vel_lag[valid_active_vel]
Y_active_pos = Y_features[valid_active_pos]
Y_active_vel = Y_features[valid_active_vel]

# -------------------------------------------------------------------------
# Feature Derivation: Acceleration and Euclidean Target Distance Error
# -------------------------------------------------------------------------
# Acceleration: Temporal derivative of velocity vector [N, 2]
X_acceleration = np.gradient(X_velocity, axis=0)

# Align target coordinates to causal feature timestamps via linear interpolation
targetX_aligned = np.interp(feature_times, time, targetX.flatten())
targetY_aligned = np.interp(feature_times, time, targetY.flatten())
target_position = np.column_stack([targetX_aligned, targetY_aligned])

# Compute causally-aligned Euclidean Target Distance Error [N, 1]
X_error = np.linalg.norm(target_position - X_position, axis=1, keepdims=True)

from sklearn.linear_model import Ridge

# -------------------------------------------------------------------------
# 8. Compute Out-of-Sample R² & SEM Across Lags + Track Top 3 Channels
# -------------------------------------------------------------------------
step_ms = 25
lag_steps = np.arange(-120, 121)  # -3000 ms to +3000 ms
lags_ms = lag_steps * step_ms

feature_labels = [
    '8-12Hz (Mu)', '18-24Hz (Beta)', '35-42Hz (Low Gamma)', 
    '42-70Hz (Mid Gamma)', '70-100Hz (HG1)', '100-140Hz (HG2)', 
    '140-190Hz (HG3)', 'LMP (<4 Hz)'
]

# Arrays to store Mean and SEM for 4 models
r2_pos_mean, r2_pos_sem = np.zeros((len(lag_steps), 8)), np.zeros((len(lag_steps), 8))
r2_vel_mean, r2_vel_sem = np.zeros((len(lag_steps), 8)), np.zeros((len(lag_steps), 8))
r2_acc_mean, r2_acc_sem = np.zeros((len(lag_steps), 8)), np.zeros((len(lag_steps), 8))
r2_err_mean, r2_err_sem = np.zeros((len(lag_steps), 8)), np.zeros((len(lag_steps), 8))

# Records list to store channel logs for CSV export
channel_records = []

# 1. Define contiguous temporal block splits on the FULL timeline FIRST
n_windows = len(X_position)
kf = KFold(n_splits=3, shuffle=False)
fold_time_splits = list(kf.split(np.arange(n_windows)))

model = Ridge(alpha=100.0)

print("Sweeping lags and logging top 3 channels per frequency band...")

for lag_i, shift in enumerate(lag_steps):
    X_pos_shifted = np.roll(X_position, shift=shift, axis=0)
    X_vel_shifted = np.roll(X_velocity, shift=shift, axis=0)
    X_acc_shifted = np.roll(X_acceleration, shift=shift, axis=0)
    X_err_shifted = np.roll(X_error, shift=shift, axis=0)
    
    valid_mask = np.ones(n_windows, dtype=bool)
    if shift > 0:
        valid_mask[:shift] = False
    elif shift < 0:
        valid_mask[shift:] = False
        
    # Active & non-boundary shifted timeline mask
    valid_active = valid_mask & active_mask
    
    r2_p_folds = np.zeros((kf.n_splits, n_channels, 8))
    r2_v_folds = np.zeros((kf.n_splits, n_channels, 8))
    r2_a_folds = np.zeros((kf.n_splits, n_channels, 8))
    r2_e_folds = np.zeros((kf.n_splits, n_channels, 8))
    
    # Iterate over the pre-defined contiguous physical time blocks
    for fold, (train_time_idx, test_time_idx) in enumerate(fold_time_splits):
        # Extract active indices within each contiguous time fold
        train_idx = train_time_idx[valid_active[train_time_idx]]
        test_idx  = test_time_idx[valid_active[test_time_idx]]
        
        # Pull data arrays using fold active indices
        X_p_tr, X_p_te = X_pos_shifted[train_idx], X_pos_shifted[test_idx]
        X_v_tr, X_v_te = X_vel_shifted[train_idx], X_vel_shifted[test_idx]
        X_a_tr, X_a_te = X_acc_shifted[train_idx], X_acc_shifted[test_idx]
        X_e_tr, X_e_te = X_err_shifted[train_idx], X_err_shifted[test_idx]
        Y_tr, Y_te     = Y_features[train_idx],    Y_features[test_idx]
        
        scaler_X_p = StandardScaler()
        scaler_X_v = StandardScaler()
        scaler_X_a = StandardScaler()
        scaler_X_e = StandardScaler()
        scaler_Y   = StandardScaler()
        
        X_p_tr, X_p_te = scaler_X_p.fit_transform(X_p_tr), scaler_X_p.transform(X_p_te)
        X_v_tr, X_v_te = scaler_X_v.fit_transform(X_v_tr), scaler_X_v.transform(X_v_te)
        X_a_tr, X_a_te = scaler_X_a.fit_transform(X_a_tr), scaler_X_a.transform(X_a_te)
        X_e_tr, X_e_te = scaler_X_e.fit_transform(X_e_tr), scaler_X_e.transform(X_e_te)
        Y_tr, Y_te     = scaler_Y.fit_transform(Y_tr),       scaler_Y.transform(Y_te)
        
        # Fit models
        model.fit(X_p_tr, Y_tr)
        Y_pred_p = model.predict(X_p_te)
        
        model.fit(X_v_tr, Y_tr)
        Y_pred_v = model.predict(X_v_te)
        
        model.fit(X_a_tr, Y_tr)
        Y_pred_a = model.predict(X_a_te)
        
        model.fit(X_e_tr, Y_tr)
        Y_pred_e = model.predict(X_e_te)
        
        for f_idx in range(Y_features.shape[1]):
            ch = f_idx // 8
            b = f_idx % 8
            r2_p_folds[fold, ch, b] = r2_score(Y_te[:, f_idx], Y_pred_p[:, f_idx])
            r2_v_folds[fold, ch, b] = r2_score(Y_te[:, f_idx], Y_pred_v[:, f_idx])
            r2_a_folds[fold, ch, b] = r2_score(Y_te[:, f_idx], Y_pred_a[:, f_idx])
            r2_e_folds[fold, ch, b] = r2_score(Y_te[:, f_idx], Y_pred_e[:, f_idx])
            

    # -------------------------------------------------------------------------
    # Process Top 3 Channels: Compute Fold-Level Mean/SEM + Log CSV Records
    # -------------------------------------------------------------------------
    models_info = [
        ('Position', r2_pos_mean, r2_pos_sem, r2_p_folds),
        ('Velocity', r2_vel_mean, r2_vel_sem, r2_v_folds),
        ('Acceleration', r2_acc_mean, r2_acc_sem, r2_a_folds),
        ('Euclidean Error', r2_err_mean, r2_err_sem, r2_e_folds)
    ]

    for model_name, mean_arr, sem_arr, fold_arr in models_info:
        # 1. Identify indices of top 3 channels per band (averaged across folds)
        ch_means = np.mean(fold_arr, axis=0)  # Shape: (n_channels, 8)
        top3_ch_indices = np.argsort(ch_means, axis=0)[-3:, :]  # Shape: (3, 8)

        # 2. Extract fold-level R² scores for these specific top 3 channels
        top3_folds = np.take_along_axis(
            fold_arr, top3_ch_indices[np.newaxis, :, :], axis=1
        )  # Shape: (n_folds, 3_channels, 8_bands)

        # 3. Average across top 3 channels FIRST to get 1 score per fold
        fold_scores = np.mean(top3_folds, axis=1)  # Shape: (n_folds, 8_bands)

        # 4. Compute Mean and SEM across CV FOLDS
        mean_arr[lag_i, :] = np.mean(fold_scores, axis=0)
        sem_arr[lag_i, :]  = np.std(fold_scores, axis=0) / np.sqrt(kf.n_splits)

        # 5. Log top 3 channel details into channel_records for CSV export
        for b_idx in range(8):
            ch_ranks = top3_ch_indices[:, b_idx][::-1]  # Rank 1 channel first
            ch_r2s = ch_means[ch_ranks, b_idx]
            
            channel_records.append({
                'Lag_ms': lags_ms[lag_i],
                'Model': model_name,
                'Frequency_Band': feature_labels[b_idx],
                'Top1_Channel': int(ch_ranks[0]),
                'Top1_R2': float(ch_r2s[0]),
                'Top2_Channel': int(ch_ranks[1]),
                'Top2_R2': float(ch_r2s[1]),
                'Top3_Channel': int(ch_ranks[2]),
                'Top3_R2': float(ch_r2s[2]),
                'Top3_Mean_R2': float(np.mean(ch_r2s))
            })
            
            
# Convert all logged records into a Pandas DataFrame
df_channels_all = pd.DataFrame(channel_records)

# -------------------------------------------------------------------------
# Export 1: Detailed CSV (All Lags)
# -------------------------------------------------------------------------
df_channels_all.to_csv('top3_channels_all_lags.csv', index=False)
print("Saved detailed channel logs to 'top3_channels_all_lags.csv'")

# -------------------------------------------------------------------------
# Export 2: Peak Lag Summary CSV (Most useful for paper/report)
# -------------------------------------------------------------------------
# Find the row corresponding to the Peak R² for each Model and Frequency Band
peak_indices = df_channels_all.groupby(['Model', 'Frequency_Band'])['Top3_Mean_R2'].idxmax()
df_peak_summary = df_channels_all.loc[peak_indices].reset_index(drop=True)

# Reorder columns logically
cols = [
    'Model', 'Frequency_Band', 'Lag_ms', 'Top3_Mean_R2',
    'Top1_Channel', 'Top1_R2', 'Top2_Channel', 'Top2_R2', 'Top3_Channel', 'Top3_R2'
]
df_peak_summary = df_peak_summary[cols]

df_peak_summary.to_csv('top3_channels_peak_summary.csv', index=False)
print("Saved peak lag channel summary to 'top3_channels_peak_summary.csv'")

# Display summary in notebook
print("\n--- Peak Lag Channel Selection Summary ---")
print(df_peak_summary.to_string(index=False))


# =========================================================================
# 8b. Save All Model Fit & Lag Sweep Results for Instant Re-plotting
# =========================================================================
import os

# Stack 2D arrays into unified 3D matrices for grid plotting
# Target Shape: (8_bands, 4_models, n_lags)
r2_means_grid = np.transpose(
    np.stack(
        [r2_pos_mean, r2_vel_mean, r2_acc_mean, r2_err_mean], axis=1
    ),  # (n_lags, 4_models, 8_bands)
    (2, 1, 0),  # (8_bands, 4_models, n_lags)
)

r2_sems_grid = np.transpose(
    np.stack(
        [r2_pos_sem, r2_vel_sem, r2_acc_sem, r2_err_sem], axis=1
    ),  # (n_lags, 4_models, 8_bands)
    (2, 1, 0),  # (8_bands, 4_models, n_lags)
)

model_names = np.array(
    ["Position", "Velocity", "Acceleration", "Euclidean Error"], dtype=str
)

save_filename = "model_fit_results_subject3.npz"

np.savez_compressed(
    save_filename,
    # 1. Unified 3D Grid Arrays (Shape: 8_bands, 4_models, n_lags)
    r2_means_grid=r2_means_grid,
    r2_sems_grid=r2_sems_grid,
    # 2. Individual Model Means (Shape: n_lags, 8_bands)
    r2_pos_mean=r2_pos_mean,
    r2_vel_mean=r2_vel_mean,
    r2_acc_mean=r2_acc_mean,
    r2_err_mean=r2_err_mean,
    # 3. Individual Model SEMs (Shape: n_lags, 8_bands)
    r2_pos_sem=r2_pos_sem,
    r2_vel_sem=r2_vel_sem,
    r2_acc_sem=r2_acc_sem,
    r2_err_sem=r2_err_sem,
    # 4. Axis & Metadata Labels
    lags_ms=lags_ms,
    feature_labels=np.array(feature_labels, dtype=str),
    model_names=model_names,
)

print(
    f"Successfully saved all model fit results to: {os.path.abspath(save_filename)}"
)

from scipy.ndimage import gaussian_filter1d

# -------------------------------------------------------------------------
# 9. Restored Unnormalized Raw R² Lag-Tuning Curves (Independent Y-Axes)
# -------------------------------------------------------------------------
feature_labels = [
    '8-12Hz (Mu)', '18-24Hz (Beta)', '35-42Hz (Low Gamma)', 
    '42-70Hz (Mid Gamma)', '70-100Hz (HG1)', '100-140Hz (HG2)', 
    '140-190Hz (HG3)', 'LMP (<4 Hz)'
]

fig, axes = plt.subplots(8, 4, figsize=(22, 18), sharex=True, dpi=100)

models_data = [
    (r2_pos_mean, r2_pos_sem, 'navy', 'Position Model'),
    (r2_vel_mean, r2_vel_sem, 'darkgreen', 'Velocity Model'),
    (r2_acc_mean, r2_acc_sem, 'crimson', 'Acceleration Model'),
    (r2_err_mean, r2_err_sem, 'purple', 'Euclidean Error Model')
]

# Set sigma_lags = 0.0 if you want completely raw, unfiltered lines
sigma_lags = 1.0 

for b_idx in range(8):
    for col_idx, (m_mean, m_sem, color, col_title) in enumerate(models_data):
        ax = axes[b_idx, col_idx]
        
        if sigma_lags > 0:
            mean_series = gaussian_filter1d(m_mean[:, b_idx], sigma=sigma_lags)
            sem_series  = gaussian_filter1d(m_sem[:, b_idx], sigma=sigma_lags)
        else:
            mean_series = m_mean[:, b_idx]
            sem_series  = m_sem[:, b_idx]
        
        peak_idx = np.argmax(mean_series)
        peak_lag = lags_ms[peak_idx]
        peak_val = mean_series[peak_idx]
        
        ax.plot(lags_ms, mean_series, color=color, linewidth=1.6)
        ax.fill_between(lags_ms, mean_series - sem_series, mean_series + sem_series, 
                        color=color, alpha=0.2, edgecolor='none')
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.4)
        ax.scatter(peak_lag, peak_val, color='red', s=20, zorder=5)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        ax.annotate(f'{peak_lag} ms', 
                    xy=(peak_lag, peak_val), 
                    xytext=(4, 2), textcoords='offset points', 
                    fontsize=7.5, color='darkred', fontweight='bold')
        
        if col_idx == 0:
            ax.set_ylabel(feature_labels[b_idx], fontweight='bold', fontsize=9)
            
        if b_idx == 0:
            ax.set_title(col_title, fontweight='bold', fontsize=11)

for col_idx in range(4):
    axes[-1, col_idx].set_xlabel('Neural Lead (+) / Lag (-) [ms]', fontweight='bold', fontsize=9)

plt.tight_layout()

plt.savefig("spectro_temporal_encoding_grid.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# -------------------------------------------------------------------------
# Variant 1: Column-Scaled Y-Axes (Shared Limits per Kinematic Model)
# -------------------------------------------------------------------------
fig, axes = plt.subplots(8, 4, figsize=(22, 18), sharex=True, dpi=100)

sigma_lags = 1.0

# 1. Pre-calculate global Y-limits per column (Model) across all 8 bands
col_ylims = []
for col_idx, (m_mean, m_sem, color, col_title) in enumerate(models_data):
    c_min, c_max = np.inf, -np.inf
    for b_idx in range(8):
        if sigma_lags > 0:
            m_s = gaussian_filter1d(m_mean[:, b_idx], sigma=sigma_lags)
            s_s = gaussian_filter1d(m_sem[:, b_idx], sigma=sigma_lags)
        else:
            m_s, s_s = m_mean[:, b_idx], m_sem[:, b_idx]

        c_min = min(c_min, np.min(m_s - s_s))
        c_max = max(c_max, np.max(m_s + s_s))

    pad = (c_max - c_min) * 0.05
    col_ylims.append((c_min - pad, c_max + pad))

# 2. Render subplots
for b_idx in range(8):
    for col_idx, (m_mean, m_sem, color, col_title) in enumerate(models_data):
        ax = axes[b_idx, col_idx]

        if sigma_lags > 0:
            mean_series = gaussian_filter1d(m_mean[:, b_idx], sigma=sigma_lags)
            sem_series = gaussian_filter1d(m_sem[:, b_idx], sigma=sigma_lags)
        else:
            mean_series = m_mean[:, b_idx]
            sem_series = m_sem[:, b_idx]

        peak_idx = np.argmax(mean_series)
        peak_lag = lags_ms[peak_idx]
        peak_val = mean_series[peak_idx]

        ax.plot(lags_ms, mean_series, color=color, linewidth=1.6)
        ax.fill_between(
            lags_ms,
            mean_series - sem_series,
            mean_series + sem_series,
            color=color,
            alpha=0.2,
            edgecolor="none",
        )
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.4)
        ax.scatter(peak_lag, peak_val, color="red", s=20, zorder=5)
        ax.grid(True, linestyle=":", alpha=0.6)

        # Apply column shared Y-limits
        ax.set_ylim(col_ylims[col_idx])

        ax.annotate(
            f"{peak_lag} ms",
            xy=(peak_lag, peak_val),
            xytext=(4, 2),
            textcoords="offset points",
            fontsize=7.5,
            color="darkred",
            fontweight="bold",
        )

        if col_idx == 0:
            ax.set_ylabel(
                feature_labels[b_idx], fontweight="bold", fontsize=9
            )

        if b_idx == 0:
            ax.set_title(col_title, fontweight="bold", fontsize=11)

for col_idx in range(4):
    axes[-1, col_idx].set_xlabel(
        "Neural Lead (+) / Lag (-) [ms]", fontweight="bold", fontsize=9
    )

plt.tight_layout()
plt.savefig(
    "spectro_temporal_encoding_column_scaled.png", dpi=300, bbox_inches="tight"
)
plt.show()
plt.close()


# -------------------------------------------------------------------------
# Variant 2: Row-Scaled Y-Axes (Shared Limits per Frequency Band Row)
# -------------------------------------------------------------------------
fig, axes = plt.subplots(8, 4, figsize=(22, 18), sharex=True, dpi=100)

sigma_lags = 1.0

# 1. Pre-calculate global Y-limits per row (Frequency Band) across all 4 models
row_ylims = []
for b_idx in range(8):
    r_min, r_max = np.inf, -np.inf
    for col_idx, (m_mean, m_sem, color, col_title) in enumerate(models_data):
        if sigma_lags > 0:
            m_s = gaussian_filter1d(m_mean[:, b_idx], sigma=sigma_lags)
            s_s = gaussian_filter1d(m_sem[:, b_idx], sigma=sigma_lags)
        else:
            m_s, s_s = m_mean[:, b_idx], m_sem[:, b_idx]

        r_min = min(r_min, np.min(m_s - s_s))
        r_max = max(r_max, np.max(m_s + s_s))

    pad = (r_max - r_min) * 0.05
    row_ylims.append((r_min - pad, r_max + pad))

# 2. Render subplots
for b_idx in range(8):
    for col_idx, (m_mean, m_sem, color, col_title) in enumerate(models_data):
        ax = axes[b_idx, col_idx]

        if sigma_lags > 0:
            mean_series = gaussian_filter1d(m_mean[:, b_idx], sigma=sigma_lags)
            sem_series = gaussian_filter1d(m_sem[:, b_idx], sigma=sigma_lags)
        else:
            mean_series = m_mean[:, b_idx]
            sem_series = m_sem[:, b_idx]

        peak_idx = np.argmax(mean_series)
        peak_lag = lags_ms[peak_idx]
        peak_val = mean_series[peak_idx]

        ax.plot(lags_ms, mean_series, color=color, linewidth=1.6)
        ax.fill_between(
            lags_ms,
            mean_series - sem_series,
            mean_series + sem_series,
            color=color,
            alpha=0.2,
            edgecolor="none",
        )
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.4)
        ax.scatter(peak_lag, peak_val, color="red", s=20, zorder=5)
        ax.grid(True, linestyle=":", alpha=0.6)

        # Apply row shared Y-limits
        ax.set_ylim(row_ylims[b_idx])

        ax.annotate(
            f"{peak_lag} ms",
            xy=(peak_lag, peak_val),
            xytext=(4, 2),
            textcoords="offset points",
            fontsize=7.5,
            color="darkred",
            fontweight="bold",
        )

        if col_idx == 0:
            ax.set_ylabel(
                feature_labels[b_idx], fontweight="bold", fontsize=9
            )

        if b_idx == 0:
            ax.set_title(col_title, fontweight="bold", fontsize=11)

for col_idx in range(4):
    axes[-1, col_idx].set_xlabel(
        "Neural Lead (+) / Lag (-) [ms]", fontweight="bold", fontsize=9
    )

plt.tight_layout()
plt.savefig(
    "spectro_temporal_encoding_row_scaled.png", dpi=300, bbox_inches="tight"
)
plt.show()
plt.close()


# -------------------------------------------------------------------------
# 10. Isolated LMP (<4 Hz) Band Plot Across Models (1x4 Subplot Row)
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(22, 4.5), sharex=True, dpi=100)

b_idx = 7  # Index corresponding to 'LMP (<4 Hz)'
sigma_lags = 1.0

for col_idx, (m_mean, m_sem, color, col_title) in enumerate(models_data):
    ax = axes[col_idx]

    if sigma_lags > 0:
        mean_series = gaussian_filter1d(m_mean[:, b_idx], sigma=sigma_lags)
        sem_series = gaussian_filter1d(m_sem[:, b_idx], sigma=sigma_lags)
    else:
        mean_series = m_mean[:, b_idx]
        sem_series = m_sem[:, b_idx]

    peak_idx = np.argmax(mean_series)
    peak_lag = lags_ms[peak_idx]
    peak_val = mean_series[peak_idx]

    ax.plot(lags_ms, mean_series, color=color, linewidth=2.0)
    ax.fill_between(
        lags_ms,
        mean_series - sem_series,
        mean_series + sem_series,
        color=color,
        alpha=0.2,
        edgecolor="none",
    )
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.4)
    ax.scatter(peak_lag, peak_val, color="red", s=35, zorder=5)
    ax.grid(True, linestyle=":", alpha=0.6)

    ax.annotate(
        f"{peak_lag} ms",
        xy=(peak_lag, peak_val),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=9,
        color="darkred",
        fontweight="bold",
    )

    ax.set_title(col_title, fontweight="bold", fontsize=12)
    ax.set_xlabel(
        "Neural Lead (+) / Lag (-) [ms]", fontweight="bold", fontsize=10
    )

    if col_idx == 0:
        ax.set_ylabel(
            f"Out-of-Sample $R^2$\n[{feature_labels[b_idx]}]",
            fontweight="bold",
            fontsize=11,
        )

plt.suptitle(
    "LMP (<4 Hz) Low-Frequency Encoding Profiles",
    fontsize=14,
    fontweight="bold",
    y=1.03,
)
plt.tight_layout()
plt.savefig("lmp_encoding_profiles.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("Saved isolated LMP plot to 'lmp_encoding_profiles.png'")


# -------------------------------------------------------------------------
# 11. Non-LMP Frequency Bands Overlaid (High-Contrast Custom Palette)
# -------------------------------------------------------------------------
import matplotlib.pyplot as plt

plt.close("all")

fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharex=True, dpi=100)

sigma_lags = 1.0
non_lmp_indices = range(7)  # Bands 0 to 6 (Excludes LMP at index 7)

# High-contrast, publication-grade color palette (Cool oscillations -> Warm High Gamma)
band_colors = [
    "#1f77b4",  # 8-12Hz (Mu): Deep Blue
    "#17becf",  # 18-24Hz (Beta): Cyan/Teal
    "#2ca02c",  # 35-42Hz (Low Gamma): Green
    "#e377c2",  # 42-70Hz (Mid Gamma): Soft Magenta
    "#ff7f0e",  # 70-100Hz (HG1): Vibrant Orange
    "#d62728",  # 100-140Hz (HG2): Red/Crimson
    "#8c564b",  # 140-190Hz (HG3): Deep Brown/Plum
]

for col_idx, (m_mean, _, _, col_title) in enumerate(models_data):
    ax = axes[col_idx]

    for b_idx in non_lmp_indices:
        if sigma_lags > 0:
            mean_series = gaussian_filter1d(m_mean[:, b_idx], sigma=sigma_lags)
        else:
            mean_series = m_mean[:, b_idx]

        color = band_colors[b_idx]
        label = feature_labels[b_idx]

        ax.plot(
            lags_ms,
            mean_series,
            color=color,
            linewidth=2.2,
            alpha=0.9,
            label=label,
        )

    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.4)
    ax.grid(True, linestyle=":", alpha=0.6)

    ax.set_title(col_title, fontweight="bold", fontsize=12)
    ax.set_xlabel(
        "Neural Lead (+) / Lag (-) [ms]", fontweight="bold", fontsize=10
    )

    if col_idx == 0:
        ax.set_ylabel("Out-of-Sample $R^2$", fontweight="bold", fontsize=11)
        ax.legend(
            loc="upper left",
            fontsize=8.5,
            framealpha=0.95,
            facecolor="white",
            edgecolor="none",
        )

plt.suptitle(
    "High-Frequency & Canonical Oscillation Encoding Profiles (Excluding LMP)",
    fontsize=14,
    fontweight="bold",
    y=1.03,
)
plt.tight_layout()
plt.savefig("non_lmp_encoding_profiles.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()